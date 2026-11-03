"""Ingestion PDF end-to-end: parsing (testo + tabelle + OCR), chunking,
upload del binario originale su GridFS, insert dei chunk in MongoDB.

L'embedding NON viene calcolato qui: i chunk vengono inseriti come testo puro
e Atlas Vector Search (autoEmbed con Voyage) genera i vettori automaticamente.
"""
from __future__ import annotations

import hashlib
import base64
import io
import re
import math
import mimetypes
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import pytesseract
import tiktoken
from PIL import Image
from gridfs import GridFS
from langdetect import DetectorFactory, LangDetectException, detect
from pymongo.collection import Collection
from pymongo.database import Database

from . import config
from .embeddings import (
    embed_voyage,
    embed_voyage_contextualized,
)

DetectorFactory.seed = 0

if config.TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD

_TOKENIZER = tiktoken.get_encoding("cl100k_base")
_SCHEMA_VERSION = "pdf_chunks_v4"
# Pre-chunked input is limited to 32K tokens per request. We keep local chunk
# boundaries to preserve page/slide citations, so backend auto-chunking is off.
_CONTEXT_MAX_TOKENS = 32_000


# ---------------------------------------------------------------- helpers ----

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for blob in iter(lambda: f.read(8192), b""):
            h.update(blob)
    return h.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _without_empty_fields(document: dict[str, Any]) -> dict[str, Any]:
    """Rimuove i campi senza valore prima della scrittura su MongoDB."""
    empty_values = (None, "", [], {})
    return {
        key: value
        for key, value in document.items()
        if value not in empty_values
    }


def _validate_embeddings(
    embeddings: list[list[float]],
    *,
    expected_count: int,
    expected_dimensions: int,
) -> None:
    """Blocca l'ingestion se la risposta Voyage non è coerente con i chunk."""
    if len(embeddings) != expected_count:
        raise RuntimeError(
            "Numero embedding non coerente: "
            f"attesi {expected_count}, ricevuti {len(embeddings)}."
        )
    for index, vector in enumerate(embeddings):
        if len(vector) != expected_dimensions:
            raise RuntimeError(
                f"Embedding #{index} con dimensione {len(vector)}; "
                f"attesa {expected_dimensions}."
            )
        if not vector or not all(
            isinstance(value, (int, float)) and math.isfinite(value)
            for value in vector
        ):
            raise RuntimeError(f"Embedding #{index} vuoto o contenente valori non validi.")


def _safe_lang(text: str) -> str | None:
    if len(text.strip()) < 20:
        return None
    try:
        return detect(text)
    except LangDetectException:
        return None


def _extract_pdf_metadata(doc: fitz.Document) -> dict[str, Any]:
    raw = doc.metadata or {}
    return {
        "title": raw.get("title") or None,
        "author": raw.get("author") or None,
        "subject": raw.get("subject") or None,
        "keywords": raw.get("keywords") or None,
        "creator": raw.get("creator") or None,
        "producer": raw.get("producer") or None,
        "creation_date": raw.get("creationDate") or None,
        "modification_date": raw.get("modDate") or None,
        "total_pages": doc.page_count,
        "format": raw.get("format") or None,
    }


_OCR_WARNED = False


def _ocr_page(page: fitz.Page) -> str | None:
    """OCR di una pagina via Tesseract. Ritorna None se Tesseract non è disponibile."""
    global _OCR_WARNED
    try:
        pix = page.get_pixmap(dpi=200)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return pytesseract.image_to_string(img, lang="ita+eng") or ""
    except pytesseract.TesseractNotFoundError:
        if not _OCR_WARNED:
            print(
                "[ingestion] Tesseract non trovato: le pagine scansionate "
                "verranno saltate. Installa con `brew install tesseract tesseract-lang`."
            )
            _OCR_WARNED = True
        return None
    except Exception as e:
        print(f"[ingestion] OCR fallito: {e}")
        return None


# --------------------------------------------------------------- parsing ----

_MARKITDOWN_EXTS = {".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls",
                    ".txt", ".html", ".htm", ".csv", ".json", ".xml", ".md"}


def parse_with_markitdown(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Parsing per formati non-PDF tramite MarkItDown (Microsoft).
    Ritorna markdown a livello documento: non assegna una pagina fittizia.
    """
    from markitdown import MarkItDown
    md_converter = MarkItDown()
    result = md_converter.convert(str(path))
    text = result.text_content or ""
    meta = {
        "title": getattr(result, "title", None) or path.stem,
        "author": None, "subject": None, "keywords": None,
        "creator": "MarkItDown", "producer": "MarkItDown",
        "creation_date": None, "modification_date": None,
        "total_pages": None,
        "format": path.suffix.lstrip(".").upper(),
    }
    pages = [{
        "page_number": None, "text": text, "tables": [], "content_type": "text",
    }]
    return meta, pages


def _office_as_pdf(path: Path, out_dir: Path) -> Path:
    """Converte un file Office in PDF mantenendo pagine/slide e grafica."""
    executable = config.OFFICE_CONVERTER or "soffice"
    try:
        subprocess.run(
            [executable, "--headless", "--convert-to", "pdf", "--outdir", str(out_dir), str(path)],
            check=True, capture_output=True, text=True, timeout=180,
        )
    except Exception as exc:
        raise RuntimeError(
            "Per ingestion visiva di DOCX/PPTX serve LibreOffice (soffice) disponibile nel PATH."
        ) from exc
    converted = out_dir / f"{path.stem}.pdf"
    if not converted.is_file():
        raise RuntimeError(f"Conversione Office in PDF non riuscita per {path.name}.")
    return converted


def parse_document(
    path: Path, *, include_visuals: bool = False
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """PDF nativo; Office via PDF per mantenere slide/pagine, MarkItDown fallback."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        return parse_pdf(path, include_visuals=include_visuals)
    if ext in {".docx", ".doc", ".pptx", ".ppt"}:
        try:
            office_meta, office_pages = parse_with_markitdown(path)
            with tempfile.TemporaryDirectory(prefix="rag-office-") as tmp:
                converted = _office_as_pdf(path, Path(tmp))
                meta, pages = parse_pdf(converted, include_visuals=include_visuals)
                meta["title"] = office_meta.get("title") or meta.get("title") or path.stem
                meta["format"] = ext.lstrip(".").upper()
                meta["source_parser"] = "MarkItDown+LibreOffice+PyMuPDF"
                # MarkItDown preserva intestazioni e struttura Office; la resa PDF
                # fornisce il legame puntuale a pagina/slide e le immagini.
                if pages and office_pages and office_pages[0].get("text"):
                    pages[0]["office_structure_text"] = office_pages[0]["text"]
                return meta, pages
        except Exception:
            if include_visuals:
                raise
    if ext in _MARKITDOWN_EXTS:
        return parse_with_markitdown(path)
    raise ValueError(f"Estensione non supportata: {ext}")


def _clean_markdown(text: str) -> str:
    """Toglie la formattazione visiva (grassetti, corsivi, a capo HTML) e lascia la struttura."""
    text = text.replace("<br>", " ")
    text = re.sub(r"\*\*", "", text)
    text = re.sub(r"(?m)^(#+\s*)_(.+?)_\s*$", r"\1\2", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _pdf_markdown(doc: fitz.Document) -> list[str] | None:
    """Testo di ogni pagina in Markdown con pymupdf4llm: titoli e tabelle con le colonne.

    Le tabelle dei bilanci restano righe con intestazione (anni), invece di numeri in fila.
    None se pymupdf4llm non è installato o fallisce: si usa il testo semplice di PyMuPDF.
    """
    try:
        import pymupdf4llm
        chunks = pymupdf4llm.to_markdown(doc, page_chunks=True, use_ocr=False, header=False, footer=False,
                                         show_progress=False)
        return [_clean_markdown(c.get("text") or "") for c in chunks]
    except Exception as exc:
        print(f"[ingestion] pymupdf4llm non disponibile ({type(exc).__name__}: {exc}); uso il testo semplice.")
        return None


def parse_pdf(
    path: Path, *, include_visuals: bool = False
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Ritorna (metadati_documento, lista_pagine).

    Ogni pagina: { page_number, text, tables, content_type }. Il testo è Markdown:
    le tabelle sono già dentro il testo, con le intestazioni delle colonne.
    content_type pagina: "text" oppure "ocr".
    """
    doc = fitz.open(path)
    meta = _extract_pdf_metadata(doc)
    markdown = _pdf_markdown(doc)
    meta["source_parser"] = "pymupdf4llm" if markdown is not None else "PyMuPDF"

    pages: list[dict[str, Any]] = []
    for i, page in enumerate(doc):
        text = markdown[i] if markdown is not None and i < len(markdown) else ""
        if len(text.strip()) < 30:
            # pymupdf4llm può scartare una pagina breve come intestazione: si usa il testo semplice.
            text = (page.get_text("text") or "").strip() or text
        content_type = "text"
        # Pagina scansionata: quasi niente testo ma immagini. OCR solo se Tesseract è installato.
        if len(text.strip()) < 30 and page.get_images(full=False):
            ocr_text = _ocr_page(page)
            if ocr_text is not None:
                text = ocr_text
                content_type = "ocr"

        page_data = {
            "page_number": i + 1,
            "text": text,
            "tables": [],
            "content_type": content_type,
        }
        if include_visuals:
            pix = page.get_pixmap(dpi=110, alpha=False)
            image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            max_pixels = 4_000_000
            if image.width * image.height > max_pixels:
                scale = (max_pixels / (image.width * image.height)) ** 0.5
                image = image.resize(
                    (int(image.width * scale), int(image.height * scale)),
                    Image.Resampling.LANCZOS,
                )
            visual = io.BytesIO()
            image.save(visual, format="JPEG", quality=90, optimize=True)
            page_data["visual_image"] = visual.getvalue()
        pages.append(page_data)
    doc.close()
    return meta, pages


# -------------------------------------------------------------- chunking ----

def _chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Token-aware chunking con overlap. Dimensioni in token (cl100k_base, ~ok per Voyage)."""
    if not text.strip():
        return []
    tokens = _TOKENIZER.encode(text)
    chunks: list[str] = []
    step = max(size - overlap, 1)
    for start in range(0, len(tokens), step):
        window = tokens[start:start + size]
        if not window:
            break
        piece = _TOKENIZER.decode(window).strip()
        if piece:
            chunks.append(piece)
        if start + size >= len(tokens):
            break
    return chunks


def _tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text))


def _markdown_blocks(text: str) -> list[tuple[str, str]]:
    """Blocchi (tipo, testo): titoli, tabelle intere, paragrafi."""
    blocks: list[tuple[str, str]] = []
    para: list[str] = []
    table: list[str] = []

    def flush() -> None:
        if para:
            blocks.append(("text", "\n".join(para).strip()))
            para.clear()
        if table:
            blocks.append(("table", "\n".join(table)))
            table.clear()

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|"):
            if para:
                blocks.append(("text", "\n".join(para).strip()))
                para.clear()
            table.append(stripped)
            continue
        if table:
            blocks.append(("table", "\n".join(table)))
            table.clear()
        if stripped.startswith("#"):
            flush()
            blocks.append(("heading", stripped))
        elif not stripped:
            flush()
        else:
            para.append(line)
    flush()
    return [b for b in blocks if b[1]]


def _split_table(table: str, size: int, prefix: str) -> list[str]:
    """Tabella troppo lunga: pezzi di righe, ognuno con l'intestazione delle colonne."""
    lines = table.splitlines()
    sep = next((i for i, l in enumerate(lines) if re.fullmatch(r"\|?\s*:?-{3,}.*", l)), None)
    header = lines[: sep + 1] if sep is not None and sep < 4 else lines[:1]
    # Le intestazioni con gli anni a volte stanno subito dopo il separatore: vanno ripetute anche loro.
    for line in lines[len(header): len(header) + 2]:
        if any(re.fullmatch(r"(19|20)\d\d", cell.strip()) for cell in line.strip("|").split("|")):
            header.append(line)
        else:
            break
    body = lines[len(header):]
    base = (prefix + "\n" if prefix else "") + "\n".join(header)
    pieces, current = [], []
    for row in body:
        if current and _tokens(base + "\n" + "\n".join(current + [row])) > size:
            pieces.append(base + "\n" + "\n".join(current))
            current = []
        current.append(row)
    if current or not pieces:
        pieces.append(base + "\n" + "\n".join(current))
    return pieces


def _chunk_markdown(text: str, size: int, overlap: int) -> list[str]:
    """Chunk che rispettano la struttura: non spezzano una tabella a metà riga,
    ripetono l'intestazione delle colonne e il titolo della sezione in ogni pezzo."""
    if not text.strip():
        return []
    chunks: list[str] = []
    current: list[str] = []
    heading = ""

    def emit() -> None:
        if current:
            chunks.append("\n\n".join(current).strip())
            current.clear()

    for kind, block in _markdown_blocks(text):
        if kind == "heading":
            heading = block
            if current and _tokens("\n\n".join(current + [block])) > size:
                emit()
            current.append(block)
            continue
        candidate = "\n\n".join(current + [block])
        if _tokens(candidate) <= size:
            current.append(block)
            continue
        if current and current[-1] == heading:
            current.pop()  # il titolo va col blocco che introduce, non in fondo al chunk precedente
        emit()
        prefix = heading if heading and heading != block else ""
        with_prefix = (prefix + "\n\n" if prefix else "") + block
        if _tokens(with_prefix) <= size:
            current.append(with_prefix)
        elif kind == "table":
            chunks.extend(_split_table(block, size, prefix))
        else:
            for piece in _chunk_text(block, size, overlap):
                chunks.append((prefix + "\n\n" if prefix else "") + piece)
    emit()
    # Un titolo rimasto da solo non serve alla ricerca.
    return [c for c in chunks if not re.fullmatch(r"#+[^\n]*", c)]


CHUNK_STRATEGIES = ("structure", "fixed")


def chunking_settings(values: dict[str, Any] | None = None) -> dict[str, Any]:
    """Parametri di chunking validati; quelli assenti prendono i valori del .env.

    strategy: "structure" → blocchi di titoli, paragrafi e tabelle intere fino a `size` token;
              l'overlap si usa solo quando un paragrafo va spezzato.
              "fixed"     → finestra fissa di `size` token che scorre di `size - overlap`.
    I chunk non attraversano mai il confine di pagina: la citazione resta precisa.
    """
    values = values or {}
    strategy = str(values.get("strategy") or "structure").lower()
    if strategy not in CHUNK_STRATEGIES:
        raise ValueError("Strategia di chunking: structure oppure fixed")
    size = int(values.get("size") or config.CHUNK_SIZE_TOKENS)
    overlap = int(values.get("overlap") if values.get("overlap") not in (None, "") else config.CHUNK_OVERLAP_TOKENS)
    if not 128 <= size <= 2048:
        raise ValueError("Dimensione del chunk: tra 128 e 2048 token")
    if not 0 <= overlap <= size // 2:
        raise ValueError("Sovrapposizione: tra 0 e metà della dimensione del chunk")
    return {"strategy": strategy, "size": size, "overlap": overlap}


def build_chunks(
    pages: list[dict[str, Any]],
    *,
    size: int,
    overlap: int,
    strategy: str = "structure",
) -> list[dict[str, Any]]:
    """Flatten pagine → lista chunk con (content, page_number, content_type).

    Pagina con immagine (modalità multimodal): un solo passaggio con immagine e testo della
    pagina. Niente passaggi di solo testo in più: sarebbero copie dello stesso testo, e reranker
    e modello vedrebbero la stessa pagina due volte occupando posti tra le fonti.
    """
    out: list[dict[str, Any]] = []
    for page in pages:
        visual = bool(page.get("visual_image"))
        split = _chunk_text if strategy == "fixed" else _chunk_markdown
        for piece in ([] if visual else split(page["text"], size, overlap)):
            out.append({
                "content": piece,
                "page_number": page["page_number"],
                "page_start": page["page_number"],
                "page_end": page["page_number"],
                "content_type": "table" if page["content_type"] == "text" and re.search(r"(?m)^\|", piece)
                                else page["content_type"],
            })
        # Ogni tabella diventa un chunk a sé (preserva la struttura)
        for tbl in page["tables"]:
            out.append({
                "content": tbl,
                "page_number": page["page_number"],
                "page_start": page["page_number"],
                "page_end": page["page_number"],
                "content_type": "table",
            })
        if visual:
            out.append({
                "content": page["text"].strip() or f"Pagina {page['page_number']}: contenuto solo visivo, nessun testo estratto.",
                "page_number": page["page_number"],
                "page_start": page["page_number"],
                "page_end": page["page_number"],
                "content_type": "visual_page",
                "visual_image": page["visual_image"],
            })
        if page.get("office_structure_text") and page["page_number"] == 1:
            out.extend({
                "content": piece,
                "page_number": None,
                "page_start": None,
                "page_end": None,
                "content_type": "office_structure",
            } for piece in _chunk_text(page["office_structure_text"], size, overlap))
    return out


# --------------------------------------------------------------- storage ----

def upload_to_gridfs(db: Database, bucket_name: str, path: Path) -> Any:
    fs = GridFS(db, collection=bucket_name)
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    with path.open("rb") as f:
        return fs.put(
            f,
            filename=path.name,
            contentType=content_type,
            uploadDate=datetime.now(timezone.utc),
        )


# ------------------------------------------------------------ end-to-end ----

def ingest_pdf(
    path: Path | str,
    *,
    db: Database,
    collection: Collection,
    storage: str = "gridfs",
    mode: str | None = None,
    user_metadata: dict[str, Any] | None = None,
    chunking: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Pipeline completa di ingest per un singolo PDF.

    storage:       "gridfs"  → upload del binario in GridFS.
                   "s3"      → placeholder (upload non configurato).
    mode:          "auto"    → autoEmbed lato Atlas.
                   "manual"  → embedding via Voyage /embeddings in `embedding` field.
                   "context" → chunk locali page-aware contestualizzati insieme via
                               Voyage /contextualizedembeddings (limite stimato 32K token/doc
                               per i chunk già suddivisi usati qui).
    user_metadata: dict opzionale di override forniti dall'utente in upload.
                   Chiavi supportate: title, author, category, status, language,
                   year (int), tags (list[str]), file_type.
                   Sovrascrivono i metadati estratti dal PDF; vengono propagati
                   identici a TUTTI i chunk del documento.
    chunking:      dict opzionale {strategy, size, overlap}: vedi `chunking_settings`.
    """
    started_at = time.perf_counter()
    mode = (mode or config.EMBED_MODE).lower()
    if mode not in ("auto", "manual", "context", "multimodal"):
        raise ValueError(f"mode non valido: {mode}")
    user_metadata = dict(user_metadata or {})
    duplicate_policy = (
        user_metadata.pop("duplicate_policy", None)
        or config.INGESTION_DUPLICATE_POLICY
    ).lower()
    if duplicate_policy not in ("skip", "replace", "append"):
        raise ValueError("duplicate_policy deve essere: skip | replace | append")
    chunk_params = chunking_settings(chunking)
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    doc_id = _sha256_file(path)
    mode_query = {"document_id": doc_id, "embedding_mode": mode}
    existing_chunks = collection.count_documents(mode_query, limit=1)
    if existing_chunks and duplicate_policy == "skip":
        return {
            "document_id": doc_id,
            "filename": path.name,
            "chunks_inserted": 0,
            "skipped": True,
            "reason": f"documento gia' indicizzato in modalita' {mode}",
            "duplicate_policy": duplicate_policy,
            "duration_seconds": round(time.perf_counter() - started_at, 2),
        }
    # Sostituzione: i passaggi vecchi si cancellano solo dopo aver inserito i nuovi,
    # così un errore di lettura o di embedding non fa perdere il documento.
    # Il file in GridFS ha lo stesso contenuto (stesso hash): si riusa.
    replaced_ids = ([row["_id"] for row in collection.find(mode_query, {"_id": 1})]
                    if existing_chunks and duplicate_policy == "replace" else [])

    pdf_meta, pages = parse_document(path, include_visuals=(mode == "multimodal"))

    storage_ref: dict[str, Any] = {"storage": storage}
    if storage == "gridfs":
        existing_file = collection.find_one(
            {"document_id": doc_id, "gridfs_file_id": {"$exists": True}},
            {"gridfs_file_id": 1, "_id": 0},
        )
        if existing_file:
            storage_ref["gridfs_file_id"] = existing_file["gridfs_file_id"]
        else:
            gridfs_id = upload_to_gridfs(db, config.MONGODB_GRIDFS_BUCKET, path)
            storage_ref["gridfs_file_id"] = str(gridfs_id)
    elif storage == "s3":
        storage_ref["s3_uri"] = None
        storage_ref["note"] = "S3 upload non configurato in questa demo"
    else:
        raise ValueError(f"storage non supportato: {storage}")

    raw_chunks = build_chunks(pages, size=chunk_params["size"], overlap=chunk_params["overlap"], strategy=chunk_params["strategy"])

    now = datetime.now(timezone.utc)

    # In modalità manual/context, calcoliamo gli embedding via Voyage API.
    embeddings: list[list[float]] = []
    if mode == "manual" and raw_chunks:
        embeddings = embed_voyage(
            [c["content"] for c in raw_chunks],
            input_type="document",
        )
        _validate_embeddings(
            embeddings,
            expected_count=len(raw_chunks),
            expected_dimensions=config.VOYAGE_EMBED_DIMENSIONS,
        )
    elif mode == "context" and raw_chunks:
        # I chunk restano quelli page-aware prodotti dal parser. Voyage li
        # riceve come un unico gruppo e genera ciascun vettore nel contesto
        # dell'intero documento, senza alterare confini o metadati pagina.
        estimated_tokens = sum(
            len(_TOKENIZER.encode(c["content"])) for c in raw_chunks
        )
        if estimated_tokens > _CONTEXT_MAX_TOKENS:
            raise RuntimeError(
                "Documento oltre il limite pre-chunked di voyage-context-4 (32K token): "
                f"circa {estimated_tokens} token, limite {_CONTEXT_MAX_TOKENS}. "
                "Usa la modalità manual/multimodal oppure riduci il documento."
            )
        contextualized = embed_voyage_contextualized(
            [[c["content"] for c in raw_chunks]],
            input_type="document",
        )
        embeddings = contextualized[0] if contextualized else []
        _validate_embeddings(
            embeddings,
            expected_count=len(raw_chunks),
            expected_dimensions=config.VOYAGE_CONTEXT_DIMENSIONS,
        )
    elif mode == "multimodal" and raw_chunks:
        from .embeddings import embed_voyage_multimodal
        mm_inputs = []
        for chunk in raw_chunks:
            image_bytes = chunk.get("visual_image")
            text_for_embedding = chunk["content"] or "Contenuto visivo del documento."
            if image_bytes:
                text_tokens = _TOKENIZER.encode(text_for_embedding)
                if len(text_tokens) > 12_000:
                    text_for_embedding = _TOKENIZER.decode(text_tokens[:12_000])
            parts = [{"type": "text", "text": text_for_embedding}]
            if image_bytes:
                parts.append({
                    "type": "image_base64",
                    "image_base64": "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("ascii"),
                })
            mm_inputs.append(parts)
        embeddings = embed_voyage_multimodal(mm_inputs, input_type="document")
        _validate_embeddings(
            embeddings,
            expected_count=len(raw_chunks),
            expected_dimensions=config.VOYAGE_MULTIMODAL_DIMENSIONS,
        )

    # Calcolati sui chunk effettivi che verranno salvati.
    total = len(raw_chunks)
    doc_lang = _safe_lang(" ".join(c["content"] for c in raw_chunks[:3])) if raw_chunks else None

    # Metadati user-supplied: override su title/author/language, nuovi campi
    # category/status/year/tags/file_type ereditati da ogni chunk.
    user_title    = user_metadata.get("title")    or None
    user_author   = user_metadata.get("author")   or None
    user_language = user_metadata.get("language") or None
    user_category = user_metadata.get("category") or None
    user_status   = user_metadata.get("status")   or None
    user_year     = user_metadata.get("year")
    user_tags     = user_metadata.get("tags")     or []
    user_ftype    = user_metadata.get("file_type") or path.suffix.lstrip(".").lower() or "pdf"

    docs: list[dict[str, Any]] = []
    for idx, ch in enumerate(raw_chunks):
        detected_lang = _safe_lang(ch["content"]) or doc_lang
        doc = {
            "document_id": doc_id,
            "filename": path.name,
            **storage_ref,
            "chunk_index": idx,
            "chunk_total": total,
            "page_number": ch["page_number"],
            "page_start": ch.get("page_start", ch["page_number"]),
            "page_end": ch.get("page_end", ch["page_number"]),
            "content_type": ch["content_type"],
            "content": ch["content"],
            "chunk_hash": _sha256_text(ch["content"]),
            "char_count": len(ch["content"]),
            "word_count": len(ch["content"].split()),
            # Lingua: override utente prevale sul rilevamento automatico
            "language": user_language or detected_lang,
            # Title/author: override utente prevale sul metadato PDF
            "title": user_title or pdf_meta["title"],
            "author": user_author or pdf_meta["author"],
            "subject": pdf_meta["subject"],
            "keywords": pdf_meta["keywords"],
            "creator": pdf_meta["creator"],
            "producer": pdf_meta["producer"],
            "creation_date": pdf_meta["creation_date"],
            "modification_date": pdf_meta["modification_date"],
            "total_pages": pdf_meta["total_pages"],
            "format": pdf_meta["format"],
            "ingested_at": now,
            # Nuovi campi user-supplied; quelli assenti vengono rimossi prima
            # della scrittura, invece di essere memorizzati come null.
            "category": user_category,
            "status": user_status,
            "year": user_year,
            "tags": user_tags,
            "file_type": user_ftype,
            "schema_version": _SCHEMA_VERSION,
            "ingestion_pipeline_version": config.INGESTION_PIPELINE_VERSION,
            "chunker_version": config.CHUNKER_VERSION,
            "chunking": {"strategy": chunk_params["strategy"], "size_tokens": chunk_params["size"],
                         "overlap_tokens": chunk_params["overlap"]},
            "embedding_mode": mode,
            "embedding_model": {
                "auto": config.VOYAGE_EMBED_MODEL,
                "manual": config.VOYAGE_EMBED_MODEL,
                "context": config.VOYAGE_CONTEXT_MODEL,
                "multimodal": config.VOYAGE_MULTIMODAL_MODEL,
            }[mode],
            "embedding_input_type": "document",
            "context_strategy": {
                "auto": "atlas_autoembed",
                "manual": "independent_local_chunks",
                "context": "page_aware_contextualized_chunks",
                "multimodal": "voyage_multimodal_page_images",
            }[mode],
        }
        if mode in ("manual", "context", "multimodal") and embeddings:
            vector_field = {
                "manual": "embedding",
                "context": "embedding_context",
                "multimodal": "embedding_multimodal",
            }[mode]
            doc[vector_field] = embeddings[idx]
            doc["embedding_dimensions"] = {
                "manual": config.VOYAGE_EMBED_DIMENSIONS,
                "context": config.VOYAGE_CONTEXT_DIMENSIONS,
                "multimodal": config.VOYAGE_MULTIMODAL_DIMENSIONS,
            }[mode]
            doc["embedding_dtype"] = "float"
        docs.append(_without_empty_fields(doc))

    if docs:
        collection.insert_many(docs)
    replaced = collection.delete_many({"_id": {"$in": replaced_ids}}).deleted_count if replaced_ids and docs else 0

    # Riporta esplicitamente cosa è stato applicato ai chunk: utile per debug/demo
    # (così l'utente vede a video se i suoi override sono andati a buon fine).
    effective_fields = {
        "title":     user_title    or pdf_meta["title"],
        "author":    user_author   or pdf_meta["author"],
        "language":  user_language or "(rilevata per chunk con langdetect)",
        "category":  user_category,
        "status":    user_status,
        "year":      user_year,
        "tags":      user_tags,
        "file_type": user_ftype,
    }
    summary: dict[str, Any] = {
        "document_id": doc_id,
        "filename": path.name,
        "chunks_inserted": len(docs),
        "replaced_chunks": replaced,
        "skipped": False,
        "duplicate_policy": duplicate_policy,
        "storage": storage_ref,
        "mode": mode,
        "embedding": {
            "model": {
                "auto": config.VOYAGE_EMBED_MODEL,
                "manual": config.VOYAGE_EMBED_MODEL,
                "context": config.VOYAGE_CONTEXT_MODEL,
                "multimodal": config.VOYAGE_MULTIMODAL_MODEL,
            }[mode],
            "dimensions": (
                config.VOYAGE_CONTEXT_DIMENSIONS
                if mode == "context"
                else (
                    config.VOYAGE_EMBED_DIMENSIONS if mode == "manual"
                    else config.VOYAGE_MULTIMODAL_DIMENSIONS if mode == "multimodal"
                    else None
                )
            ),
            "strategy": {
                "auto": "atlas_autoembed",
                "manual": "independent_local_chunks",
                "context": "page_aware_contextualized_chunks",
                "multimodal": "voyage_multimodal_page_images",
            }[mode],
        },
        "user_metadata_provided": user_metadata,
        "effective_metadata_applied_to_chunks": effective_fields,
        "pdf_native_metadata": pdf_meta,
        "file_size_bytes": path.stat().st_size,
        "pages_processed": len(pages),
        "pages_with_text": sum(bool(page.get("text", "").strip()) for page in pages),
        "pages_ocr": sum(page.get("content_type") == "ocr" for page in pages),
        "tables_detected": sum(len(re.findall(r"(?m)^\|[ :]*-{3,}", page.get("text") or "")) for page in pages),
        "chunks_by_type": {
            kind: sum(chunk.get("content_type") == kind for chunk in docs)
            for kind in sorted({chunk.get("content_type", "text") for chunk in docs})
        },
        "duration_seconds": round(time.perf_counter() - started_at, 2),
    }
    if mode == "context":
        summary["context_note"] = (
            f"Modalità context page-aware: voyage-context-4 ha contestualizzato "
            f"{len(docs)} chunk locali conservando pagina e tipo contenuto."
        )
    if mode == "multimodal":
        summary["multimodal_note"] = (
            f"{sum(c.get('content_type') == 'visual_page' for c in raw_chunks)} pagine/slide "
            "abbinate al testo estratto e vettorizzate con Voyage multimodale."
        )
    return summary
