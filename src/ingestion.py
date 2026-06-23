"""Ingestion PDF end-to-end: parsing (testo + tabelle + OCR), chunking,
upload del binario originale su GridFS, insert dei chunk in MongoDB.

L'embedding NON viene calcolato qui: i chunk vengono inseriti come testo puro
e Atlas Vector Search (autoEmbed con Voyage) genera i vettori automaticamente.
"""
from __future__ import annotations

import hashlib
import io
import math
import mimetypes
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import pdfplumber
import pytesseract
import tiktoken
from bson import ObjectId
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
_SCHEMA_VERSION = "pdf_chunks_v3_1"
_CONTEXT_MAX_TOKENS = 120_000


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


def _extract_pdf_images(doc: fitz.Document, min_size: int = 150) -> list[dict[str, Any]]:
    """Estrae le immagini significative da un PDF, skip elementi decorativi piccoli.

    Ritorna lista di dict: {page_number, bytes (PNG), width, height}
    """
    out: list[dict[str, Any]] = []
    for page_idx, page in enumerate(doc):
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            try:
                pix = fitz.Pixmap(doc, xref)
                if pix.width < min_size or pix.height < min_size:
                    continue
                if pix.n - pix.alpha > 3:  # CMYK → convert to RGB
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                out.append({
                    "page_number": page_idx + 1,
                    "bytes": pix.tobytes("png"),
                    "width": pix.width,
                    "height": pix.height,
                })
            except Exception:
                continue
    return out


_VISION_PROMPT = (
    "Descrivi tecnicamente questa immagine in italiano in modo dettagliato "
    "(max 150 parole). Se è un diagramma/schema elenca i componenti e le "
    "relazioni. Se è una tabella riassumi i dati. Se è una foto descrivi "
    "oggetti e contesto. Non inventare: se non capisci, dillo."
)


def _describe_with_anthropic(image_bytes: bytes) -> str | None:
    if not config.ANTHROPIC_API_KEY:
        return None
    import base64
    from anthropic import Anthropic
    try:
        client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
        image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/png", "data": image_b64,
                    }},
                    {"type": "text", "text": _VISION_PROMPT},
                ],
            }],
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    except Exception as e:
        print(f"[ingestion] Claude vision fallita: {e}")
        return None


def _describe_with_gemini(image_bytes: bytes) -> str | None:
    if not config.GEMINI_API_KEY:
        return None
    import base64
    import httpx
    try:
        image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{config.GEMINI_MODEL}:generateContent")
        payload = {
            "contents": [{
                "parts": [
                    {"text": _VISION_PROMPT},
                    {"inline_data": {"mime_type": "image/png", "data": image_b64}},
                ],
            }],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 300},
        }
        # API key in header (NON in URL → non appare nei log/errori)
        headers = {"x-goog-api-key": config.GEMINI_API_KEY}
        with httpx.Client(timeout=60.0) as client:
            r = client.post(url, headers=headers, json=payload)
            if r.status_code == 429:
                print("[ingestion] Gemini rate limit (429), salto image description")
                return None
            r.raise_for_status()
            data = r.json()
        return "".join(
            part.get("text", "")
            for c in data.get("candidates", [])
            for part in c.get("content", {}).get("parts", [])
        ) or None
    except Exception as e:
        print(f"[ingestion] Gemini vision fallita: {e}")
        return None


def _describe_with_ollama(image_bytes: bytes) -> str | None:
    """Vision via Ollama local — usa OLLAMA_VISION_MODEL (es. llava:7b)."""
    import base64
    import httpx
    try:
        image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
        url = f"{config.OLLAMA_BASE_URL}/api/generate"
        payload = {
            "model": config.OLLAMA_VISION_MODEL,
            "prompt": _VISION_PROMPT,
            "images": [image_b64],
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 300},
        }
        with httpx.Client(timeout=180.0) as client:
            r = client.post(url, json=payload)
            if r.status_code == 404:
                print(f"[ingestion] Modello Ollama vision non trovato: "
                      f"{config.OLLAMA_VISION_MODEL}. Esegui `ollama pull "
                      f"{config.OLLAMA_VISION_MODEL}`.")
                return None
            r.raise_for_status()
            data = r.json()
        return (data.get("response") or "").strip() or None
    except Exception as e:
        print(f"[ingestion] Ollama vision fallita: {e}")
        return None


def describe_image(image_bytes: bytes) -> str | None:
    """Dispatch image description in base a LLM_PROVIDER.
    Ritorna None se il provider attivo non ha chiave o non supporta vision."""
    if config.LLM_PROVIDER == "anthropic":
        return _describe_with_anthropic(image_bytes)
    if config.LLM_PROVIDER == "gemini":
        return _describe_with_gemini(image_bytes)
    if config.LLM_PROVIDER == "ollama":
        return _describe_with_ollama(image_bytes)
    return None


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


def _extract_tables(path: Path, page_idx: int) -> list[str]:
    out: list[str] = []
    try:
        with pdfplumber.open(path) as pdf:
            if page_idx >= len(pdf.pages):
                return out
            page = pdf.pages[page_idx]
            tables = page.extract_tables() or []
            for tbl in tables:
                rows = []
                for row in tbl:
                    cells = [(c or "").strip() for c in row]
                    rows.append("| " + " | ".join(cells) + " |")
                if rows:
                    if len(rows) > 1:
                        cols = rows[0].count("|") - 1
                        sep = "| " + " | ".join(["---"] * cols) + " |"
                        rows.insert(1, sep)
                    out.append("\n".join(rows))
    except Exception:
        # pdfplumber può fallire su PDF corrotti/criptati: non blocchiamo l'ingest.
        pass
    return out


# --------------------------------------------------------------- parsing ----

_MARKITDOWN_EXTS = {".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls",
                    ".txt", ".html", ".htm", ".csv", ".json", ".xml", ".md"}


def parse_with_markitdown(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Parsing per formati non-PDF tramite MarkItDown (Microsoft).
    Ritorna markdown come 1 sola "pagina" — la struttura sezione/slide è preservata
    nei marker markdown del testo (##, ###, ecc.) che il chunker rispetta.
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
        "total_pages": 1,
        "format": path.suffix.lstrip(".").upper(),
    }
    pages = [{
        "page_number": 1, "text": text, "tables": [], "content_type": "text",
    }]
    return meta, pages


def parse_document(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Dispatch per estensione: PDF → parser nativo, altri → MarkItDown."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        return parse_pdf(path)
    if ext in _MARKITDOWN_EXTS:
        return parse_with_markitdown(path)
    raise ValueError(f"Estensione non supportata: {ext}")


def parse_pdf(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Ritorna (metadati_documento, lista_pagine).

    Ogni pagina: { page_number, text, tables, content_type }
    content_type pagina: "text" oppure "ocr" (le tabelle diventano chunk separati).
    """
    doc = fitz.open(path)
    meta = _extract_pdf_metadata(doc)

    pages: list[dict[str, Any]] = []
    for i, page in enumerate(doc):
        text = page.get_text("text") or ""
        content_type = "text"
        # Fallback OCR: pagina con pochissimo testo ma con immagini → scansione
        if len(text.strip()) < 30 and page.get_images(full=False):
            ocr_text = _ocr_page(page)
            if ocr_text is not None:
                text = ocr_text
                content_type = "ocr"
            # Se OCR non disponibile, manteniamo il poco testo estratto da PyMuPDF.

        pages.append({
            "page_number": i + 1,
            "text": text,
            "tables": _extract_tables(path, i),
            "content_type": content_type,
        })
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


def build_chunks(
    pages: list[dict[str, Any]],
    *,
    size: int,
    overlap: int,
) -> list[dict[str, Any]]:
    """Flatten pagine → lista chunk con (content, page_number, content_type)."""
    out: list[dict[str, Any]] = []
    for page in pages:
        for piece in _chunk_text(page["text"], size, overlap):
            out.append({
                "content": piece,
                "page_number": page["page_number"],
                "page_start": page["page_number"],
                "page_end": page["page_number"],
                "content_type": page["content_type"],
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
) -> dict[str, Any]:
    """Pipeline completa di ingest per un singolo PDF.

    storage:       "gridfs"  → upload del binario in GridFS.
                   "s3"      → placeholder (upload non configurato).
    mode:          "auto"    → autoEmbed lato Atlas.
                   "manual"  → embedding via Voyage /embeddings in `embedding` field.
                   "context" → chunk locali page-aware contestualizzati insieme via
                               Voyage /contextualizedembeddings (fino a 120K token/doc).
    user_metadata: dict opzionale di override forniti dall'utente in upload.
                   Chiavi supportate: title, author, category, status, language,
                   year (int), tags (list[str]), file_type.
                   Sovrascrivono i metadati estratti dal PDF; vengono propagati
                   identici a TUTTI i chunk del documento.
    """
    mode = (mode or config.EMBED_MODE).lower()
    if mode not in ("auto", "manual", "context"):
        raise ValueError(f"mode non valido: {mode}")
    user_metadata = dict(user_metadata or {})
    duplicate_policy = (
        user_metadata.pop("duplicate_policy", None)
        or config.INGESTION_DUPLICATE_POLICY
    ).lower()
    if duplicate_policy not in ("skip", "replace", "append"):
        raise ValueError("duplicate_policy deve essere: skip | replace | append")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    doc_id = _sha256_file(path)
    existing_chunks = collection.count_documents({"document_id": doc_id}, limit=1)
    if existing_chunks and duplicate_policy == "skip":
        return {
            "document_id": doc_id,
            "filename": path.name,
            "chunks_inserted": 0,
            "skipped": True,
            "reason": "documento gia' presente",
            "duplicate_policy": duplicate_policy,
        }
    if existing_chunks and duplicate_policy == "replace":
        existing_refs = list(collection.find(
            {"document_id": doc_id},
            {"gridfs_file_id": 1, "_id": 0},
        ))
        collection.delete_many({"document_id": doc_id})
        if storage == "gridfs":
            fs = GridFS(db, collection=config.MONGODB_GRIDFS_BUCKET)
            for ref in {r.get("gridfs_file_id") for r in existing_refs if r.get("gridfs_file_id")}:
                try:
                    fs.delete(ObjectId(ref))
                except Exception:
                    pass

    pdf_meta, pages = parse_document(path)

    storage_ref: dict[str, Any] = {"storage": storage}
    if storage == "gridfs":
        gridfs_id = upload_to_gridfs(db, config.MONGODB_GRIDFS_BUCKET, path)
        storage_ref["gridfs_file_id"] = str(gridfs_id)
    elif storage == "s3":
        storage_ref["s3_uri"] = None
        storage_ref["note"] = "S3 upload non configurato in questa demo"
    else:
        raise ValueError(f"storage non supportato: {storage}")

    raw_chunks = build_chunks(
        pages,
        size=config.CHUNK_SIZE_TOKENS,
        overlap=config.CHUNK_OVERLAP_TOKENS,
    )

    # Image processing opzionale: estrae immagini dal PDF, le descrive con
    # Claude vision, aggiunge un chunk per ognuna con content_type="image".
    process_images = user_metadata.get("process_images") and path.suffix.lower() == ".pdf"
    images_processed = 0
    if process_images:
        doc_imgs = fitz.open(path)
        images = _extract_pdf_images(doc_imgs)
        doc_imgs.close()
        for img in images:
            desc = describe_image(img["bytes"])
            if not desc:
                continue  # skip se provider vision non disponibile o errore
            raw_chunks.append({
                "content": f"[IMMAGINE p.{img['page_number']} · {img['width']}×{img['height']}px] {desc}",
                "page_number": img["page_number"],
                "page_start": img["page_number"],
                "page_end": img["page_number"],
                "content_type": "image",
            })
            images_processed += 1

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
                "Documento troppo lungo per voyage-context-4: "
                f"circa {estimated_tokens} token, limite {_CONTEXT_MAX_TOKENS}. "
                "Usa la modalità manual oppure suddividi il documento."
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
            "embedding_mode": mode,
            "embedding_model": (
                config.VOYAGE_CONTEXT_MODEL
                if mode == "context"
                else config.VOYAGE_EMBED_MODEL
            ),
            "embedding_input_type": "document",
            "context_strategy": {
                "auto": "atlas_autoembed",
                "manual": "independent_local_chunks",
                "context": "page_aware_contextualized_chunks",
            }[mode],
        }
        if mode in ("manual", "context") and embeddings:
            doc["embedding"] = embeddings[idx]
            doc["embedding_dimensions"] = (
                config.VOYAGE_CONTEXT_DIMENSIONS
                if mode == "context"
                else config.VOYAGE_EMBED_DIMENSIONS
            )
            doc["embedding_dtype"] = "float"
        docs.append(_without_empty_fields(doc))

    if docs:
        collection.insert_many(docs)

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
        "skipped": False,
        "duplicate_policy": duplicate_policy,
        "images_processed": images_processed if process_images else 0,
        "storage": storage_ref,
        "mode": mode,
        "embedding": {
            "model": (
                config.VOYAGE_CONTEXT_MODEL
                if mode == "context"
                else config.VOYAGE_EMBED_MODEL
            ),
            "dimensions": (
                config.VOYAGE_CONTEXT_DIMENSIONS
                if mode == "context"
                else (
                    config.VOYAGE_EMBED_DIMENSIONS
                    if mode == "manual"
                    else None
                )
            ),
            "strategy": {
                "auto": "atlas_autoembed",
                "manual": "independent_local_chunks",
                "context": "page_aware_contextualized_chunks",
            }[mode],
        },
        "user_metadata_provided": user_metadata,
        "effective_metadata_applied_to_chunks": effective_fields,
        "pdf_native_metadata": pdf_meta,
    }
    if mode == "context":
        summary["context_note"] = (
            f"Modalità context page-aware: voyage-context-4 ha contestualizzato "
            f"{len(docs)} chunk locali conservando pagina e tipo contenuto."
        )
    return summary
