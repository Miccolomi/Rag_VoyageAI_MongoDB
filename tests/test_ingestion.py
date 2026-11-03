from pathlib import Path

import mongomock
import pytest
from mongomock.gridfs import enable_gridfs_integration

from src import ingestion
from src.ingestion import build_chunks, ingest_pdf as ingest_document

enable_gridfs_integration()


def test_pagina_multimodale_e_un_solo_passaggio_senza_copie_del_testo():
    testo = " ".join(["parola"] * 1200)
    pages = [{"page_number": 1, "text": testo, "tables": [], "content_type": "text", "visual_image": b"jpeg"}]
    chunks = build_chunks(pages, size=512, overlap=64)
    assert [chunk["content_type"] for chunk in chunks] == ["visual_page"]
    assert chunks[0]["visual_image"] == b"jpeg" and chunks[0]["content"] == testo


def test_pagina_multimodale_senza_testo_ha_una_descrizione():
    pages = [{"page_number": 4, "text": "  ", "tables": [], "content_type": "text", "visual_image": b"jpeg"}]
    (chunk,) = build_chunks(pages, size=512, overlap=64)
    assert chunk["content"].startswith("Pagina 4:")


def test_pagina_testuale_usa_finestre_e_nessun_office_structure():
    pages = [{"page_number": 3, "text": " ".join(f"w{i}" for i in range(1200)), "tables": [],
              "content_type": "text", "office_structure_text": "titolo"}]
    chunks = build_chunks(pages, size=512, overlap=64)
    assert {chunk["content_type"] for chunk in chunks} == {"text"}
    assert len(chunks) >= 3
    assert all(chunk["page_number"] == 3 for chunk in chunks)


def _fake_db():
    client = mongomock.MongoClient()
    db = client["rag_test"]
    return db, db["pdf_chunks"]


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "nota.txt"
    path.write_text(text, encoding="utf-8")
    return path


def test_replace_inserisce_prima_di_cancellare(tmp_path):
    db, coll = _fake_db()
    path = _write(tmp_path, "Prima versione del documento di prova con abbastanza parole per la lingua.")
    first = ingest_document(path, db=db, collection=coll, mode="auto")
    assert first["chunks_inserted"] >= 1
    second = ingest_document(path, db=db, collection=coll, mode="auto",
                             user_metadata={"duplicate_policy": "replace"})
    assert second["replaced_chunks"] == first["chunks_inserted"]
    assert coll.count_documents({}) == second["chunks_inserted"]


def test_replace_fallito_non_perde_il_documento(tmp_path, monkeypatch):
    db, coll = _fake_db()
    path = _write(tmp_path, "Documento che deve sopravvivere a un errore durante la sostituzione.")
    first = ingest_document(path, db=db, collection=coll, mode="auto")

    def boom(*args, **kwargs):
        raise RuntimeError("errore simulato durante il parsing")
    monkeypatch.setattr(ingestion, "parse_document", boom)
    with pytest.raises(RuntimeError):
        ingest_document(path, db=db, collection=coll, mode="auto",
                        user_metadata={"duplicate_policy": "replace"})
    assert coll.count_documents({}) == first["chunks_inserted"]


def test_skip_non_reinserisce(tmp_path):
    db, coll = _fake_db()
    path = _write(tmp_path, "Documento di prova per la politica skip dei duplicati in ingestion.")
    ingest_document(path, db=db, collection=coll, mode="auto")
    again = ingest_document(path, db=db, collection=coll, mode="auto",
                            user_metadata={"duplicate_policy": "skip"})
    assert again["skipped"]


def _make_pdf(tmp_path: Path, pages: int = 3) -> Path:
    import fitz
    doc = fitz.open()
    for number in range(pages):
        page = doc.new_page()
        # Non in cima alla pagina: pymupdf4llm scarta le intestazioni.
        page.insert_text((72, 300), f"Pagina {number + 1}: purchases of property plant and equipment {1500 + number}")
    path = tmp_path / "bilancio.pdf"
    doc.save(path)
    doc.close()
    return path


def test_pdf_reale_auto_e_multimodal(tmp_path, monkeypatch):
    db, coll = _fake_db()
    path = _make_pdf(tmp_path)
    meta, pages = ingestion.parse_pdf(path)
    assert meta["total_pages"] == 3 and [p["page_number"] for p in pages] == [1, 2, 3]

    auto = ingest_document(path, db=db, collection=coll, mode="auto")
    assert auto["chunks_by_type"] == {"text": 3}

    def fake_mm(inputs, input_type="document"):
        assert all(any(part["type"] == "image_base64" for part in parts) for parts in inputs)
        return [[0.0] * ingestion.config.VOYAGE_MULTIMODAL_DIMENSIONS for _ in inputs]
    import src.embeddings as embeddings
    monkeypatch.setattr(embeddings, "embed_voyage_multimodal", fake_mm)
    mm = ingest_document(path, db=db, collection=coll, mode="multimodal")
    assert mm["chunks_by_type"] == {"visual_page": 3}
    # stesso file GridFS condiviso tra le modalità
    assert auto["storage"]["gridfs_file_id"] == mm["storage"]["gridfs_file_id"]


def test_chunking_settings_defaults_and_limits():
    import pytest
    from src import config
    from src.ingestion import chunking_settings
    assert chunking_settings(None) == {"strategy": "structure", "size": config.CHUNK_SIZE_TOKENS,
                                       "overlap": config.CHUNK_OVERLAP_TOKENS}
    assert chunking_settings({"strategy": "fixed", "size": 256, "overlap": 0})["overlap"] == 0
    for bad in ({"strategy": "semantic"}, {"size": 64}, {"size": 256, "overlap": 200}):
        with pytest.raises(ValueError):
            chunking_settings(bad)


def test_build_chunks_fixed_window_uses_overlap():
    from src.ingestion import _TOKENIZER, build_chunks
    text = " ".join(f"parola{i}" for i in range(600))
    page = {"page_number": 1, "text": text, "tables": [], "content_type": "text"}
    chunks = build_chunks([page], size=256, overlap=32, strategy="fixed")
    assert len(chunks) > 1
    first, second = (_TOKENIZER.encode(c["content"]) for c in chunks[:2])
    assert len(first) <= 256 and first[-32:] == second[:32]
