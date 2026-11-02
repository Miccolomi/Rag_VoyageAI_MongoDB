from pathlib import Path

import mongomock
import pytest
from mongomock.gridfs import enable_gridfs_integration

from src import ingestion
from src.ingestion import build_chunks, context_groups, ingest_document

enable_gridfs_integration()


def test_gruppi_di_contesto_contigui_sotto_il_limite():
    assert context_groups([10, 10, 10], limit=25) == [(0, 2), (2, 3)]
    assert context_groups([5] * 4, limit=100) == [(0, 4)]
    assert context_groups([], limit=10) == []


def test_gruppi_di_contesto_coprono_tutti_i_chunk_senza_sovrapposizioni():
    counts = [500] * 130  # ~65K token, come un 10-K di media lunghezza
    groups = context_groups(counts, limit=28000)
    assert groups[0][0] == 0 and groups[-1][1] == len(counts)
    assert all(a[1] == b[0] for a, b in zip(groups, groups[1:]))
    assert all(sum(counts[s:e]) <= 28000 for s, e in groups)


def test_chunk_troppo_grande_da_solo():
    with pytest.raises(RuntimeError):
        context_groups([50], limit=10)


def test_pagina_multimodale_produce_un_solo_chunk_di_testo():
    pages = [{"page_number": 1, "text": " ".join(["parola"] * 1200), "tables": ["| a | b |"],
              "content_type": "text", "visual_image": b"jpeg"}]
    chunks = build_chunks(pages, size=512, overlap=64)
    kinds = [chunk["content_type"] for chunk in chunks]
    assert kinds == ["visual_page", "table"]


def test_pagina_testuale_usa_finestre_e_nessun_office_structure():
    pages = [{"page_number": 3, "text": " ".join(f"w{i}" for i in range(1200)), "tables": [],
              "content_type": "text", "office_structure_text": "titolo"}]
    chunks = build_chunks(pages, size=512, overlap=64)
    assert {chunk["content_type"] for chunk in chunks} == {"text"}
    assert len(chunks) == 3
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
    remaining = list(coll.find({}, {"ingestion_id": 1}))
    assert {doc["ingestion_id"] for doc in remaining} == {second["ingestion_id"]}


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


def test_context_suddivide_il_documento_lungo(tmp_path, monkeypatch):
    db, coll = _fake_db()
    path = _write(tmp_path, " ".join(f"parola{i}" for i in range(3000)))
    calls = []

    def fake_context(documents, input_type="document"):
        calls.append(len(documents[0]))
        return [[[0.0] * ingestion.config.VOYAGE_CONTEXT_DIMENSIONS for _ in documents[0]]]
    monkeypatch.setattr(ingestion, "embed_voyage_contextualized", fake_context)
    monkeypatch.setattr(ingestion, "_CONTEXT_GROUP_TOKENS", 1200)
    summary = ingest_document(path, db=db, collection=coll, mode="context")
    assert len(calls) > 1
    assert sum(calls) == summary["chunks_inserted"]
    assert summary["embedding"]["strategy"] == "page_aware_contextualized_chunk_groups"
    assert coll.count_documents({"embedding_context": {"$exists": True}}) == summary["chunks_inserted"]


def _make_pdf(tmp_path: Path, pages: int = 3) -> Path:
    import fitz
    doc = fitz.open()
    for number in range(pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"Pagina {number + 1}: purchases of property plant and equipment {1500 + number}")
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
