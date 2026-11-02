"""Avvio dell'app Streamlit senza servizi esterni.

MongoDB è sostituito da mongomock: le funzioni Atlas Search non esistono e
finiscono nei blocchi try dell'app. Il test verifica che tutte le schede
vengano eseguite senza eccezioni Python (nomi non definiti, import, ecc.).
"""
from pathlib import Path

import mongomock
import pytest
from streamlit.testing.v1 import AppTest

from src import config

APP = Path(__file__).resolve().parent.parent / "app.py"


@pytest.fixture()
def fake_mongo(monkeypatch):
    client = mongomock.MongoClient()
    coll = client[config.MONGODB_DB][config.MONGODB_COLLECTION]
    coll.insert_many([
        {"document_id": "d1", "filename": "3M_2018_10K.pdf", "embedding_mode": "auto",
         "content": "Purchases of property, plant and equipment 1,577", "page_number": 60,
         "chunk_hash": "h1", "year": 2018, "language": "en", "file_type": "pdf"},
        {"document_id": "d1", "filename": "3M_2018_10K.pdf", "embedding_mode": "manual",
         "content": "Purchases of property, plant and equipment 1,577", "page_number": 60,
         "chunk_hash": "h1", "year": 2018, "language": "en", "file_type": "pdf"},
    ])
    monkeypatch.setattr(config, "get_client", lambda: client)
    # Il ping del health check usa un MongoClient reale con timeout breve: atteso errore gestito.
    return client


@pytest.mark.parametrize("page", ["story", "upload", "query", "agentic", "admin"])
def test_tutte_le_pagine_si_avviano(fake_mongo, page):
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.session_state["page_nav"] = page
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert "RAG Demo" in at.title[0].value


def _passage(n, mode="auto", **extra):
    return {"_id": f"c{n}", "document_id": "d1", "filename": "3M_2018_10K.pdf", "page_number": 60 + n,
            "chunk_index": n, "content": f"Purchases of PP&E were 1,577 million ({n})",
            "embedding_mode": mode, "score": 0.03 - n * 0.001, "_hybrid_rank": n,
            "_rerank_score": 0.9 - n * 0.1,
            "scoreDetails": {"details": [{"inputPipelineName": "text", "rank": n},
                                         {"inputPipelineName": f"vector_{mode}", "rank": n + 1}]},
            **extra}


def test_risultato_query_e_percorso_agentico_si_visualizzano(fake_mongo):
    from datetime import datetime, timezone
    passages = [_passage(1), _passage(2, "manual")]
    answer = "Le spese sono state 1.577 milioni [3M_2018_10K.pdf (p. 61)](#1)."
    result = {
        "query_v2": "Capex 3M 2018?", "rerank_instructions_v2": "", "filters_v2": None,
        "scope_v2": {"database": "rag", "collection": "pdf_chunks", "modes": ["auto", "manual"],
                     "rows": [], "text_index": "t", "unavailable": [], "mismatched": [], "unclassified": 0},
        "pipeline_v2": [{"$limit": 1}], "top_k_v2": 20, "relevance_threshold_v2": 0.15,
        "timings_ms_v2": {"x": 1.0}, "atlas_results_v2": passages, "raw_v2": passages,
        "reranked_v2": passages, "rerank_mode_v2": "native", "top_rerank_v2": 0.8,
        "reviewed_v2": {"answer": {"answer": answer, "provider": "anthropic", "model": "m", "usage": {}},
                        "attempts": [{"request": "q", "response": {"answer": answer, "request_messages": []},
                                      "check": {"status": "cited"}}],
                        "review_error": None, "blocked": False},
        "text_deep_v2": [], "vector_by_mode_v2": {},
    }
    now = datetime.now(timezone.utc)
    db = fake_mongo[config.MONGODB_DB]
    db["agentic_sessions"].insert_one({"_id": "s1", "title": "t", "created_at": now, "updated_at": now,
                                       "memory": {"messages": [], "turn_summaries": [], "feedback": [],
                                                  "turn_count": 1}, "active_run": None})
    db["agentic_runs"].insert_one({
        "_id": "r1", "session_id": "s1", "question": "Capex 3M 2018?", "status": "complete",
        "created_at": now, "updated_at": now, "settings": {},
        "result": {"answer": answer, "sources": [{"filename": "3M_2018_10K.pdf", "page": 61, "content": "x"}]},
        "events": [
            {"type": "plan", "search_query": "capex 3M 2018", "comment": "c",
             "planner_error": "RuntimeError: 401", "max_searches": 2},
            {"type": "retrieval", "iteration": 1, "search_query": "capex", "candidate_count": 2,
             "ranked_sources": []},
            {"type": "grading", "iteration": 1, "evaluated": True, "sufficient": True, "comment": "ok",
             "supporting_quotes": [{"ref": 1, "quote": "Purchases of PP&E"}]},
            {"type": "decision", "iteration": 1, "reason": "evidence_sufficient", "action": "stop"},
            {"type": "answer", "status": "generated", "sources": []},
        ],
    })
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.session_state["last_query_result"] = result
    at.session_state["agentic_session_id"] = "s1"
    at.session_state["page_nav"] = "agentic"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    texts = " ".join(str(getattr(el, "value", "")) for el in at.main)
    assert "Verificata" in texts or "Da verificare" in texts
    at.session_state["page_nav"] = "query"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    texts = " ".join(str(getattr(el, "value", "")) for el in at.main)
    assert "Verifica delle citazioni" in texts
    assert "Trust panel" not in texts


def test_query_mostra_risposta_senza_citazioni_con_avviso(fake_mongo):
    passages = [_passage(1)]
    answer = "Le spese sono state 1.577 milioni."
    result = {
        "query_v2": "Capex 3M 2018?", "rerank_instructions_v2": "", "filters_v2": None,
        "scope_v2": {"database": "rag", "collection": "pdf_chunks", "modes": ["auto"],
                     "rows": [], "text_index": "t", "unavailable": [], "mismatched": [], "unclassified": 0},
        "pipeline_v2": [{"$limit": 1}], "top_k_v2": 20, "relevance_threshold_v2": 0.15,
        "timings_ms_v2": {}, "atlas_results_v2": passages, "raw_v2": passages,
        "reranked_v2": passages, "rerank_mode_v2": "native", "top_rerank_v2": 0.8,
        "reviewed_v2": {"answer": {"answer": answer, "provider": "anthropic", "model": "m", "usage": {}},
                        "attempts": [{"request": "q", "response": {"answer": answer}, "check": {}}],
                        "review_error": None, "blocked": False},
        "text_deep_v2": [], "vector_by_mode_v2": {},
    }
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.session_state["last_query_result"] = result
    at.session_state["page_nav"] = "query"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(answer in md.value for md in at.markdown)
    assert any("Controllo dei riferimenti" in w.value for w in at.warning)
