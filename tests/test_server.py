"""Server web: pagine, API JSON e flussi NDJSON, con Atlas e modello simulati."""
import json

import mongomock
import pytest
from starlette.testclient import TestClient

from src import agentic_search, answer_quality, config, generation, search_scope, web_services

DOCS = [{"_id": "c1", "document_id": "d1", "filename": "3M_2018_10K.pdf", "page_number": 60,
         "content": "Purchases of property, plant and equipment (1,577)", "_rerank_score": 0.8}]


@pytest.fixture()
def client(monkeypatch):
    db = mongomock.MongoClient()["rag"]
    monkeypatch.setattr(config, "get_db", lambda: db)
    monkeypatch.setattr(config, "get_collection", lambda: db["pdf_chunks"])
    monkeypatch.setattr(search_scope, "require_scope", lambda m: {"database": "rag", "collection": "pdf_chunks", "modes": ["auto"], "rows": []})
    monkeypatch.setattr(search_scope, "load_active_mode", lambda default=None: "auto")
    monkeypatch.setattr(generation, "llm_available", lambda: True)
    monkeypatch.setattr(agentic_search, "retrieve_candidates", lambda c, q, **k: (DOCS, DOCS))
    monkeypatch.setattr(agentic_search, "rerank_union", lambda coll, q, cands, *, top_n, instructions="": cands)
    monkeypatch.setattr(agentic_search, "assess_evidence", lambda *a, **k: {
        "evaluated": True, "sufficient": True, "comment": "", "followup_query": None,
        "elements": [{"element": "capex", "found": True, "ref": 1, "value": "1,577"}]})
    monkeypatch.setattr(answer_quality, "generate_answer", lambda r, p: {
        "answer": "Capex 1.577 milioni di dollari [#1].", "model": "m", "usage": {}, "truncated": False})
    import server
    return TestClient(server.app), db


def _events(response):
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


def test_pagina_e_statici(client):
    c, _ = client
    assert "RAG Demo" in c.get("/").text
    assert c.get("/web/app.js").status_code == 200 and c.get("/web/style.css").status_code == 200


def test_sessione_vuota_riusata(client):
    c, _ = client
    a, b = c.post("/api/sessions").json(), c.post("/api/sessions").json()
    assert a["id"] == b["id"]


def test_agente_in_streaming_e_vista(client):
    c, _ = client
    sid = c.post("/api/sessions").json()["id"]
    ev = _events(c.post("/api/agent", json={"session_id": sid, "question": "Capex 3M 2018?"}))
    assert ev[0] == {"type": "session", "session_id": sid} and ev[-1]["type"] == "done"
    run_id = next(e["run_id"] for e in ev if e["type"] == "start")
    view = c.get(f"/api/runs/{run_id}").json()
    assert view["status"] == "complete" and view["cited"] == [1] and view["verdict"]["kind"] == "ok"
    assert c.get(f"/api/sessions/{sid}/runs").json()[0]["id"] == run_id
    assert c.post("/api/sessions").json()["id"] != sid  # la sessione usata non viene riusata


def test_domanda_vuota(client):
    c, _ = client
    assert c.post("/api/agent", json={"question": " "}).status_code == 400
    assert c.post("/api/query", json={"question": ""}).status_code == 400


def test_errore_nel_flusso_diventa_evento(client, monkeypatch):
    c, _ = client
    def boom(*a, **k):
        raise RuntimeError("Atlas giù")
        yield
    monkeypatch.setattr(web_services, "run_query", boom)
    ev = _events(c.post("/api/query", json={"question": "x"}))
    assert ev == [{"type": "error", "message": "RuntimeError: Atlas giù"}]


def test_riparti_da_zero(client, monkeypatch):
    c, db = client
    db["pdf_chunks"].insert_many([{"content": "a"}, {"content": "b"}])
    db["agentic_sessions"].insert_one({"_id": "s1"})
    db["agentic_runs"].insert_one({"_id": "r1", "session_id": "s1"})
    seen = {}
    def fake_setup(recreate=False):
        seen["recreate"] = recreate
        yield {"type": "done", "created": [], "updated": [], "unchanged": [], "errors": []}
    monkeypatch.setattr(web_services, "setup_database", fake_setup)
    monkeypatch.setattr(web_services, "GridFS", lambda db, collection: type("G", (), {"find": lambda self: []})())
    assert c.post("/api/admin/reset", json={"sessions": True}).status_code == 400  # senza conferma
    ev = _events(c.post("/api/admin/reset", json={"confirm": "AZZERA", "sessions": True, "recreate_indexes": True}))
    assert ev[-1]["type"] == "done", ev
    assert seen["recreate"] is True
    assert db["pdf_chunks"].count_documents({}) == 0
    assert db["agentic_sessions"].count_documents({}) == 0 and db["agentic_runs"].count_documents({}) == 0


def test_sessioni_con_data_e_cancellazione(client):
    c, db = client
    s = c.post("/api/sessions").json()
    assert s["title"].startswith("Sessione del ") and " alle " in s["title"]
    c.post("/api/agent", json={"session_id": s["id"], "question": "Capex 3M 2018?"})
    row = c.get("/api/sessions").json()[0]
    assert row["first_question"] == "Capex 3M 2018?" and row["title"] == s["title"] and row["turns"] == 1
    assert c.get("/api/admin/sessions").json()["sessions"] == 1
    r = c.delete("/api/admin/sessions").json()
    assert r["sessions_deleted"] == 1 and r["runs_deleted"] == 1


def test_valore_solo_con_la_risposta_definitiva(client, monkeypatch):
    c, db = client
    docs = [{"_id": "a", "filename": "3M_2018_10K.pdf", "page_number": 1, "content": "61 manufacturing facilities", "_rerank_score": .9},
            {"_id": "b", "filename": "3M_2020_10K.pdf", "page_number": 1, "content": "64 manufacturing facilities", "_rerank_score": .8}]
    monkeypatch.setattr(agentic_search, "retrieve_candidates", lambda coll, q, **k: (docs, docs))
    monkeypatch.setattr(agentic_search, "assess_evidence", lambda *a, **k: {
        "evaluated": True, "sufficient": True, "comment": "", "followup_query": None,
        "elements": [{"element": "stabilimenti", "found": True, "ref": 1, "value": "61",
                      "values": [{"year": 2018, "filename": "3M_2018_10K.pdf", "value": "61 manufacturing facilities"},
                                 {"year": 2020, "filename": "3M_2020_10K.pdf", "value": "64 manufacturing facilities"}]}]})
    sid = c.post("/api/sessions").json()["id"]
    ev = _events(c.post("/api/agent", json={"session_id": sid, "question": "Stabilimenti di 3M?"}))
    parent = next(e["run_id"] for e in ev if e["type"] == "start")
    view = c.get(f"/api/runs/{parent}").json()
    assert view["status"] == "needs_input" and view.get("value") is None
    ev = _events(c.post("/api/agent", json={"session_id": sid, "question": "Stabilimenti di 3M? (anno 2020)",
                                            "clarification": {"parent_run_id": parent, "context": "L’utente ha scelto l’anno 2020."}}))
    child = next(e["run_id"] for e in ev if e["type"] == "start")
    view = c.get(f"/api/runs/{child}").json()
    assert view["status"] == "complete" and view["value"]["title"].startswith("Ti ha chiesto l’anno")
