"""Banco di prova: domande, campione, valutazione e sintesi (senza Atlas né modelli)."""
import json
import time

import mongomock
import pytest
from starlette.testclient import TestClient

from src import benchmark, config, search_scope, web_services


def _write(tmp_path):
    rows = []
    for i, t in enumerate(["metrics-generated", "domain-relevant", "novel-generated"] * 4):
        rows.append({"financebench_id": f"id{i:02d}", "company": "3M", "doc_name": "3M_2018_10K", "question_type": t,
                     "question": f"Domanda {i}?", "answer": "$1577.00",
                     "evidence": [{"doc_name": "3M_2018_10K", "evidence_page_num": "59"}]})
    path = tmp_path / "q.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return path


def test_pagine_da_zero_a_uno_e_campione_bilanciato(tmp_path):
    qs = benchmark.load_questions(_write(tmp_path))
    assert qs[0]["pages"] == [60] and qs[0]["filename"] == "3M_2018_10K.pdf"
    picked = benchmark.pick(qs, 6)
    assert [q["type"] for q in picked].count("domain-relevant") == 2


def test_valutazione_e_sintesi(monkeypatch):
    item = {"id": "a", "company": "3M", "type": "metrics-generated", "question": "Capex?", "answer": "$1577.00",
            "filename": "3M_2018_10K.pdf", "pages": [60]}
    monkeypatch.setitem(benchmark.EVALUATORS, "retrieval", lambda it: {"ms": 500, **benchmark._hits(
        [{"filename": "x.pdf", "page_number": 1}, {"filename": "3M_2018_10K.pdf", "page_number": 60}], it)})
    monkeypatch.setitem(benchmark.EVALUATORS, "query", lambda it: {"ms": 9000, "answer": "1.577 milioni [#1]", "status": "answered",
                                                                 **benchmark._hits([{"filename": "3M_2018_10K.pdf", "page": 60}], it)})
    monkeypatch.setitem(benchmark.EVALUATORS, "agent", lambda it: {"ms": 20000, "answer": "", "status": "abstained",
                                                                 "searches": 2, **benchmark._hits([], it)})
    monkeypatch.setattr(benchmark.agentic_search, "_complete", lambda *a, **k: '{"verdict": "correct", "reason": "uguale"}')
    row = benchmark.evaluate(item, ["retrieval", "query", "agent"])
    assert row["retrieval"]["page_rank"] == 2 and row["query"]["verdict"] == "correct"
    assert row["agent"]["verdict"] == "abstained"
    s = benchmark.summarize([row], ["retrieval", "query", "agent"])
    assert s["retrieval"]["page_at_1"] == 0 and s["retrieval"]["page_at_5"] == 1
    assert s["query"]["correct"] == 1 and s["agent"]["abstained"] == 1 and s["agent"]["multi_search"] == 1


def test_errore_di_un_percorso_non_ferma_la_domanda(monkeypatch):
    item = {"id": "a", "company": "3M", "type": "x", "question": "q", "answer": "r", "filename": "f.pdf", "pages": []}
    def boom(it):
        raise RuntimeError("gateway 403")
    monkeypatch.setitem(benchmark.EVALUATORS, "query", boom)
    row = benchmark.evaluate(item, ["query"])
    assert "gateway 403" in row["query"]["error"]
    assert benchmark.summarize([row], ["query"])["query"]["errors"] == 1


@pytest.fixture()
def api(monkeypatch, tmp_path):
    db = mongomock.MongoClient()["rag"]
    db["pdf_chunks"].insert_one({"filename": "3M_2018_10K.pdf", "embedding_mode": "auto"})
    monkeypatch.setattr(config, "get_db", lambda: db)
    monkeypatch.setattr(config, "get_collection", lambda: db["pdf_chunks"])
    monkeypatch.setattr(search_scope, "load_active_mode", lambda default=None: "auto")
    monkeypatch.setattr(benchmark, "QUESTIONS_PATH", _write(tmp_path))
    monkeypatch.setattr(benchmark, "evaluate", lambda item, paths: {"id": item["id"], "type": item["type"], "company": "3M",
                                                                   "question": "q", "reference": "r", "filename": "f.pdf", "pages": [],
                                                                   "query": {"ms": 1, "verdict": "correct", "status": "answered", "sources": []}})
    import server
    return TestClient(server.app), db


def test_api_misura_completa(api):
    c, db = api
    ov = c.get("/api/benchmark").json()
    assert ov["available"] == 12 and ov["runs"] == []
    run_id = c.post("/api/benchmark", json={"paths": ["query"], "size": 6}).json()["id"]
    for _ in range(50):
        run = c.get(f"/api/benchmark/{run_id}").json()
        if run["status"] != "running":
            break
        time.sleep(0.05)
    assert run["status"] == "done" and run["progress"]["done"] == 6 and run["summary"]["query"]["correct"] == 1
    assert c.post("/api/benchmark", json={"paths": [], "size": 6}).status_code == 400


def test_verifica_componenti(monkeypatch):
    db = mongomock.MongoClient()["rag"]
    monkeypatch.setattr(config, "get_db", lambda: db)
    monkeypatch.setattr(config, "get_collection", lambda: db["pdf_chunks"])
    monkeypatch.setattr(search_scope, "load_active_mode", lambda default=None: "auto")
    monkeypatch.setattr(db, "command", lambda *a, **k: {"ok": 1}, raising=False)
    import src.generation as gen
    monkeypatch.setattr(gen, "generate_answer", lambda q, c: (_ for _ in ()).throw(RuntimeError("RBAC: access denied")))
    web_services._HEALTH.clear()
    out = web_services.health(force=True)
    names = {c["name"]: c for c in out["checks"]}
    assert out["ok"] is False
    assert names["Documenti"]["ok"] is False and "RBAC" in names["Modello della risposta"]["detail"]
