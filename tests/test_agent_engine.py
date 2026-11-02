"""Motore dell'agente senza interfaccia: eventi in ordine, salvati in MongoDB."""
import mongomock
import pytest

from src import agent, agentic_memory, agentic_search, answer_quality, generation, search_scope

SCOPE = {"database": "rag", "collection": "pdf_chunks", "modes": ["auto"], "rows": []}


def _doc(n, content, score=0.9, filename="3M_2018_10K.pdf"):
    return {"_id": f"c{n}", "document_id": "d1", "filename": filename, "page_number": n,
            "content": content, "_rerank_score": score}


@pytest.fixture()
def env(monkeypatch):
    db = mongomock.MongoClient()["rag"]
    monkeypatch.setattr(search_scope, "require_scope", lambda mode: SCOPE)
    monkeypatch.setattr(generation, "llm_available", lambda: True)
    monkeypatch.setattr(answer_quality, "generate_answer", lambda request, passages: {
        "answer": "Capex 1.577 milioni [#1].", "provider": "x", "model": "m", "usage": {}, "truncated": False})
    monkeypatch.setattr(agentic_search, "rerank_union",
                        lambda coll, q, cands, *, top_n, instructions="": cands[:top_n])
    session = agentic_memory.create_session(db["agentic_sessions"])
    return db, session["_id"]


def _run(env, **kw):
    db, sid = env
    req = agent.AgentRequest(session_id=sid, **kw)
    events = list(agent.run_agent(req, runs=db["agentic_runs"], sessions=db["agentic_sessions"], collection=None))
    run = db["agentic_runs"].find_one({"_id": events[0]["run_id"]})
    return events, run


def test_due_ricerche_e_risposta(env, monkeypatch):
    docs = iter([[_doc(1, "capex (1,577)")], [_doc(2, "employed 93,516 people")]])
    monkeypatch.setattr(agentic_search, "retrieve_candidates", lambda c, q, **k: (lambda d: (d, d))(next(docs)))
    grades = iter([
        {"evaluated": True, "sufficient": False, "comment": "manca", "followup_query": "3M employees 2018",
         "elements": [{"element": "capex", "found": True, "ref": 1, "value": "1,577"},
                      {"element": "dipendenti", "found": False}]},
        {"evaluated": True, "sufficient": True, "comment": "ok", "followup_query": None,
         "elements": [{"element": "capex", "found": True, "ref": 1, "value": "1,577"},
                      {"element": "dipendenti", "found": True, "ref": 2, "value": "93,516"}]},
    ])
    monkeypatch.setattr(agentic_search, "assess_evidence", lambda *a, **k: next(grades))
    events, run = _run(env, question="Capex 3M 2018 e dipendenti?", retries=1)
    types = [e["type"] for e in events]
    assert types[0] == "start" and types[-1] == "done"
    assert types.count("retrieval") == 2 and "agent_rank_fusion" in types and "generating" in types
    assert run["status"] == "complete" and run["result"]["answer"] == "Capex 1.577 milioni [#1]."
    saved = [e["type"] for e in run["events"]]
    assert "generating" not in saved and saved.count("grading") == 2


def test_fuori_tema_senza_modello(env, monkeypatch):
    monkeypatch.setattr(agentic_search, "retrieve_candidates",
                        lambda c, q, **k: ([_doc(1, "x", 0.02)], [_doc(1, "x", 0.02)]))
    monkeypatch.setattr(agentic_search, "assess_evidence", lambda *a, **k: pytest.fail("non serve"))
    events, run = _run(env, question="Ricetta della pizza?")
    assert run["status"] == "blocked" and events[-1]["status"] == "blocked"


def test_chiede_l_anno(env, monkeypatch):
    docs = [_doc(1, "61 manufacturing facilities", filename="3M_2018_10K.pdf"),
            _doc(2, "64 manufacturing facilities", filename="3M_2020_10K.pdf")]
    monkeypatch.setattr(agentic_search, "retrieve_candidates", lambda c, q, **k: (docs, docs))
    monkeypatch.setattr(agentic_search, "_complete", lambda *a, **k: (
        '{"elements": [{"element": "stabilimenti", "found": true, "ref": 1, "value": "61 manufacturing facilities", '
        '"other_values": [{"ref": 2, "value": "64 manufacturing facilities"}]}], "comment": "", "followup_query": null}'))
    events, run = _run(env, question="Dove si trovano gli stabilimenti di 3M?")
    assert run["status"] == "needs_input"
    clar = next(e for e in events if e["type"] == "clarification")
    assert [o["year"] for o in clar["options"]] == [2020, 2018]


def test_errore_salvato(env, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("Atlas giù")
    monkeypatch.setattr(agentic_search, "retrieve_candidates", boom)
    events, run = _run(env, question="x")
    assert events[-2]["type"] == "error" and run["status"] == "failed"
