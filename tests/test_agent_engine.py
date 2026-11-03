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


def test_pagina_chiusa_a_meta_ricerca_non_resta_in_corso(env, monkeypatch):
    db, sid = env
    monkeypatch.setattr(agentic_search, "retrieve_candidates", lambda c, q, **k: ([_doc(1, "capex")], [_doc(1, "capex")]))
    stream = agent.run_agent(agent.AgentRequest(session_id=sid, question="Capex 3M 2018?"),
                             runs=db["agentic_runs"], sessions=db["agentic_sessions"], collection=None)
    run_id = next(stream)["run_id"]
    next(stream)  # piano
    stream.close()  # come quando il browser chiude la connessione
    run = db["agentic_runs"].find_one({"_id": run_id})
    assert run["status"] == "failed" and "interrotta" in run["events"][-1]["comment"]
    assert db["agentic_sessions"].find_one({"_id": sid})["active_run"] is None


def test_ricerca_conclusa_non_diventa_fallita(env, monkeypatch):
    db, sid = env
    monkeypatch.setattr(agentic_search, "retrieve_candidates",
                        lambda c, q, **k: ([_doc(1, "x", 0.02)], [_doc(1, "x", 0.02)]))
    events, run = _run(env, question="Ricetta della pizza?")
    agentic_memory.fail_run(db["agentic_runs"], db["agentic_sessions"], session_id=sid, run_id=run["_id"], message="tardi")
    assert db["agentic_runs"].find_one({"_id": run["_id"]})["status"] == "blocked"


def test_domanda_doppia_una_ricerca_per_parte_e_fonti_alternate(env, monkeypatch):
    adobe = [_doc(10 + i, f"Adobe R&D {i}", filename="ADOBE_2022_10K.pdf") for i in range(6)]
    tre_m = [_doc(20 + i, f"3M capex {i}", filename="3M_2018_10K.pdf") for i in range(6)]
    monkeypatch.setattr(agentic_search, "plan_search", lambda q, h, m=None: {
        "search_query": q, "search_queries": ["Adobe R&D 2022", "3M capex 2018"], "comment": "", "error": None})
    searched = []

    def retrieve(c, q, **k):
        searched.append(q)
        docs = adobe if "Adobe" in q else tre_m
        return docs, docs
    monkeypatch.setattr(agentic_search, "retrieve_candidates", retrieve)
    seen_pools = []

    def grade(question, query, pool, executed):
        seen_pools.append([d["filename"] for d in pool[:8]])
        return {"evaluated": True, "sufficient": True, "comment": "", "followup_query": None,
                "elements": [{"element": "R&D Adobe", "found": True}, {"element": "capex 3M", "found": True}]}
    monkeypatch.setattr(agentic_search, "assess_evidence", grade)
    events, run = _run(env, question="Quanto ha speso Adobe in R&D nel 2022? E quali sono le spese in conto capitale di 3M nel 2018?")
    assert searched == ["Adobe R&D 2022", "3M capex 2018"]
    plan = next(e for e in events if e["type"] == "plan")
    assert plan["sub_queries"] == ["Adobe R&D 2022", "3M capex 2018"]
    # Le fonti davanti al valutatore alternano le due aziende.
    assert seen_pools[0][:4] == ["ADOBE_2022_10K.pdf", "3M_2018_10K.pdf"] * 2
    assert run["status"] == "complete"


def test_mancano_due_elementi_una_ricerca_per_elemento(env, monkeypatch):
    searched = []
    monkeypatch.setattr(agentic_search, "retrieve_candidates",
                        lambda c, q, **k: (searched.append(q) or ([_doc(len(searched), q)], [_doc(len(searched), q)])))
    grades = iter([
        {"evaluated": True, "sufficient": False, "comment": "mancano", "followup_query": "tutto insieme",
         "missing_elements": ["R&D Adobe 2022", "capex 3M 2018"],
         "elements": [{"element": "R&D Adobe 2022", "found": False}, {"element": "capex 3M 2018", "found": False}]},
        {"evaluated": True, "sufficient": True, "comment": "ok", "followup_query": None,
         "elements": [{"element": "R&D Adobe 2022", "found": True}, {"element": "capex 3M 2018", "found": True}]},
    ])
    monkeypatch.setattr(agentic_search, "assess_evidence", lambda *a, **k: next(grades))
    _run(env, question="Spese di ricerca di Adobe 2022 insieme al capex di 3M 2018", retries=5)
    assert searched[1:] == ["R&D Adobe 2022", "capex 3M 2018"]


def test_pianificatore_restituisce_le_ricerche_separate(monkeypatch):
    monkeypatch.setattr(agentic_search, "_complete", lambda *a, **k: (
        '{"search_query": "Adobe R&D 2022 e capex 3M 2018", "search_queries": ["Adobe R&D 2022", "3M capex 2018", "3M capex 2018"], "comment": "due ricerche"}'))
    plan = agentic_search.plan_search("domanda doppia", [], [])
    assert plan["search_queries"] == ["Adobe R&D 2022", "3M capex 2018"]


def _niente(query):
    return {"evaluated": True, "sufficient": False, "comment": "non pertinente", "followup_query": query,
            "elements": [{"element": "ricetta della pizza", "found": False}]}


def test_stessi_passaggi_si_ferma_e_non_cita_fonti(env, monkeypatch):
    # Pochi passaggi nella modalità: ogni ricerca restituisce gli stessi.
    docs = [_doc(1, "MongoDB platform", 0.25), _doc(2, "data management", 0.24)]
    monkeypatch.setattr(agentic_search, "retrieve_candidates", lambda c, q, **k: (docs, docs))
    queries = iter(["pizza ricetta", "pizza impasto", "pizza cottura"])
    monkeypatch.setattr(agentic_search, "assess_evidence", lambda *a, **k: _niente(next(queries)))
    monkeypatch.setattr(answer_quality, "generate_answer", lambda *a, **k: pytest.fail("non serve"))
    events, run = _run(env, question="Come si prepara la pizza napoletana?", retries=5)
    assert [e["type"] for e in events].count("retrieval") == 2
    stop = [e for e in events if e["type"] == "decision" and e["action"] == "stop"]
    assert stop[-1]["reason"] == "no_new_passages"
    assert run["status"] == "blocked" and run["result"]["sources"] == []


def test_niente_trovato_in_due_ricerche_si_ferma(env, monkeypatch):
    counter = iter(range(1, 100))
    monkeypatch.setattr(agentic_search, "retrieve_candidates",
                        lambda c, q, **k: (lambda n: ([_doc(n, "revenue", 0.3)], [_doc(n, "revenue", 0.3)]))(next(counter)))
    queries = iter(["pizza ricetta", "pizza impasto", "pizza cottura"])
    monkeypatch.setattr(agentic_search, "assess_evidence", lambda *a, **k: _niente(next(queries)))
    monkeypatch.setattr(answer_quality, "generate_answer", lambda *a, **k: pytest.fail("non serve"))
    events, run = _run(env, question="Come si prepara la pizza napoletana?", retries=5)
    assert [e["type"] for e in events].count("retrieval") == 2
    assert [e for e in events if e["type"] == "decision"][-1]["reason"] == "not_in_documents"
    assert run["status"] == "blocked" and "non sembrano contenerl" in run["result"]["answer"]


def test_valori_di_altri_anni_non_fermano_la_ricerca():
    grade = {"evaluated": True, "elements": [{"element": "stabilimenti 2018", "found": False,
                                              "values": [{"ref": 1, "value": "80 facilities", "year": 2017}]}]}
    assert agent._nothing_found(grade) is False
    assert agent._nothing_found({"evaluated": True, "elements": [{"element": "x", "found": False}]}) is True
