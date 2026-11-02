"""Ciclo agentico end-to-end con modelli e Atlas simulati."""
from pathlib import Path

import mongomock
import pytest
from streamlit.testing.v1 import AppTest

from src import agentic_search, answer_quality, config, generation, search_scope

APP = Path(__file__).resolve().parent.parent / "app.py"
SCOPE = {"database": "rag", "collection": "pdf_chunks", "modes": ["auto"], "rows": [],
         "text_index": "t", "unavailable": [], "mismatched": [], "unclassified": 0}


def _doc(n, content, score):
    return {"_id": f"c{n}", "document_id": "d1", "filename": "3M_2018_10K.pdf", "page_number": n,
            "content": content, "_rerank_score": score, "_hybrid_rank": n}


@pytest.fixture()
def agent(monkeypatch):
    client = mongomock.MongoClient()
    client[config.MONGODB_DB][config.MONGODB_COLLECTION].insert_one(
        {"document_id": "d1", "filename": "3M_2018_10K.pdf", "embedding_mode": "auto", "content": "x", "chunk_hash": "h"})
    monkeypatch.setattr(config, "get_client", lambda: client)
    monkeypatch.setattr(search_scope, "require_scope", lambda mode: SCOPE)
    monkeypatch.setattr(generation, "llm_available", lambda: True)
    monkeypatch.setattr(agentic_search, "plan_search",
                        lambda q, h, m=None: pytest.fail("prima domanda: il pianificatore non va chiamato"))
    calls = {"retrieve": [], "assess": [], "union": [], "generate": 0}

    def fake_generate(request, passages):
        calls["generate"] += 1
        return {"answer": "Capex 2018 1.577 milioni [a (p. 1)](#1), 2017 1.373 milioni [a (p. 2)](#2).",
                "provider": "anthropic", "model": "m", "usage": {}, "truncated": False}
    monkeypatch.setattr(answer_quality, "generate_answer", fake_generate)

    def fake_union(coll, query, candidates, *, top_n, instructions=""):
        calls["union"].append((query, [c["_id"] for c in candidates]))
        return sorted(candidates, key=lambda c: c["_id"])[:top_n]
    monkeypatch.setattr(agentic_search, "rerank_union", fake_union)
    return client, calls


def _run(question):
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.session_state["page_nav"] = "agentic"
    at.run()
    at.session_state["agentic_question_selection"] = "Domanda libera"
    at.session_state["agentic_question"] = question
    at.session_state["agentic_question_draft"] = question
    at.run()
    at.button(key="agentic_submit").click().run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_seconda_ricerca_mirata_e_valutazione_cumulativa(agent, monkeypatch):
    client, calls = agent

    def fake_retrieve(coll, query, **kw):
        calls["retrieve"].append(query)
        docs = [_doc(1, "PP&E purchases 2018 (1,577)", 0.9)] if len(calls["retrieve"]) == 1 \
            else [_doc(2, "PP&E purchases 2017 (1,373)", 0.8)]
        return docs, docs
    monkeypatch.setattr(agentic_search, "retrieve_candidates", fake_retrieve)

    def fake_assess(question, query, candidates, previous=None):
        calls["assess"].append(([c["_id"] for c in candidates], list(previous or [])))
        if len(calls["assess"]) == 1:
            return {"evaluated": True, "sufficient": False, "comment": "manca il 2017",
                    "followup_query": "3M capex 2017 purchases of PP&E", "missing_elements": ["2017"],
                    "supporting_refs": [1], "supporting_quotes": []}
        return {"evaluated": True, "sufficient": True, "comment": "ok", "followup_query": None,
                "missing_elements": [], "supporting_refs": [1, 2],
                "supporting_quotes": [{"ref": 2, "quote": "PP&E purchases 2017"}]}
    monkeypatch.setattr(agentic_search, "assess_evidence", fake_assess)

    at = _run("Capex di 3M nel 2018 e nel 2017?")
    page = " ".join(str(getattr(el, "value", "")) for el in at.main)
    assert "Verificata: 2 numeri su 2" in page
    assert "La risposta soddisfa" not in page          # modalità automatica: niente feedback
    # La seconda ricerca usa solo l'elemento mancante, non ripete la domanda.
    assert calls["retrieve"][1] == "3M capex 2017 purchases of PP&E"
    # La seconda valutazione vede l'insieme di entrambe le ricerche.
    assert calls["assess"][1][0] == ["c1", "c2"]
    assert len(calls["assess"][1][1]) == 2
    # Prima domanda della sessione: niente pianificatore, la domanda è usata così com'è.
    assert calls["retrieve"][0] == "Capex di 3M nel 2018 e nel 2017?"
    assert calls["union"] and calls["union"][0][0] == "Capex di 3M nel 2018 e nel 2017?"
    assert calls["generate"] >= 1  # la risposta finta cita #2 con un solo passaggio: c'è il secondo tentativo
    run = client[config.MONGODB_DB]["agentic_runs"].find_one()
    assert run["status"] == "complete"
    assert not any(e["type"] == "source_review" for e in run["events"])
    plan = next(e for e in run["events"] if e["type"] == "plan")
    assert plan["planner_skipped"]
    assert all("duration_ms" in e for e in run["events"] if e["type"] in {"retrieval", "grading", "answer"})


def test_domanda_fuori_tema_non_chiama_il_modello(agent, monkeypatch):
    client, calls = agent
    monkeypatch.setattr(agentic_search, "retrieve_candidates",
                        lambda coll, query, **kw: ([_doc(1, "x", 0.02)], [_doc(1, "x", 0.02)]))
    monkeypatch.setattr(agentic_search, "assess_evidence",
                        lambda *a, **k: pytest.fail("il valutatore non deve essere chiamato"))
    at = _run("Come si fa la pizza napoletana?")
    assert calls["generate"] == 0
    run = client[config.MONGODB_DB]["agentic_runs"].find_one()
    assert run["status"] == "blocked"
    assert any(e.get("reason") == "off_topic" for e in run["events"])
    assert any("fuori tema" in str(getattr(el, "value", "")) for el in at.main)


def test_risposta_troncata_viene_segnalata():
    check = answer_quality.inspect_answer("Il valore è 1.373 [a (p. 2)](#1) e", 3, truncated=True)
    assert check["status"] == "truncated"
    assert "LLM_MAX_TOKENS" in answer_quality.explain_answer_check("truncated")


def test_numero_aggiunto_dal_modello_risulta_non_trovato():
    from src.citation_check import check_citations
    sources = [{"content": "employed 93,516 people", "filename": "a.pdf"}]
    rows = check_citations("Dipendenti 93.516, di cui 37.412 negli USA [a (p. 4)](#1).", sources)
    assert [r["status"] for r in rows] == ["found", "not_found"]


def test_agente_mostra_il_tentativo_senza_link_come_la_query(agent, monkeypatch):
    client, calls = agent
    answers = iter([
        "3M ha 61 stabilimenti [1](https://www.3m.com/locations).",   # indirizzo web: mai mostrato
        "La sede produttiva principale è St. Paul.",                   # senza citazioni: mostrato con avviso
    ])
    monkeypatch.setattr(answer_quality, "generate_answer", lambda request, passages: {
        "answer": next(answers), "provider": "ollama", "model": "m", "usage": {}, "truncated": False})
    docs = [_doc(1, "3M operates 61 manufacturing facilities in 29 states", 0.9)]
    monkeypatch.setattr(agentic_search, "retrieve_candidates", lambda coll, query, **kw: (docs, docs))
    monkeypatch.setattr(agentic_search, "assess_evidence", lambda *a, **k: {
        "evaluated": True, "sufficient": True, "comment": "ok", "followup_query": None,
        "elements": [{"element": "stabilimenti", "found": True, "ref": 1, "value": "61"}],
        "supporting_quotes": [{"ref": 1, "quote": "61"}]})
    at = _run("3M’s manufacturing operations where ?")
    run = client[config.MONGODB_DB]["agentic_runs"].find_one()
    assert run["status"] == "complete"
    assert run["result"]["answer"] == "La sede produttiva principale è St. Paul."
    page = " ".join(str(getattr(el, "value", "")) for el in at.main)
    assert "Da verificare" in page and "non cita i passaggi" in page
    assert "www.3m.com" not in page.split("Dettaglio")[0]


def test_due_tentativi_con_link_restano_bloccati():
    from src.answer_quality import best_publishable_attempt
    reviewed = {"attempts": [
        {"check": {"status": "external_links"}, "response": {"answer": "x https://a.com"}},
        {"check": {"status": "external_links"}, "response": {"answer": "y www.b.com"}},
    ]}
    assert best_publishable_attempt(reviewed) is None


def test_agente_chiede_l_anno_e_poi_risponde_con_il_documento_scelto(agent, monkeypatch):
    client, calls = agent
    docs = [{**_doc(1, "3M operates 61 manufacturing facilities in 29 states", 0.9), "filename": "3M_2018_10K.pdf"},
            {**_doc(2, "3M operates 64 manufacturing facilities in 29 states", 0.8), "filename": "3M_2020_10K.pdf"}]
    seen_filters = []

    def fake_retrieve(coll, query, filters=None, **kw):
        seen_filters.append(filters)
        chosen = [d for d in docs if not filters or d["filename"] == filters.get("filename")]
        return chosen, chosen
    monkeypatch.setattr(agentic_search, "retrieve_candidates", fake_retrieve)
    real_assess = agentic_search.assess_evidence

    def fake_assess(question, query, candidates, previous=None):
        if len(candidates) == 2:
            monkeypatch.setattr(agentic_search, "_complete", lambda *a, **k: (
                '{"elements": [{"element": "stabilimenti USA", "found": true, "ref": 1, "value": "61 manufacturing facilities", '
                '"other_values": [{"ref": 2, "value": "64 manufacturing facilities"}]}], "comment": "", "followup_query": null}'))
        else:
            monkeypatch.setattr(agentic_search, "_complete", lambda *a, **k: (
                '{"elements": [{"element": "stabilimenti USA", "found": true, "ref": 1, "value": "64 manufacturing facilities", '
                '"other_values": []}], "comment": "", "followup_query": null}'))
        return real_assess(question, query, candidates, previous)
    monkeypatch.setattr(agentic_search, "assess_evidence", fake_assess)
    monkeypatch.setattr(agentic_search, "plan_search",
                        lambda q, h, m=None: {"search_query": q, "comment": "", "error": None})

    at = _run("Dove si trovano gli stabilimenti produttivi di 3M?")
    run = client[config.MONGODB_DB]["agentic_runs"].find_one()
    assert run["status"] == "needs_input" and calls["generate"] == 0
    page = " ".join(str(getattr(el, "value", "")) for el in at.main)
    assert "L’agente ha una domanda per te" in page
    # Scelgo il 2020: nuova ricerca filtrata sul documento 2020, poi la risposta.
    at.button(key=f"clar_{run['_id']}_2020").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert seen_filters[-1] == {"filename": "3M_2020_10K.pdf"}
    child = client[config.MONGODB_DB]["agentic_runs"].find_one({"parent_run_id": run["_id"]})
    assert child["status"] == "complete" and "(anno 2020)" in child["question"]
    assert calls["generate"] >= 1  # la risposta finta cita #2 con un solo passaggio: c'è il secondo tentativo


def test_dati_assenti_verdetto_non_presente(agent, monkeypatch):
    client, calls = agent
    docs = [_doc(1, "General Mills sells refrigerated dough and frozen pizza", 0.4)]
    monkeypatch.setattr(agentic_search, "retrieve_candidates", lambda coll, query, **kw: (docs, docs))
    monkeypatch.setattr(agentic_search, "assess_evidence", lambda *a, **k: {
        "evaluated": True, "sufficient": False, "comment": "", "followup_query": None,
        "elements": [{"element": "ricetta della pizza", "found": False, "ref": None, "value": ""}]})
    monkeypatch.setattr(answer_quality, "generate_answer", lambda request, passages: {
        "answer": "I passaggi citano solo pizze surgelate [#1], nessuna ricetta.", "provider": "x", "model": "m",
        "usage": {}, "truncated": False})
    at = _run("Come si prepara la pizza napoletana?")
    page = " ".join(str(getattr(el, "value", "")) for el in at.main)
    assert "Non presente nei documenti" in page and "non risponde a memoria" in page
    assert "Da verificare" not in page


def test_dato_mancante_nei_documenti_e_parziale_non_da_verificare(agent, monkeypatch):
    client, calls = agent
    docs = [_doc(1, "3M operates 80 manufacturing facilities in 29 states", 0.9)]
    monkeypatch.setattr(agentic_search, "retrieve_candidates", lambda coll, query, **kw: (docs, docs))
    monkeypatch.setattr(agentic_search, "assess_evidence", lambda *a, **k: {
        "evaluated": True, "sufficient": False, "comment": "", "followup_query": None,
        "elements": [{"element": "numero stabilimenti", "found": True, "ref": 1, "value": "80"},
                     {"element": "località specifiche", "found": False, "ref": None, "value": ""}]})
    monkeypatch.setattr(answer_quality, "generate_answer", lambda request, passages: {
        "answer": "3M aveva 80 stabilimenti in 29 Stati [#1]; le località non sono elencate.", "provider": "x",
        "model": "m", "usage": {}, "truncated": False})
    at = _run("Dove si trovano gli stabilimenti di 3M nel 2017?")
    page = " ".join(str(getattr(el, "value", "")) for el in at.main)
    assert "Parziale: 1 dati su 2" in page and "località specifiche" in page
    assert "Da verificare" not in page


def test_numeri_verificati_mai_non_presente(agent, monkeypatch):
    client, calls = agent
    docs = [_doc(1, "The Company operates 61 manufacturing facilities in 29 states", 0.9)]
    monkeypatch.setattr(agentic_search, "retrieve_candidates", lambda coll, query, **kw: (docs, docs))
    monkeypatch.setattr(agentic_search, "assess_evidence", lambda *a, **k: {
        "evaluated": True, "sufficient": False, "comment": "", "followup_query": None,
        "elements": [{"element": "località degli stabilimenti", "found": False, "ref": None, "value": ""}]})
    monkeypatch.setattr(answer_quality, "generate_answer", lambda request, passages: {
        "answer": "Negli Stati Uniti 61 stabilimenti in 29 Stati [#1].", "provider": "x",
        "model": "m", "usage": {}, "truncated": False})
    at = _run("Dove si trovano gli stabilimenti produttivi di 3M?")
    page = " ".join(str(getattr(el, "value", "")) for el in at.main)
    assert "Non presente nei documenti" not in page and "Parziale" in page
