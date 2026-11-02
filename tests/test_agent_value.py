from src.agent_value import agent_value_html, summarize_run


def _run(events, question="Dove si trovano gli stabilimenti produttivi di 3M?"):
    return {"question": question, "events": events}


SEARCH1 = {"type": "retrieval", "iteration": 1, "search_query": "q", "duration_ms": 1200}


def test_chiede_l_anno():
    data = summarize_run(_run([
        {"type": "plan", "planner_skipped": True}, SEARCH1,
        {"type": "grading", "iteration": 1, "sufficient": True, "duration_ms": 8000,
         "elements": [{"element": "stabilimenti", "found": True}]},
        {"type": "clarification", "options": [{"year": 2022}, {"year": 2020}, {"year": 2017}]},
        {"type": "decision", "action": "stop", "reason": "needs_clarification"},
        {"type": "answer", "status": "needs_input"}]))
    assert data["title"] == "Ha chiesto quale anno usare"
    assert "2022, 2020, 2017" in data["agent"]
    assert data["steps"][-1]["icon"] == "❓"


def test_seconda_ricerca_porta_tutti_i_dati():
    data = summarize_run(_run([
        {"type": "plan", "planner_skipped": True}, SEARCH1,
        {"type": "grading", "iteration": 1, "sufficient": False,
         "elements": [{"element": "capex", "found": True}, {"element": "dipendenti", "found": False}]},
        {"type": "retrieval", "iteration": 2, "search_query": "3M employees 2018"},
        {"type": "grading", "iteration": 2, "sufficient": True,
         "elements": [{"element": "capex", "found": True}, {"element": "dipendenti", "found": True}]},
        {"type": "answer", "status": "generated", "attempts": 1}]))
    assert data["title"] == "Ha cercato il dato mancante"
    assert data["standard"] == "Risponderebbe con 1 dato su 2"
    assert data["standard_found"] == {"capex"} and data["agent_found"] == {"capex", "dipendenti"}


def test_memoria():
    data = summarize_run(_run([
        {"type": "plan", "planner_skipped": False, "planned_query": "Stabilimenti produttivi di 3M nel 2017",
         "memory": {"turns": 2, "read_ms": 3.2, "previous_questions": ["Dove sono gli stabilimenti di 3M?", "(anno 2020)"]}},
        SEARCH1,
        {"type": "grading", "iteration": 1, "sufficient": True, "elements": [{"element": "x", "found": True}]},
        {"type": "answer", "status": "generated", "attempts": 1}], question="E nel 2017?"))
    assert data["title"] == "Ha ricordato la conversazione, salvata in MongoDB"
    assert "E nel 2017?" in data["standard"] and "2 domande precedenti" in data["agent"]
    assert data["memory"]["previous_questions"][0].startswith("Dove")
    assert any("MongoDB" in s["text"] for s in data["steps"])


def test_valore_dopo_la_scelta_dell_anno():
    parent = {"events": [{"type": "clarification", "options": [{"year": 2020}, {"year": 2018}]}]}
    data = summarize_run(_run([
        {"type": "plan", "planner_skipped": True, "feedback_context": "L’utente ha scelto l’anno 2020."}, SEARCH1,
        {"type": "grading", "iteration": 1, "sufficient": True, "elements": [{"element": "x", "found": True}]},
        {"type": "answer", "status": "generated", "attempts": 1}], question="Stabilimenti 3M? (anno 2020)"), parent=parent)
    assert data["title"].startswith("Ti ha chiesto l’anno") and "2020" in data["agent"]


def test_non_risponde_a_memoria():
    data = summarize_run(_run([
        {"type": "plan", "planner_skipped": True}, SEARCH1,
        {"type": "grading", "iteration": 1, "sufficient": False, "elements": [{"element": "ricetta", "found": False}]},
        {"type": "answer", "status": "generated", "attempts": 1}], question="Come si prepara la pizza?"))
    assert data["title"] == "Non ha risposto a memoria"


def test_html_senza_ricerche_non_mostra_nulla():
    assert agent_value_html({"events": []}) is None
    html, height = agent_value_html(_run([SEARCH1]), dark=True)
    assert "Riproduci" in html and height > 100
