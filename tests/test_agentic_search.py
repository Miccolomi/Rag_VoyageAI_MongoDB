from src import agentic_search
from src.agentic_search import assess_evidence, decide_next_search, plan_search, quote_in_passage


def test_estratto_con_apostrofi_tipografici_e_spazi():
    passage = "Purchases of property, plant and equipment (PP&E)   were $1,577 million in 2018."
    quote = "purchases of property, plant and equipment (PP&E) were $1,577 million"
    assert quote_in_passage(quote, passage)


def test_estratto_con_accenti_e_virgolette_diverse():
    passage = "L'utile netto è aumentato del 12% rispetto all'anno precedente."
    assert quote_in_passage("L’utile netto e aumentato del 12 %", passage)


def test_estratto_inventato_non_passa():
    assert not quote_in_passage("ricavi pari a 99 miliardi", "I ricavi AWS sono stati 80.096 milioni.")


def test_valutatore_sufficiente_con_estratto_riformulato_tipograficamente(monkeypatch):
    passage = "Research and development – $2,987 million in fiscal 2022."
    monkeypatch.setattr(agentic_search, "_complete", lambda *a, **k: (
        '{"elements": [{"element": "R&D Adobe 2022", "found": true, "ref": 1, '
        '"quote": "Research and development - $2,987 million"}], "comment": "ok", "followup_query": null}'))
    grade = assess_evidence("R&D Adobe 2022?", "R&D Adobe 2022", [{"content": passage, "filename": "a.pdf"}])
    assert grade["evaluated"] and grade["sufficient"]
    assert grade["supporting_quotes"][0]["ref"] == 1


TABELLA_3M = """Geographic Area Supplemental Information


 Property, Plant and

Equipment - net
 Employees as of December 31,
Capital Spending
 as of December 31,
(Millions, except Employees)
2018

2017

2016

2018
United States
 37,412 36,958 35,748 $
994 $
852 $
834 $ 4,915 $ 4,891
Total Company
 93,516 91,536 91,584 $ 1,577 $1,373 $1,420 $ 8,738 $ 8,866
"""


def test_tabella_appiattita_riga_ricomposta_vale_come_prova(monkeypatch):
    seen = {}

    def fake(system, prompt, **kw):
        seen["prompt"] = prompt
        seen["system"] = system
        return ('{"elements": ['
                '{"element": "capex 3M 2018", "found": true, "ref": 1, "quote": "Total Company ... Capital Spending 2018 $ 1,577"},'
                '{"element": "dipendenti fine 2018", "found": true, "ref": 1, "quote": "Total Company 93,516"}],'
                '"comment": "Entrambi nella tabella del passaggio 1", "followup_query": null}')
    monkeypatch.setattr(agentic_search, "_complete", fake)
    grade = assess_evidence("capex di 3M nel 2018 e dipendenti a fine 2018?", "q",
                            [{"content": TABELLA_3M, "filename": "3M_2018_10K.pdf"}])
    assert grade["sufficient"]
    assert [e["found"] for e in grade["elements"]] == [True, True]
    assert "capital spending" in seen["system"]          # sinonimi nel prompt
    assert "\n\n" not in seen["prompt"]                   # righe vuote tolte prima dell'invio


def test_valore_inventato_non_conta_come_trovato(monkeypatch):
    monkeypatch.setattr(agentic_search, "_complete", lambda *a, **k: (
        '{"elements": [{"element": "capex 2018", "found": true, "ref": 1, "quote": "Capital Spending 2018 $ 1,600"}],'
        '"comment": "x", "followup_query": "3M capital spending 2018"}'))
    grade = assess_evidence("capex 3M 2018?", "q", [{"content": TABELLA_3M}])
    assert not grade["sufficient"]
    assert grade["unverified_elements"] == ["capex 2018"]
    assert grade["followup_query"] == "3M capital spending 2018"


def test_solo_un_anno_non_prova_nulla():
    assert not agentic_search.evidence_in_passage("Capital spending 2018", TABELLA_3M)


def test_errore_del_pianificatore_resta_visibile(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("401 chiave non valida")
    monkeypatch.setattr(agentic_search, "_complete", boom)
    plan = plan_search("E nel 2017?", [])
    assert plan["search_query"] == "E nel 2017?"
    assert "401" in plan["error"]


def test_decisione_continua_solo_con_query_nuova():
    grade = {"evaluated": True, "sufficient": False, "followup_query": "capex 3M cash flow 2017"}
    decision = decide_next_search(grade, iteration=0, retries=1, seen_queries={"spese 3m 2017"})
    assert decision["action"] == "continue"
    again = decide_next_search(grade, iteration=0, retries=1, seen_queries={"capex 3m cash flow 2017"})
    assert again["reason"] == "duplicate_query"


def test_ollama_decisioni_json_limitate_e_nel_contesto(monkeypatch):
    from src import config
    monkeypatch.setattr(config, "AGENT_LLM_PROVIDER", "ollama")
    monkeypatch.setattr(config, "OLLAMA_NUM_CTX", 8192)
    sent = {}

    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": '{"elements": [{"element": "x", "found": false, "ref": null, "value": null}], '
                                           '"comment": "manca x", "followup_query": "x 2018"}'}}

    def fake_post(url, json=None, timeout=None):
        sent.update(json)
        return Resp()
    monkeypatch.setattr(agentic_search.httpx, "post", fake_post)
    docs = [{"content": "a" * 10000} for _ in range(8)]
    grade = assess_evidence("domanda", "q", docs)
    assert grade["evaluated"] and not grade["sufficient"]
    assert sent["format"] == "json"
    assert sent["options"]["num_predict"] == 700
    assert sent["options"]["num_ctx"] == 8192
    # 8 passaggi devono stare nel contesto: istruzioni + passaggi sotto ~3 caratteri per token.
    assert len(sent["messages"][1]["content"]) < (8192 - 500) * 3
    assert agentic_search.grader_passage_chars(8) < 3000


PASSAGGI_ANNI = [
    {"filename": "3M_2018_10K.pdf", "content": "In the U.S., 3M operates 61 manufacturing facilities in 29 states."},
    {"filename": "3M_2020_10K.pdf", "content": "3M operates 64 manufacturing facilities in 29 states."},
    {"filename": "3M_2019_10K.pdf", "content": "Capital spending $ 1,577 in 2018"},
    {"filename": "3M_2018_10K.pdf", "content": "Capital spending $ 1,577"},
]


def _grade_con(monkeypatch, elements_json):
    monkeypatch.setattr(agentic_search, "_complete", lambda *a, **k: elements_json)
    return assess_evidence("q", "q", PASSAGGI_ANNI)


def test_valori_diversi_in_anni_diversi_chiede_l_anno(monkeypatch):
    grade = _grade_con(monkeypatch, '{"elements": [{"element": "stabilimenti USA", "found": true, "ref": 1, '
                       '"value": "61 manufacturing facilities in 29 states", "other_values": [{"ref": 2, "value": "64 manufacturing facilities in 29 states"}]}], '
                       '"comment": "", "followup_query": null}')
    amb = agentic_search.find_year_ambiguity(grade, "Dove si trovano gli stabilimenti produttivi di 3M?")
    assert amb and [o["year"] for o in amb["options"]] == [2020, 2018]
    assert amb["options"][0]["filename"] == "3M_2020_10K.pdf"
    # Con l'anno nella domanda non chiede nulla.
    assert agentic_search.find_year_ambiguity(grade, "Quanti stabilimenti produttivi aveva 3M nel 2018?") is None


def test_stesso_valore_in_documenti_diversi_non_e_ambiguo(monkeypatch):
    grade = _grade_con(monkeypatch, '{"elements": [{"element": "capex", "found": true, "ref": 4, "value": "$ 1,577", '
                       '"other_values": [{"ref": 3, "value": "$ 1,577"}]}], "comment": "", "followup_query": null}')
    assert agentic_search.find_year_ambiguity(grade, "Qual è il capex di 3M?") is None


def test_valore_inventato_non_crea_ambiguita(monkeypatch):
    grade = _grade_con(monkeypatch, '{"elements": [{"element": "stabilimenti", "found": true, "ref": 1, '
                       '"value": "61 manufacturing facilities", "other_values": [{"ref": 2, "value": "70 manufacturing facilities"}]}], '
                       '"comment": "", "followup_query": null}')
    assert agentic_search.find_year_ambiguity(grade, "Dove sono gli stabilimenti di 3M?") is None
