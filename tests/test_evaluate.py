import json

from scripts import evaluate_retrieval as ev


def test_domande_filtrate_sul_corpus_e_pagine_in_base_1(tmp_path):
    path = tmp_path / "q.jsonl"
    rows = [
        {"financebench_id": "a", "doc_name": "3M_2018_10K", "question": "capex?",
         "evidence": [{"doc_name": "3M_2018_10K", "evidence_page_num": 59}]},
        {"financebench_id": "b", "doc_name": "ASSENTE_2020_10K", "question": "x?", "evidence": []},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    questions = ev.load_questions(path, {"3M_2018_10K.pdf"}, page_base=0)
    assert [q["id"] for q in questions] == ["a"]
    assert questions[0]["pages"] == {60}


def test_rango_documento_e_pagina():
    results = [{"filename": "x.pdf", "page_number": 1}, {"filename": "a.pdf", "page_number": 3},
               {"filename": "a.pdf", "page_number": 60}]
    assert ev.first_rank(results, "a.pdf", None) == 2
    assert ev.first_rank(results, "a.pdf", {60}) == 3
    assert ev.first_rank(results, "b.pdf", None) is None


def test_sintesi_markdown():
    rows = [
        {"config": "hybrid", "doc_rank": 1, "page_rank": 2, "has_pages": True, "latency_ms": 100.0, "error": ""},
        {"config": "hybrid", "doc_rank": None, "page_rank": None, "has_pages": True, "latency_ms": 300.0, "error": ""},
    ]
    table = ev.summarize(rows, [1, 5])
    assert "| hybrid | 2 | 50% | 50% | 0% | 50% | 0.250 | 200 | 0 |" in table
