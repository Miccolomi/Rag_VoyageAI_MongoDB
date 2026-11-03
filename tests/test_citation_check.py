from src.citation_check import check_citations

P = [{"content": "Purchases of property, plant and equipment (PP&E) (1,577) (1,373) (1,420) in millions."},
     {"content": "Capital spending totaled $1.6 billion in 2018. Margin 14.86%."}]

def st(answer):
    return [(r["claim"], r["status"]) for r in check_citations(answer, P)]

def test_separatori_diversi():
    assert st("Spesa di 1.577 milioni [a (p. 60)](#1).") == [("1.577 milioni", "found")]

def test_unita_diverse():
    assert st("Circa 1,6 miliardi [a (p. 47)](#2).") == [("1,6 miliardi", "found")]

def test_arrotondato_da_tabella_in_milioni():
    assert st("Circa 1,6 miliardi [a (p. 60)](#1).") == [("1,6 miliardi", "rounded")]

def test_non_trovato():
    assert st("Previsione 1.800 milioni [a (p. 60)](#1).") == [("1.800 milioni", "not_found")]

def test_mille_ottocento_non_e_uno_virgola_otto():
    assert check_citations("Valore 1.800 [a (p. 1)](#1).", [{"content": "growth 1.8 times"}])[0]["status"] == "not_found"

def test_percentuale_arrotondata():
    assert st("Margine del 14,9% [a (p. 47)](#2).") == [("14,9%", "rounded")]

def test_anni_pagine_e_nomi_esclusi():
    assert st("Nel 2018 il 10-K di 3M a p. 60 [a (p. 60)](#1) parla del Q1.") == []

def test_senza_citazione():
    assert st("Un valore di 1.577 milioni.") == [("1.577 milioni", "uncited")]

def test_citazione_nel_paragrafo():
    rows = check_citations("Spesa di 1.577 milioni. Nel 2017 erano 1.373 milioni [a (p. 60)](#1).", P)
    assert [r["status"] for r in rows] == ["found", "found"]


def test_giorno_della_data_non_e_un_dato():
    rows = check_citations("Dipendenti al 31 dicembre 2018: 93.516 [#1]. Al December 31 erano tanti [#1].",
                           [{"content": "At December 31, 2018, the Company employed 93,516 people"}])
    assert [r["claim"] for r in rows] == ["93.516"]


def test_stesso_valore_scritto_due_volte_conta_una_volta():
    from src.citation_check import check_citations, summarize
    passages = [{"content": "Purchases of property, plant and equipment (PP&E) (1,577) (1,373)"}]
    rows = check_citations("Le spese sono state di 1.577 milioni di dollari (1.577 nel rendiconto) [#1].", passages)
    assert summarize(rows)["found"] == 1


def test_valore_calcolato_dichiarato_con_la_formula():
    from src.citation_check import check_citations, summarize
    passages = [{"content": "75 manufacturing facilities in 29 states; 107 manufacturing and converting facilities in 36 countries"}]
    rows = check_citations("3M aveva 182 stabilimenti complessivi, di cui 75 negli Stati Uniti e 107 all’estero [#1].", passages)
    computed = [r for r in rows if r["status"] == "computed"]
    assert computed and computed[0]["claim"] == "182" and computed[0]["formula"] == "75 + 107"
    assert summarize(rows)["not_found"] == 0


def test_numero_assente_senza_operandi_resta_non_trovato():
    from src.citation_check import check_citations
    rows = check_citations("Gli stabilimenti erano 182 [#1].", [{"content": "75 facilities and 107 facilities"}])
    assert rows[0]["status"] == "not_found"
