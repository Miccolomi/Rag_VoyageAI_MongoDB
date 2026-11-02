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
