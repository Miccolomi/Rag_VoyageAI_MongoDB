"""Verifica deterministica dei numeri della risposta nei passaggi citati.

Per ogni numero della risposta cerca lo stesso valore nel testo del passaggio
citato nella stessa frase (o, se la frase non cita nulla, nello stesso
paragrafo). Nessuna chiamata a modelli: solo confronto di testo e di valori.

Gestisce le differenze di formato più comuni tra risposta in italiano e
documenti in inglese: separatori ("1.577" / "1,577"), unità ("1,6 miliardi" /
"$1.6 billion", oppure tabelle "in millions") e arrotondamenti
("1,6 miliardi" contro "1,577" milioni → trovato, arrotondato).
"""
from __future__ import annotations

import math
import re
from typing import Any

# Citazione markdown [file (p. N)](#N) o (#1, #2), oppure marker compatto [#N].
_CITE_LINK = re.compile(r"\[[^\]]*\]\(\s*(#?\d+(?:\s*[,;]\s*#?\d+)*)\s*\)")
_CITE_MARK = re.compile(r"\[#(\d+)\](?!\s*\()")

# Un numero non attaccato a lettere: esclude 3M, Q1, FY2018, 10-K.
_NUMBER = re.compile(r"(?<![A-Za-z\d.,])(\d+(?:[.,]\d+)*)(?![A-Za-z\d]|-[A-Za-z])")
_UNIT_AFTER = re.compile(
    r"\s*(%|mila\b|thousands?\b|milion[ie]\b|mln\b|millions?\b|mn\b|miliard[io]\b|mld\b|billions?\b|bn\b)",
    re.I,
)
_MONTHS = (r"(?:gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre|"
           r"january|february|march|april|may|june|july|august|september|october|november|december)")
_MONTH_AFTER = re.compile(r"\s+" + _MONTHS + r"\b", re.I)
_MONTH_BEFORE = re.compile(r"\b" + _MONTHS + r"\s+$", re.I)
_PAGE_REF = re.compile(r"\(?\b(?:p|pp|pag|pagina|pagine|page)\.?\s*\d+(?:\s*[-–,]\s*\d+)*\)?", re.I)
_LIST_MARKER = re.compile(r"^\s*\d+[.)]\s+", re.M)
_SENTENCE_END = re.compile(r"(?<=[.!?;])\s+(?=[A-ZÀ-Ý\"“(*\-•])")

_SCALE = {"mila": 1e3, "thousand": 1e3, "thousands": 1e3,
          "milioni": 1e6, "milione": 1e6, "mln": 1e6, "million": 1e6, "millions": 1e6, "mn": 1e6,
          "miliardi": 1e9, "miliardo": 1e9, "mld": 1e9, "billion": 1e9, "billions": 1e9, "bn": 1e9}


def _interpretations(raw: str, italian: bool = False) -> list[tuple[float, float]]:
    """Valori possibili di una stringa numerica: (valore, ultima cifra significativa).

    italian=True per la risposta: "1.800" è milleottocento, mai 1,8.
    """
    groups = re.split(r"[.,]", raw)
    seps = re.findall(r"[.,]", raw)
    if not seps:
        return [(float(raw), 1.0)]
    out: list[tuple[float, float]] = []
    if len(set(seps)) == 1:
        # Solo punti o solo virgole: migliaia se tutti i gruppi dopo il primo hanno 3 cifre.
        if all(len(g) == 3 for g in groups[1:]) and len(groups[0]) <= 3:
            out.append((float("".join(groups)), 1.0))
        if len(seps) == 1 and not (italian and seps[0] == "." and len(groups[1]) == 3):
            decimals = len(groups[1])
            out.append((float(f"{groups[0]}.{groups[1]}"), 10.0 ** -decimals))
    else:
        # Due separatori diversi: l'ultimo è quello decimale.
        dec_sep = seps[-1]
        idx = raw.rfind(dec_sep)
        integer = re.sub(r"[.,]", "", raw[:idx])
        fraction = raw[idx + 1:]
        if integer.isdigit() and fraction.isdigit():
            out.append((float(f"{integer}.{fraction}"), 10.0 ** -len(fraction)))
    return out


def _numbers(text: str, italian: bool = False) -> list[dict[str, Any]]:
    found = []
    for match in _NUMBER.finditer(text):
        raw = match.group(1)
        unit_match = _UNIT_AFTER.match(text, match.end())
        unit = unit_match.group(1).lower() if unit_match else ""
        # Giorno di una data ("31 dicembre", "December 31"): non è un dato da verificare.
        if raw.isdigit() and 1 <= int(raw) <= 31 and (
                _MONTH_AFTER.match(text, match.end()) or _MONTH_BEFORE.search(text[:match.start()])):
            continue
        found.append({
            "raw": raw,
            "start": match.start(),
            "end": unit_match.end() if unit_match else match.end(),
            "percent": unit == "%",
            "scale": _SCALE.get(unit),
            "values": _interpretations(raw, italian),
        })
    return found


def _is_claim(number: dict[str, Any]) -> bool:
    """Esclude anni e piccoli interi senza unità: non sono dati da verificare."""
    if number["scale"] or number["percent"]:
        return True
    raw = number["raw"]
    if raw.isdigit():
        value = int(raw)
        if 1900 <= value <= 2100 or value < 10:
            return False
    return True


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


def _significant_digits(raw_value: float, place: float) -> int:
    if raw_value == 0:
        return 0
    return max(0, int(math.floor(math.log10(abs(raw_value)) - math.log10(place))) + 1)


def _match(claim: dict[str, Any], candidate: dict[str, Any]) -> str | None:
    """'exact', 'rounded' oppure None."""
    if claim["percent"] != candidate["percent"] and (claim["percent"] or candidate["percent"]):
        # Un 15% non coincide con un 15 qualsiasi; un 15 senza % può però comparire come 15%.
        if claim["percent"]:
            return None
    claim_scale = claim["scale"] or 1.0
    cand_scales = [candidate["scale"]] if candidate["scale"] else [1.0, 1e3, 1e6, 1e9]
    best = None
    for value, place in claim["values"]:
        for cand_value, _ in candidate["values"]:
            if _close(value, cand_value):
                return "exact"
            for scale in (cand_scales if claim["scale"] else []):
                target = cand_value * scale
                claimed = value * claim_scale
                if _close(claimed, target):
                    return "exact"
                # Arrotondamento solo per importi con unità ("1,6 miliardi"):
                # almeno 2 cifre significative e passaggio più preciso.
                step = place * claim_scale
                if claim["scale"] and _significant_digits(value, place) >= 2 and target > step:
                    if _close(round(target / step) * step, claimed):
                        best = "rounded"
            if claim["percent"] and candidate["percent"]:
                # Percentuali: 14,9% contro 14,86%.
                if _significant_digits(value, place) >= 2 and _close(round(cand_value / place) * place, value):
                    best = best or "rounded"
    return best


def _clean(text: str) -> str:
    text = _CITE_LINK.sub(" ", text)
    text = _CITE_MARK.sub(" ", text)
    text = _PAGE_REF.sub(" ", text)
    return _LIST_MARKER.sub("", text)


def _cited(text: str) -> list[int]:
    refs: list[int] = []
    for match in _CITE_LINK.finditer(text):
        refs += [int(n) for n in re.findall(r"\d+", match.group(1))]
    refs += [int(n) for n in _CITE_MARK.findall(text)]
    return list(dict.fromkeys(refs))


def _excerpt(content: str, start: int, end: int, width: int = 70) -> dict[str, str]:
    left = max(0, start - width)
    right = min(len(content), end + width)
    # Taglia su un confine di parola, non a metà.
    if left > 0:
        space = content.find(" ", left, start)
        left = space + 1 if space != -1 else left
    if right < len(content):
        space = content.rfind(" ", end, right)
        right = space if space != -1 else right
    before = re.sub(r"\s+", " ", content[left:start])
    after = re.sub(r"\s+", " ", content[end:right])
    return {
        "before": ("…" if left > 0 else "") + before.lstrip(),
        "match": content[start:end],
        "after": after.rstrip() + ("…" if right < len(content) else ""),
    }


def check_citations(answer: str, passages: list[dict]) -> list[dict[str, Any]]:
    """Una riga per ogni numero della risposta.

    status: "found" (stesso valore), "rounded" (valore arrotondato),
    "not_found" (assente dai passaggi citati), "uncited" (frase senza citazione).
    """
    rows: list[dict[str, Any]] = []
    passage_numbers: dict[int, list[dict[str, Any]]] = {}
    seen: set[tuple[str, tuple[int, ...]]] = set()

    for paragraph in re.split(r"\n\s*\n|\n(?=\s*[-*•]|\s*\d+[.)]\s)", answer or ""):
        paragraph_refs = _cited(paragraph)
        for sentence in _SENTENCE_END.split(paragraph):
            refs = _cited(sentence) or paragraph_refs
            refs = [n for n in refs if 1 <= n <= len(passages)]
            text = _clean(sentence)
            for claim in _numbers(text, italian=True):
                if not _is_claim(claim):
                    continue
                shown = text[claim["start"]:claim["end"]].strip()
                key = (shown, tuple(refs))
                if key in seen:
                    continue
                seen.add(key)
                row = {"claim": shown, "refs": refs, "status": "uncited" if not refs else "not_found",
                       "ref": refs[0] if refs else None, "excerpt": None}
                for ref in refs:
                    content = passages[ref - 1].get("content") or ""
                    if ref not in passage_numbers:
                        passage_numbers[ref] = _numbers(content)
                    hit = None
                    for candidate in passage_numbers[ref]:
                        kind = _match(claim, candidate)
                        if kind == "exact" or (kind == "rounded" and hit is None):
                            hit = (kind, candidate)
                            if kind == "exact":
                                break
                    if hit and (row["status"] != "found"):
                        row.update(status="found" if hit[0] == "exact" else "rounded", ref=ref,
                                   excerpt=_excerpt(content, hit[1]["start"], hit[1]["end"]))
                        if hit[0] == "exact":
                            break
                rows.append(row)
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"found": 0, "rounded": 0, "not_found": 0, "uncited": 0}
    for row in rows:
        counts[row["status"]] += 1
    return counts
