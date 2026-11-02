"""Controlli di forma della risposta; non certificano la correttezza dei contenuti."""
from __future__ import annotations

import re
import html
from copy import deepcopy

from .generation import generate_answer

_CITATIONS = re.compile(r"\[#(\d+)\](?!\s*\()|\[[^\]]+\]\(\s*#(\d+)\s*\)")
_OPEN_ROLE = re.compile(r"^\s*(?:\\?<|&lt;)(?:assistente|assistant|risposta|answer)(?:\\?>|&gt;)\s*", re.I)
_CLOSE_ROLE = re.compile(r"\s*(?:\\?<|&lt;)/(?:assistente|assistant|risposta|answer)(?:\\?>|&gt;)\s*$", re.I)
_INSUFFICIENT = re.compile(
    r"non (?:ho|abbiamo) informazioni sufficienti|"
    r"(?:nei passaggi forniti\s+)?non trovo informazioni|"
    r"(?:passaggi|testi|documenti) (?:forniti )?non (?:contengono|forniscono)|"
    r"(?:insufficient information|not enough information|cannot find information)",
    re.I,
)


_BARE_REF = re.compile(r"\[(\d{1,2})\](?!\s*\()")
_MULTI_REF = re.compile(r"\[#?\d{1,2}(?:\s*[,;]\s*#?\d{1,2})+\](?!\s*\()")


def clean_answer(text: str) -> str:
    """Rimuove involucri di ruolo ai bordi e uniforma [2] in [#2], preservando il testo."""
    # [#2, #5] o [2; 5] -> [#2] [#5]
    cleaned = _MULTI_REF.sub(lambda m: " ".join(f"[#{n}]" for n in re.findall(r"\d+", m.group(0))), (text or "").strip())
    cleaned = _BARE_REF.sub(r"[#\1]", cleaned)
    for _ in range(3):
        previous = cleaned
        cleaned = _CLOSE_ROLE.sub("", _OPEN_ROLE.sub("", cleaned)).strip()
        if cleaned == previous:
            break
    return cleaned


_WEB_ADDRESS = re.compile(r"(?:[a-z][a-z0-9+.-]*://|www\.|mailto:|href\s*=|src\s*=)", re.I)
_LINK = re.compile(r"\[[^\]]*\]\(\s*([^)]*?)\s*\)")
_INTERNAL_REFS = re.compile(r"#?\d+(?:\s*[,;]\s*#?\d+)*")


def inspect_answer(text: str, passage_count: int, *, truncated: bool = False) -> dict:
    """Controllo di forma della risposta.

    - Un indirizzo web (http, www, mailto, href) in qualsiasi punto della
      risposta è sempre un riferimento esterno: la risposta va bloccata.
    - Una citazione interna scritta in un formato diverso da [..](#N), per
      esempio [..](#2, #5), NON è un collegamento esterno: i numeri vengono
      riconosciuti e il formato viene segnalato in format_issues.
    """
    text = clean_answer(text)
    decoded = html.unescape(text)
    numbers = {int(a or b) for a, b in _CITATIONS.findall(decoded)}
    format_issues: list[str] = []
    web_in_link = False
    for destination in _LINK.findall(decoded):
        if re.fullmatch(r"#\d+", destination):
            continue
        if _WEB_ADDRESS.search(destination):
            web_in_link = True
        elif destination and _INTERNAL_REFS.fullmatch(destination):
            numbers.update(int(n) for n in re.findall(r"\d+", destination))
            format_issues.append(destination)
        else:
            format_issues.append(destination)
    forbidden_links = web_in_link or bool(_WEB_ADDRESS.search(decoded))
    numbers_sorted = sorted(numbers)
    valid = [n for n in numbers_sorted if 1 <= n <= passage_count]
    invalid = [n for n in numbers_sorted if n not in valid]
    status = "external_links" if forbidden_links else "truncated" if truncated and text else "cited" if valid and not invalid else "invalid_citations" if invalid else "insufficient" if _INSUFFICIENT.search(text) else "uncited" if text else "empty"
    return {"status": status, "valid_citations": valid, "invalid_citations": invalid,
            "forbidden_links": forbidden_links, "format_issues": format_issues[:5], "truncated": truncated}


def explain_answer_check(status: str | None, check: dict | None = None) -> str:
    """Explain the recorded formal check without asserting factual correctness."""
    messages = {
        "external_links": ("La risposta contiene un indirizzo web (http, www o mailto). Il modello non naviga su internet: "
                           "l'indirizzo lo ha scritto lui, ricordandolo o inventandolo, quindi la risposta non viene solo dai documenti."),
        "invalid_citations": "La risposta indica numeri di passaggi che non esistono tra quelli forniti al modello.",
        "uncited": "La risposta non contiene riferimenti riconoscibili ai passaggi forniti al modello.",
        "empty": "Il modello ha restituito una risposta vuota.",
        "cited": "La risposta contiene riferimenti nel formato riconosciuto e nei limiti dei passaggi forniti.",
        "insufficient": "Il modello dichiara che le informazioni fornite non sono sufficienti.",
        "truncated": "La risposta è stata interrotta dal limite di lunghezza (LLM_MAX_TOKENS) e non è completa.",
        "off_topic": "Il passaggio più pertinente è sotto la soglia di pertinenza: la domanda sembra fuori tema rispetto ai documenti.",
    }
    message = messages.get(status, "Il motivo dettagliato non è disponibile per questa esecuzione.")
    invalid = (check or {}).get("invalid_citations") or []
    if status == "invalid_citations" and invalid:
        message += " Riferimenti fuori intervallo: " + ", ".join(f"#{n}" for n in invalid) + "."
    return message


def answer_diagnostics(reviewed: dict) -> dict:
    """Persist bounded raw responses and checks, never provider requests or credentials."""
    attempts = []
    for item in reviewed.get("attempts", []):
        response = item.get("response") or {}
        text = str(response.get("answer") or "")
        attempts.append({"check": item.get("check") or {}, "provider": response.get("provider"),
                         "model": response.get("model"), "response": text[:12000],
                         "response_truncated": len(text) > 12000})
    return {"block_reason": reviewed.get("block_reason"), "attempts": attempts,
            "retry_error_type": reviewed.get("review_error_type")}


def generation_request(query: str, *, retry: bool = False) -> str:
    """Mantiene la domanda originale; al retry richiede un controllo mirato."""
    if not retry:
        return query
    return (
        f"{query}\n\n"
        "Verifica aggiuntiva prima della risposta: controlla tutti i passaggi e, per i dati "
        "tabellari, associa entità, anno, riga, colonna e unità. Distingui un dato assente "
        "da una fonte specifica assente. Rispondi solo per ciò che è supportato; "
        "se manca evidenza o l'allineamento è ambiguo, dichiaralo. "
        "Cita ogni dato con il numero del passaggio, per esempio [#2]: niente parentesi tonde, link o indirizzi web."
    )


def generate_single_answer(query: str, passages: list[dict]) -> dict:
    """RAG classica per la pagina Query: una sola chiamata al modello.

    Nessun secondo tentativo e nessun blocco: il controllo di forma viene
    registrato e mostrato come avviso accanto alla risposta. Restituisce la
    stessa struttura di generate_reviewed_answer, così il resto della pagina
    (trust panel, prompt inviato) funziona senza modifiche.
    """
    generated = generate_answer(query, passages)
    answer = clean_answer(generated.get("answer") or "")
    check = inspect_answer(answer, len(passages), truncated=bool(generated.get("truncated")))
    return {"answer": {**generated, "answer": answer},
            "attempts": [{"request": query, "response": deepcopy(generated), "check": check}],
            "check": check, "review_error": None, "review_error_type": None,
            "blocked": False, "block_reason": None}


_PUBLISHABLE_ORDER = {"cited": 0, "insufficient": 1, "invalid_citations": 2, "uncited": 3}


def best_publishable_attempt(reviewed: dict) -> tuple[str, str] | None:
    """Il tentativo migliore che si può mostrare con un avviso, come nella pagina Query.

    Mai uno con indirizzi web o troncato: quelli restano bloccati.
    Restituisce (testo, stato del controllo) oppure None.
    """
    candidates = []
    for attempt in reviewed.get("attempts", []):
        status = (attempt.get("check") or {}).get("status")
        text = clean_answer((attempt.get("response") or {}).get("answer") or "")
        if status in _PUBLISHABLE_ORDER and text:
            candidates.append((_PUBLISHABLE_ORDER[status], text, status))
    if not candidates:
        return None
    _, text, status = min(candidates, key=lambda item: item[0])
    return text, status


def generate_reviewed_answer(query: str, passages: list[dict]) -> dict:
    """Generazione della risposta: al massimo due tentativi.

    Nessun estratto viene sostituito alla risposta e nessuna citazione viene
    assegnata dall'app. Gli output originali restano disponibili in attempts;
    se il secondo tentativo fallisce, viene valutato il primo.
    """
    result = {"answer": {}, "attempts": [], "review_error": None, "review_error_type": None}
    for attempt in range(2):
        request = generation_request(query, retry=attempt > 0)
        try:
            generated = generate_answer(request, passages)
        except Exception as exc:
            if not result["attempts"]:
                raise
            result["review_error"] = str(exc)
            result["review_error_type"] = type(exc).__name__
            break
        quality = inspect_answer(generated.get("answer") or "", len(passages),
                                 truncated=bool(generated.get("truncated")))
        result["attempts"].append({
            "request": request, "response": deepcopy(generated), "check": quality,
        })
        result["answer"] = {**generated, "answer": clean_answer(generated.get("answer") or "")}
        if quality["status"] == "cited":
            break
    final_check = inspect_answer(result["answer"].get("answer", ""), len(passages),
                                 truncated=bool(result["answer"].get("truncated")))
    result["blocked"] = final_check["status"] not in {"cited", "insufficient"}
    result["block_reason"] = final_check["status"] if result["blocked"] else None
    if result["blocked"]:
        # Retain raw attempts for diagnosis only; never expose them as the answer.
        result["answer"] = {**result["answer"], "answer": ""}
    return result
