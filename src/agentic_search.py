"""LLM decisions for the bounded, observable agentic retrieval demo."""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from typing import Any

import httpx
from anthropic import Anthropic

from . import config

log = logging.getLogger(__name__)


def _model_label() -> str:
    return {
        "anthropic": config.ANTHROPIC_AGENT_MODEL,
        "gemini": config.GEMINI_MODEL,
        "ollama": config.OLLAMA_AGENT_MODEL,
        "gateway": config.AI_GATEWAY_AGENT_MODEL,
    }.get(config.AGENT_LLM_PROVIDER, "unknown")


def _complete(system: str, prompt: str, *, max_tokens: int = 700) -> str:
    """Call the configured provider for a short, constrained agent decision."""
    provider = config.AGENT_LLM_PROVIDER
    if provider == "gateway":
        from . import ai_gateway
        return ai_gateway.respond(system, prompt, model=config.AI_GATEWAY_AGENT_MODEL,
                                  max_tokens=max_tokens, json_mode=True, timeout=90)["answer"]
    if provider == "anthropic":
        if not config.ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY non impostata nel .env")
        response = Anthropic(api_key=config.ANTHROPIC_API_KEY).messages.create(
            model=config.ANTHROPIC_AGENT_MODEL,
            max_tokens=max_tokens,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if getattr(block, "type", None) == "text")

    if provider == "gemini":
        if not config.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY non impostata nel .env")
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{config.GEMINI_MODEL}:generateContent"
        )
        response = httpx.post(
            url,
            headers={"x-goog-api-key": config.GEMINI_API_KEY},
            json={
                "system_instruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0, "maxOutputTokens": max_tokens},
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        return "".join(
            part.get("text", "")
            for candidate in data.get("candidates", [])
            for part in candidate.get("content", {}).get("parts", [])
        )

    if provider == "ollama":
        response = httpx.post(
            f"{config.OLLAMA_BASE_URL}/api/chat",
            json={
                "model": config.OLLAMA_AGENT_MODEL,
                "stream": False,
                # Le decisioni sono sempre JSON: Ollama vincola l'output, niente testo attorno.
                "format": "json",
                "keep_alive": config.OLLAMA_KEEP_ALIVE,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                # num_predict limita la risposta: senza, un modello locale può scrivere a lungo.
                "options": {"temperature": 0, "num_ctx": config.OLLAMA_NUM_CTX, "num_predict": max_tokens},
            },
            timeout=300,
        )
        response.raise_for_status()
        return str((response.json().get("message") or {}).get("content", ""))

    raise ValueError(f"LLM_PROVIDER non valido: {provider!r}")


def _read_json(text: str) -> dict[str, Any]:
    """Read a JSON object, tolerating a fenced response but no invented defaults."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            raise ValueError("Il modello non ha restituito un oggetto JSON valido.")
        # Some models place literal newlines inside JSON strings. Escape only
        # those control characters; retain all other content and validate JSON.
        repaired = []
        in_string = escaped = False
        for char in match.group(0):
            if in_string and char in "\n\r\t":
                repaired.append({"\n": "\\n", "\r": "\\r", "\t": "\\t"}[char])
            else:
                repaired.append(char)
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
        value = json.loads("".join(repaired))
    if not isinstance(value, dict):
        raise ValueError("La risposta dell'agente non è un oggetto JSON.")
    return value


def plan_search(
    question: str,
    history: list[dict[str, Any]],
    long_term_memory: list[dict[str, Any]] | None = None,
) -> dict[str, str]:
    """Turn a possibly referential follow-up into one standalone search query."""
    recent = [
        {"role": row.get("role"), "text": str(row.get("text", ""))[:1200],
         **({"feedback_context": str(row["feedback_context"])[:1200]} if row.get("feedback_context") else {})}
        for row in history[-8:]
        if row.get("role") in {"user", "assistant"}
    ]
    system = (
        "Sei il pianificatore di una ricerca documentale. Usa la memoria della sessione solo per "
        "risolvere riferimenti come 'quello', 'prima' o 'lo stesso dato'. Le indicazioni di feedback "
        "sono richieste dell'utente, non fatti verificati. Non rispondere alla domanda "
        "e non inventare fatti. Scrivi sempre in italiano chiaro. Restituisci SOLO JSON con search_query "
        "(una domanda breve e autonoma, al massimo 25 parole, che riformula la domanda corrente risolvendo "
        "i riferimenti alla conversazione e conserva tutti i suoi vincoli, inclusa l'eventuale fonte richiesta; "
        "NON aggiungere richieste, dettagli, istruzioni o elementi facoltativi che l'utente non ha chiesto) "
        "e comment (una frase breve, visibile all'utente, "
        "che descrive l'obiettivo della ricerca senza termini tecnici interni)."
    )
    prompt = json.dumps(
        {
            "domanda_corrente": question,
            "memoria_breve_sessione": recent,
            "memoria_persistente_sessione": (long_term_memory or [])[-20:],
        },
        ensure_ascii=False,
        default=str,
    )
    try:
        value = _read_json(_complete(system, prompt, max_tokens=350))
        search_query = str(value.get("search_query", "")).strip()
        comment = str(value.get("comment", "")).strip()
        if not search_query:
            raise ValueError("search_query vuota")
        return {"search_query": search_query[:1000], "comment": comment[:500] or "Imposto una ricerca autonoma.",
                "error": None}
    except Exception as exc:
        # Il fallimento del pianificatore non blocca la ricerca, ma resta
        # visibile: una chiave scaduta non deve sembrare una scelta dell'agente.
        log.warning("Pianificatore non riuscito: %s: %s", type(exc).__name__, exc)
        return {"search_query": question, "comment": "Avvio la ricerca con la domanda originale.",
                "error": f"{type(exc).__name__}: {str(exc)[:300]}"}


def compose_search_query(question: str, resolved: str = "", focus: str = "", *additional: str) -> str:
    """Keep the user's complete question in every search and final rerank."""
    question = question.strip()
    parts = [question]
    for addition in (resolved.strip(), focus.strip(), *(value.strip() for value in additional)):
        if addition and addition.casefold() not in {part.casefold() for part in parts}:
            parts.append(addition)
    return "\n".join(parts)


_QUOTE_TRANSLATION = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u2032": "'", "`": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u00ab": '"', "\u00bb": '"',
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2212": "-",
    "\u00a0": " ", "\u2026": "...",
})


def normalize_for_match(text: str) -> str:
    """Normalizza tipografia, accenti e spazi per verificare un estratto.

    I modelli riscrivono spesso apostrofi, virgolette, trattini e accenti:
    il confronto deve ignorare queste differenze, non il contenuto.
    """
    text = unicodedata.normalize("NFKC", text or "").translate(_QUOTE_TRANSLATION)
    text = "".join(char for char in unicodedata.normalize("NFKD", text)
                   if not unicodedata.combining(char))
    text = re.sub(r"\s*([.,;:!?()\[\]'\"%-])\s*", r"\1", text)
    return " ".join(text.casefold().split())


def quote_in_passage(quote: str, passage: str) -> bool:
    """True se l'estratto compare nel passaggio, a meno di tipografia e spazi."""
    normalized_quote = normalize_for_match(quote).strip(" .\"'")
    return len(normalized_quote) >= 8 and normalized_quote in normalize_for_match(passage)


GRADER_PASSAGES = 8
GRADER_PASSAGE_CHARS = 3000


def grader_passage_chars(count: int) -> int:
    """Caratteri per passaggio che stanno nel contesto del modello.

    Con Ollama il contesto è fisso (OLLAMA_NUM_CTX): se il prompt lo supera,
    Ollama taglia l'inizio, cioè le istruzioni. Stima prudente di 3 caratteri
    per token, togliendo istruzioni (~900 token) e risposta (500 token).
    """
    if config.AGENT_LLM_PROVIDER != "ollama" or not count:
        return GRADER_PASSAGE_CHARS
    budget_chars = max(config.OLLAMA_NUM_CTX - 1400, 1000) * 3
    return max(600, min(GRADER_PASSAGE_CHARS, budget_chars // count))


def compact_text(text: str) -> str:
    """Toglie righe vuote e spazi ripetuti: le tabelle estratte dai PDF ne sono piene."""
    lines = (" ".join(line.split()) for line in str(text or "").splitlines())
    return "\n".join(line for line in lines if line)


_NUMBER_IN_QUOTE = re.compile(r"\d[\d.,]*\d|\d")


def evidence_in_passage(quote: str, passage: str) -> bool:
    """Estratto letterale, oppure tutti i suoi numeri presenti nel passaggio.

    Nelle tabelle estratte dai PDF il modello spesso ricompone una riga
    ("Total Company ... 1,577") che non compare identica: in quel caso basta
    che ogni numero citato (almeno uno) sia nel passaggio.
    """
    if quote_in_passage(quote, passage):
        return True
    numbers = _NUMBER_IN_QUOTE.findall(quote or "")
    if not numbers:
        # Valore testuale breve (un nome, una sigla): basta che compaia nel passaggio.
        needle = normalize_for_match(quote).strip(" .\"'")
        return len(needle) >= 3 and needle in normalize_for_match(passage)
    # Serve almeno un valore che non sia un anno: "2018" da solo non prova nulla.
    if not any(not (n.isdigit() and 1900 <= int(n) <= 2100) for n in numbers):
        return False
    text = " ".join(str(passage or "").split())
    return all(re.search(rf"(?<![\d.,]){re.escape(n)}(?![\d]|[.,]\d)", text) for n in numbers)


_YEAR = re.compile(r"(?<!\d)(19[5-9]\d|20[0-4]\d)(?!\d)")


def document_year(doc: dict[str, Any]) -> int | None:
    """Anno del documento: metadato `year`, altrimenti l'anno nel nome del file (3M_2018_10K.pdf)."""
    year = doc.get("year")
    if isinstance(year, (int, float)) and 1950 <= int(year) <= 2049:
        return int(year)
    match = _YEAR.search(str(doc.get("filename") or ""))
    return int(match.group(1)) if match else None


def mentions_year(*texts: str) -> bool:
    return any(_YEAR.search(text or "") for text in texts)


def _value_key(value: str) -> tuple[str, ...]:
    numbers = _NUMBER_IN_QUOTE.findall(value or "")
    if numbers:
        return tuple(re.sub(r"[.,]", "", n) for n in numbers)
    return (normalize_for_match(value),)


def find_year_ambiguity(grade: dict[str, Any], *question_texts: str) -> dict[str, Any] | None:
    """Valori diversi per lo stesso dato in documenti di anni diversi, con una domanda senza anno.

    Lo decide il codice, non il modello: il valutatore indica i valori per passaggio,
    ogni valore è stato ritrovato nel suo passaggio, l'anno viene dal documento.
    Lo stesso valore ripetuto in più documenti (dato comparativo) non è un'ambiguità.
    """
    if mentions_year(*question_texts) or not grade.get("evaluated", True):
        return None
    ambiguous: list[str] = []
    by_year: dict[int, dict[str, Any]] = {}
    for element in grade.get("elements") or []:
        values = [v for v in element.get("values") or [] if v.get("year")]
        keys_by_year: dict[int, set] = {}
        for item in values:
            keys_by_year.setdefault(item["year"], set()).add(_value_key(item["value"]))
        distinct = {key for keys in keys_by_year.values() for key in keys}
        if len(keys_by_year) < 2 or len(distinct) < 2:
            continue
        ambiguous.append(element["element"])
        for item in values:
            option = by_year.setdefault(item["year"], {"year": item["year"], "filename": item.get("filename"),
                                                       "values": {}})
            option["values"].setdefault(element["element"], item["value"])
    if not ambiguous:
        return None
    options = [by_year[year] for year in sorted(by_year, reverse=True)]
    return {"elements": ambiguous, "options": options}


def assess_evidence(
    question: str,
    search_query: str,
    candidates: list[dict[str, Any]],
    previous_queries: list[str] | None = None,
) -> dict[str, Any]:
    """Valuta, elemento per elemento, se i passaggi bastano. Non risponde mai.

    candidates è l'insieme cumulativo delle ricerche fatte finora, riordinato
    rispetto alla domanda completa: la valutazione riguarda tutte le prove
    raccolte, non solo l'ultima ricerca. Un elemento conta come trovato solo
    se l'estratto indicato dal modello si ritrova nel passaggio.
    """
    evidence = []
    selected = candidates[:GRADER_PASSAGES]
    per_passage = grader_passage_chars(len(selected))
    for index, doc in enumerate(selected, start=1):
        evidence.append({
            "ref": index,
            "file": doc.get("filename"),
            "anno_documento": document_year(doc),
            "page": doc.get("page_number"),
            "text": compact_text(doc.get("content", ""))[:per_passage],
        })
    system = (
        "Sei il valutatore di copertura di una ricerca RAG su documenti finanziari. Non rispondere alla "
        "domanda. Scomponi la domanda negli elementi informativi richiesti (per esempio: un valore per "
        "ciascuna azienda, anno o metrica) e, per ciascuno, stabilisci se i passaggi lo contengono. "
        "Gli elementi sono SOLO quelli che la domanda chiede esplicitamente: non aggiungere dettagli "
        "facoltativi o «eventuali». Se i passaggi rispondono a un livello di dettaglio diverso da quello "
        "immaginato (per esempio quanti stabilimenti in quanti Stati e Paesi invece di un elenco di città), "
        "l'elemento è TROVATO con quel valore: i documenti finanziari raramente scendono più nel dettaglio.\n"
        "Regole di lettura:\n"
        "- Le tabelle sono estratte da PDF e appiattite: le intestazioni compaiono prima, in ordine, e i "
        "valori di ogni riga seguono lo stesso ordine delle colonne. Ricostruisci le colonne dall'ordine "
        "delle intestazioni e degli anni prima di dire che un valore manca.\n"
        "- Considera i sinonimi finanziari: capex = capital spending = capital expenditures = purchases "
        "of property, plant and equipment (PP&E); dipendenti = employees; ricavi = net sales = revenue.\n"
        "- Un documento di un anno successivo può riportare il dato di un anno precedente nella colonna "
        "corrispondente: è un'evidenza valida.\n"
        "Scrivi tutto in italiano. Restituisci SOLO JSON con:\n"
        "elements: lista di oggetti {element: descrizione breve in italiano leggibile (es. \"capex 3M 2018\"), "
        "found: boolean, ref: numero del passaggio o null, value: il valore esatto come scritto nel "
        "passaggio, con unità se presente (es. \"$ 1,577\" oppure \"93,516\"), o null, "
        "other_values: lista di {ref, value} con lo stesso dato riportato in ALTRI passaggi, anche se "
        "uguale o di un anno diverso (massimo 4; lista vuota se non ce ne sono)};\n"
        "comment: una frase in italiano che dice cosa è stato trovato e cosa manca;\n"
        "followup_query: se manca almeno un elemento, una ricerca breve SOLO per gli elementi mancanti, "
        "autonoma (include azienda, anno e metrica, preferibilmente nei termini usati nei documenti, "
        "anche in inglese), diversa dalle ricerche già eseguite; altrimenti null."
    )
    prompt = json.dumps({
        "domanda": question,
        "ricerche_gia_eseguite": list(previous_queries or [search_query]),
        "passaggi_recuperati": evidence,
    }, ensure_ascii=False)
    try:
        response_text = _complete(system, prompt, max_tokens=700)
    except Exception as exc:
        failure_detail = type(exc).__name__
        if isinstance(exc, httpx.TimeoutException):
            failure_reason = "timeout"
        elif isinstance(exc, httpx.ConnectError):
            failure_reason = "connection"
        elif isinstance(exc, httpx.HTTPStatusError):
            failure_reason = "service_error"
            failure_detail = f"HTTP {exc.response.status_code}"
        else:
            failure_reason = "request_error"
        return {"evaluated": False, "sufficient": False, "comment": "",
                "followup_query": None, "failure_reason": failure_reason,
                "failure_detail": failure_detail}

    try:
        value = _read_json(response_text)
        raw_elements = value.get("elements")
        if not isinstance(raw_elements, list) or not raw_elements:
            raise ValueError("elements mancante")
        elements = []
        for item in raw_elements[:8]:
            if not isinstance(item, dict) or not str(item.get("element") or "").strip():
                continue
            ref = item.get("ref")
            raw_value = item.get("value", item.get("quote"))
            claimed = item.get("found") is True
            valid_ref = type(ref) is int and 1 <= ref <= len(evidence)
            value_text = " ".join(str(raw_value).split())[:200] if isinstance(raw_value, (str, int, float)) else ""
            # Il valore deve comparire nel passaggio indicato: numeri tutti presenti, oppure testo letterale.
            verified = bool(claimed and valid_ref and value_text and evidence_in_passage(value_text, evidence[ref - 1]["text"]))
            # Tutti i valori dichiarati per questo dato, ciascuno ritrovato nel proprio passaggio.
            found_values = []
            pairs = [(ref, value_text)] if verified else []
            for extra in (item.get("other_values") or [])[:4] if isinstance(item.get("other_values"), list) else []:
                if isinstance(extra, dict):
                    pairs.append((extra.get("ref"), " ".join(str(extra.get("value") or "").split())[:200]))
            for extra_ref, extra_value in pairs:
                if (type(extra_ref) is int and 1 <= extra_ref <= len(evidence) and extra_value
                        and evidence_in_passage(extra_value, evidence[extra_ref - 1]["text"])):
                    found_values.append({"ref": extra_ref, "value": extra_value,
                                         "year": evidence[extra_ref - 1]["anno_documento"],
                                         "filename": evidence[extra_ref - 1]["file"]})
            elements.append({
                "element": str(item["element"]).strip().replace("_", " ")[:200],
                "found": verified,
                "claimed_found": claimed,
                "ref": ref if valid_ref else None,
                "value": value_text if verified else "",
                "quote": value_text if verified else "",
                "values": found_values,
            })
        if not elements:
            raise ValueError("nessun elemento valido")
        sufficient = all(element["found"] for element in elements)
        followup = value.get("followup_query")
        return {
            "evaluated": True,
            "sufficient": sufficient,
            "comment": str(value.get("comment") or "").strip()[:600],
            "elements": elements,
            "missing_elements": [e["element"] for e in elements if not e["found"]],
            "unverified_elements": [e["element"] for e in elements if e["claimed_found"] and not e["found"]],
            "supporting_refs": list(dict.fromkeys(e["ref"] for e in elements if e["found"])),
            "supporting_quotes": [{"ref": e["ref"], "quote": e["quote"]} for e in elements if e["found"]],
            "followup_query": str(followup).strip()[:1000] if followup and not sufficient else None,
        }
    except (ValueError, TypeError):
        return {
            "evaluated": False,
            "sufficient": False,
            "comment": "",
            "followup_query": None,
            "failure_reason": "invalid_response",
        }


_QUERY_FILLER_WORDS = {
    "a", "al", "alla", "alle", "ai", "agli", "all", "che", "con", "d", "da", "dal", "dalla", "dalle",
    "dei", "degli", "del", "della", "delle", "di", "e", "ed", "i", "il", "in", "l", "la", "le",
    "nei", "nel", "nella", "nelle", "o", "per", "quale", "quali", "quanto", "riporta",
    "riportare", "sono", "stati", "state", "stato", "stata", "un", "una", "uno", "è", "erano", "entrambi",
    "an", "and", "are", "as", "at", "by", "for", "from", "is", "of", "on", "or", "the", "to",
    "what", "were", "which", "with",
}


def _query_signature(query: str) -> tuple[str, ...]:
    """Normalize formatting and function words while keeping order and constraints."""
    normalized = unicodedata.normalize("NFKC", query).casefold()
    tokens = re.findall(r"[^\W_]+|[<>]=?|[=+−–-]|[%$€£]", normalized, re.UNICODE)
    return tuple(token for token in tokens if token not in _QUERY_FILLER_WORDS)


def decide_next_search(grade: dict[str, Any], *, iteration: int, retries: int,
                       seen_queries: set[str]) -> dict[str, str | None]:
    """Choose the next action; iteration is zero-based and retries excludes the first search."""
    if not grade.get("evaluated", True):
        return {"action": "stop", "reason": "evaluation_unavailable", "query": None}
    if grade.get("sufficient"):
        return {"action": "stop", "reason": "evidence_sufficient", "query": None}
    if iteration >= retries:
        return {"action": "stop", "reason": "limit_reached", "query": None}
    query = str(grade.get("followup_query") or "").strip()
    if not query:
        return {"action": "stop", "reason": "no_followup_query", "query": None}
    if query.casefold() in seen_queries:
        return {"action": "stop", "reason": "duplicate_query", "query": query}
    signature = _query_signature(query)
    if not signature or any(signature == _query_signature(previous) for previous in seen_queries):
        return {"action": "stop", "reason": "rephrased_duplicate", "query": query}
    return {"action": "continue", "reason": "evidence_incomplete", "query": query}


def model_label() -> str:
    return _model_label()


def retrieve_candidates(
    collection,
    query: str,
    *,
    filters: dict[str, Any] | None,
    modes: list[str],
    top_k: int,
    instructions: str = "",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run the app's hybrid Atlas retrieval and rerank path for one agent query."""
    from .rerank import rerank
    from .retrieval import build_native_rerank_pipeline, build_pipeline
    from .search_scope import prepare_vectors

    query_vectors = prepare_vectors(query, modes)
    if config.RERANK_BACKEND == "native":
        pipeline = build_native_rerank_pipeline(
            query,
            filters=filters,
            top_k=top_k,
            top_n=top_k,
            embed_mode="all",
            modes=modes,
            query_vectors=query_vectors,
            instructions=instructions or None,
            include_candidates=True,
        )
        ranked = list(collection.aggregate(pipeline))
        candidates = sorted(ranked, key=lambda row: row.get("_hybrid_rank", 10**9))
        return candidates, ranked

    pipeline = build_pipeline(
        query,
        filters=filters,
        top_k=top_k,
        embed_mode="all",
        modes=modes,
        query_vectors=query_vectors,
    )
    candidates = list(collection.aggregate(pipeline))
    if not config.VOYAGE_API_KEY:
        raise RuntimeError("Il reranking tramite API richiede VOYAGE_API_KEY.")
    ranked = rerank(query, candidates, top_n=min(top_k, len(candidates)), instructions=instructions or None)
    return candidates, ranked


def rerank_union(
    collection,
    query: str,
    candidates: list[dict[str, Any]],
    *,
    top_n: int,
    instructions: str = "",
) -> list[dict[str, Any]]:
    """Rerank distinct candidates collected across turns against the original question."""
    if not candidates:
        return []
    from .rerank import format_rerank_query, rerank

    if config.RERANK_BACKEND == "voyage_api" and config.VOYAGE_API_KEY:
        return rerank(query, candidates, top_n=min(top_n, len(candidates)), instructions=instructions or None)
    if config.RERANK_BACKEND == "voyage_api":
        raise RuntimeError("Il riordino finale richiede VOYAGE_API_KEY con il backend API.")

    if any(row.get("_id") is None for row in candidates):
        raise ValueError("Impossibile riordinare i passaggi: manca un identificativo MongoDB.")

    rerank_text = {"$concat": [
        "Titolo: ", {"$ifNull": ["$title", ""]},
        "\nFonte: ", {"$ifNull": ["$filename", ""]},
        ", pagina ", {"$convert": {"input": "$page_number", "to": "string", "onNull": "n/d", "onError": "n/d"}},
        "\nTipo contenuto: ", {"$ifNull": ["$content_type", ""]},
        "\nCategoria: ", {"$ifNull": ["$category", ""]},
        "\nContenuto:\n", {"$ifNull": ["$content", ""]},
    ]}
    pipeline = [
        {"$match": {"_id": {"$in": [row["_id"] for row in candidates if row.get("_id") is not None]}}},
        {"$set": {"_rerank_text": rerank_text}},
        {"$rerank": {
            "query": {"text": format_rerank_query(query, instructions or None)},
            "model": config.VOYAGE_RERANK_MODEL,
            "path": "_rerank_text",
            "numDocsToRerank": len(candidates),
        }},
        {"$set": {"_rerank_score": {"$meta": "score"}}},
        {"$unset": "_rerank_text"},
        {"$limit": min(top_n, len(candidates))},
        {"$project": {"document_id": 1, "filename": 1, "page_number": 1,
                       "chunk_index": 1, "content_type": 1, "content": 1,
                       "title": 1, "category": 1, "gridfs_file_id": 1,
                       "embedding_mode": 1, "_rerank_score": 1}},
    ]
    ranked = list(collection.aggregate(pipeline))
    if len(ranked) < min(top_n, len(candidates)):
        raise RuntimeError("Il riordino finale non ha restituito tutti i passaggi richiesti; riprova la ricerca.")
    return ranked
