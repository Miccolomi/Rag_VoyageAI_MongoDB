"""Servizi per l'interfaccia web: Query, viste sulle ricerche agentiche, Upload, Admin.

Il server web (server.py) chiama queste funzioni e le espone come API JSON o come flussi di eventi.
"""
from __future__ import annotations

import re
import tempfile
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator

from bson import ObjectId
from gridfs import GridFS

from . import config, field_encryption, search_scope
from .agent_value import summarize_run
from .answer_quality import explain_answer_check, generate_single_answer, inspect_answer
from .citation_check import check_citations, summarize as summarize_checks
from .generation import llm_available
from .rerank import format_rerank_query, rerank
from .retrieval import build_native_rerank_pipeline, build_pipeline, search
from .retrieval_details import branch_ranks, passage_key

EMBED_MODES = ["auto", "manual", "context", "multimodal"]

DEMO_QUESTIONS = [
    "Dove si trovano gli stabilimenti produttivi di 3M?",
    "Quanti stabilimenti produttivi aveva 3M nel 2018?",
    "Quali sono state le spese in conto capitale di 3M nel 2018, in milioni di dollari? Usa il rendiconto finanziario.",
    "Quanto ha speso Adobe per ricerca e sviluppo nell’esercizio 2022? Indica l’importo in milioni di dollari.",
    "Quali sono stati i ricavi e l’utile operativo del segmento AWS di Amazon nel 2022? Riporta entrambi in milioni di dollari.",
    "Come si prepara la pizza napoletana?",
]


_WEB_IN_ERROR = re.compile(r"\s*(See\s+)?(https?://|www\.)\S+", re.I)


def public_error(exc: BaseException | str, limit: int = 300) -> str:
    """Messaggio d'errore da mostrare in pagina: niente indirizzi web (Atlas e i provider a volte li includono)."""
    text = exc if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
    return _WEB_IN_ERROR.sub("", text).strip()[:limit]


def snippet(text: str | None, length: int = 70) -> str:
    """Prime parole del passaggio, su una riga, per distinguerlo dagli altri della stessa pagina."""
    flat = " ".join(str(text or "").split())
    return flat if len(flat) <= length else flat[: length - 1].rstrip() + "…"


def short_file(filename: str | None) -> str:
    name = str(filename or "Documento")
    return name[:-4] if name.lower().endswith(".pdf") else name


def page_label(page: Any) -> str:
    return "pagina n/d" if page in (None, "") else f"p. {page}"


def llm_label() -> str:
    return {"anthropic": f"Anthropic · {config.ANTHROPIC_MODEL}", "ollama": f"Ollama · {config.OLLAMA_MODEL}",
            "gemini": f"Gemini · {config.GEMINI_MODEL}",
            "gateway": f"AI Gateway · {config.AI_GATEWAY_MODEL}"}.get(config.LLM_PROVIDER, config.LLM_PROVIDER)


# ─── Stato generale ──────────────────────────────────────────────────────

def info() -> dict[str, Any]:
    mode = search_scope.load_active_mode(config.EMBED_MODE)
    status: dict[str, Any] = {"database": config.MONGODB_DB, "collection": config.MONGODB_COLLECTION,
                              "active_mode": mode, "llm": llm_label(), "llm_ready": llm_available(),
                              "rerank": f"{config.VOYAGE_RERANK_MODEL} ({config.RERANK_BACKEND})",
                              "embed_model": {"auto": config.VOYAGE_EMBED_MODEL, "manual": config.VOYAGE_EMBED_MODEL,
                                              "context": config.VOYAGE_CONTEXT_MODEL,
                                              "multimodal": config.VOYAGE_MULTIMODAL_MODEL}.get(mode),
                              "modes": {m: search_scope.MODE_DESCRIPTIONS[m][0] for m in EMBED_MODES},
                              "models": {"embed": config.VOYAGE_EMBED_MODEL, "context": config.VOYAGE_CONTEXT_MODEL,
                                         "multimodal": config.VOYAGE_MULTIMODAL_MODEL, "rerank": config.VOYAGE_RERANK_MODEL},
                              "embed_dimensions": config.VOYAGE_EMBED_DIMENSIONS,
                              "chunking": {"strategy": "structure", "size": config.CHUNK_SIZE_TOKENS,
                                           "overlap": config.CHUNK_OVERLAP_TOKENS}}
    try:
        config.get_db().command("ping")
        scope = search_scope.inspect_scope(mode)
        status["atlas"] = "ok"
        status["counts"] = {row["Modalità"]: row["Chunk presenti"] for row in scope["rows"]}
        status["documents"] = len(config.get_collection().distinct("document_id", {"embedding_mode": mode}))
        status["indexes"] = [{"mode": row["Modalità"], "name": row["Indice vettoriale"], "status": row["Stato indice"],
                              "model": row["Modello"]} for row in scope["rows"]]
        status["unavailable"] = scope["unavailable"]
        status["text_index"] = {"name": scope["text_index"], "status": scope.get("text_index_status", "MANCANTE")}
        status["ready"] = bool(status["counts"].get(mode)) and not scope["unavailable"]
    except Exception as exc:
        status.update({"atlas": public_error(exc, 200), "ready": False, "counts": {}})
    return status


_HEALTH: dict[str, Any] = {}


def health(force: bool = False) -> dict[str, Any]:
    """Controlli reali, non solo di configurazione: ogni componente viene chiamato davvero.

    Il risultato resta valido 5 minuti: aprire le pagine non costa una chiamata al modello ogni volta.
    """
    if not force and _HEALTH.get("at") and time.time() - _HEALTH["at"] < 300:
        return _HEALTH["data"]
    checks: list[dict[str, Any]] = []

    def check(name: str, what: str, fn) -> Any:
        started = time.perf_counter()
        try:
            detail = fn()
            checks.append({"name": name, "what": what, "ok": True, "detail": detail,
                           "ms": round((time.perf_counter() - started) * 1000)})
            return True
        except Exception as exc:
            checks.append({"name": name, "what": what, "ok": False, "ms": round((time.perf_counter() - started) * 1000),
                           "detail": public_error(exc)})
            return False

    mode = search_scope.load_active_mode(config.EMBED_MODE)

    def atlas():
        config.get_db().command("ping")
        return f"Database {config.MONGODB_DB} raggiungibile"

    def documents():
        count = config.get_collection().count_documents({"embedding_mode": mode})
        if not count:
            raise RuntimeError(f"Nessun passaggio caricato nella modalità attiva ({mode})")
        return f"{count:,} passaggi nella modalità {mode}".replace(",", ".")

    def indexes():
        scope = search_scope.inspect_scope(mode)
        if scope["unavailable"]:
            raise RuntimeError("Indici non interrogabili: " + ", ".join(scope["unavailable"]))
        names = [row["Indice vettoriale"] for row in scope["rows"] if row["Inclusa"] == "Sì"]
        return f"Indice testuale e vettoriale pronti ({', '.join(names)})"

    def pipeline():
        scope = search_scope.require_scope(mode)
        vectors = search_scope.prepare_vectors("capital expenditures", scope["modes"])
        native = config.RERANK_BACKEND == "native"
        builder = build_native_rerank_pipeline if native else build_pipeline
        # Stessi parametri della pagina Query: con pochi candidati Atlas rifiuta $rerank.
        options = {"top_n": config.RERANK_TOP_N} if native else {"retrieval_mode": "hybrid"}
        docs = list(config.get_collection().aggregate(builder(
            "capital expenditures", top_k=config.RETRIEVAL_TOP_K, embed_mode=mode, modes=scope["modes"], query_vectors=vectors, **options)))
        if not docs:
            raise RuntimeError("La pipeline non ha restituito risultati")
        return ("$search + $vectorSearch + $rankFusion" + (" + $rerank" if native else "")
                + f": {len(docs)} risultati, il primo da {docs[0].get('filename')}")

    def answer_model():
        from .generation import generate_answer
        out = generate_answer("Rispondi solo con la parola OK.",
                              [{"content": "OK", "filename": "verifica", "page_number": 1}])
        if not (out.get("answer") or "").strip():
            raise RuntimeError("Il modello ha risposto vuoto")
        return f"{llm_label()} risponde"

    def agent_model():
        from . import agentic_search
        agentic_search._complete('Rispondi solo con JSON: {"ok": true}', '{"ok": true}', max_tokens=20)
        return f"{agentic_search.model_label()} risponde"

    ok_atlas = check("MongoDB Atlas", "Connessione al cluster", atlas)
    if ok_atlas:
        check("Documenti", "Passaggi presenti nella modalità attiva", documents)
        check("Indici di ricerca", "Atlas Search e Vector Search interrogabili", indexes)
        check("Ricerca ibrida e rerank", "Una query vera: parole, significato, fusione e rerank", pipeline)
        check("Cifratura lato client", "CSFLE: chiave dati nel key vault, cifratura e decifratura di prova",
              field_encryption.self_test)
    check("Modello della risposta", "Una chiamata vera al modello che scrive le risposte", answer_model)
    if (config.AGENT_LLM_PROVIDER, config.AI_GATEWAY_AGENT_MODEL if config.AGENT_LLM_PROVIDER == "gateway" else "") != \
            (config.LLM_PROVIDER, config.AI_GATEWAY_MODEL if config.LLM_PROVIDER == "gateway" else ""):
        check("Modello dell'agente", "Una chiamata vera al modello che decide i passi dell'agente", agent_model)
    data = {"ok": all(c["ok"] for c in checks), "checks": checks,
            "checked_at": datetime.now(timezone.utc).isoformat()}
    _HEALTH.update({"at": time.time(), "data": data})
    return data


def available_filters() -> dict[str, list]:
    coll = config.get_collection()
    out = {}
    for field in ("filename", "category", "author", "year", "status", "file_type"):
        values = [v for v in coll.distinct(field) if v not in (None, "", [])]
        if values:
            out[field] = sorted(values, key=str)
    return out


def build_filters(values: dict[str, Any] | None) -> dict | None:
    out = {}
    for field, value in (values or {}).items():
        if value in (None, "", []):
            continue
        out[field] = int(value) if field == "year" and str(value).isdigit() else value
    return out or None


def pdf_bytes(file_id: str) -> tuple[bytes, str] | None:
    try:
        grid = GridFS(config.get_db(), collection=config.MONGODB_GRIDFS_BUCKET).get(ObjectId(file_id))
        return grid.read(), grid.filename or "documento"
    except Exception:
        return None


# ─── Query: RAG classica, passo per passo ────────────────────────────────

def _passage_view(doc: dict, *, cited: set[int] | None = None, number: int | None = None) -> dict[str, Any]:
    return {"key": passage_key(doc), "filename": doc.get("filename"), "file_label": short_file(doc.get("filename")),
            "page": doc.get("page_number"), "page_label": page_label(doc.get("page_number")),
            "snippet": snippet(doc.get("content"), 140), "content": (doc.get("content") or "")[:6000],
            "rerank_score": doc.get("_rerank_score"), "rrf_score": doc.get("score"),
            "branch_ranks": branch_ranks(doc), "gridfs_file_id": str(doc["gridfs_file_id"]) if doc.get("gridfs_file_id") else None,
            "number": number, "cited": bool(cited and number in cited)}


def run_query(question: str, *, filters: dict | None = None, top_k: int | None = None, top_n: int | None = None,
              rerank_instructions: str = "", fuzzy: bool = False, mode: str | None = None,
              threshold: float | None = None) -> Iterator[dict[str, Any]]:
    """Esegue la Query e restituisce un evento per passo, poi il risultato completo."""
    top_k = top_k or config.RETRIEVAL_TOP_K
    top_n = top_n or config.RERANK_TOP_N
    threshold = config.RERANK_RELEVANCE_THRESHOLD if threshold is None else threshold
    timings: dict[str, float] = {}
    mode = mode or search_scope.load_active_mode(config.EMBED_MODE)
    try:
        scope = search_scope.require_scope(mode)
    except Exception as exc:
        yield {"type": "error", "message": str(exc)}
        return
    yield {"type": "step", "step": "embedding", "label": "Preparo la domanda"}
    started = time.perf_counter()
    vectors = search_scope.prepare_vectors(question, scope["modes"])
    if vectors:
        timings["embedding"] = (time.perf_counter() - started) * 1000
    native = config.RERANK_BACKEND == "native"
    builder = build_native_rerank_pipeline if native else build_pipeline
    options = ({"top_n": top_n, "instructions": rerank_instructions, "include_candidates": True} if native
               else {"retrieval_mode": "hybrid"})
    pipeline = builder(question, filters=filters, top_k=top_k, embed_mode=mode, fuzzy=fuzzy,
                       modes=scope["modes"], query_vectors=vectors, **options)
    yield {"type": "step", "step": "atlas", "label": "Ricerca ibrida e reranking in Atlas" if native else "Ricerca ibrida in Atlas"}
    started = time.perf_counter()
    try:
        atlas = list(config.get_collection().aggregate(pipeline))
    except Exception as exc:
        yield {"type": "error", "message": f"Atlas non ha completato la pipeline: {exc}"}
        return
    timings["atlas"] = (time.perf_counter() - started) * 1000
    raw = sorted(atlas, key=lambda d: d.get("_hybrid_rank", 10 ** 9)) if native else atlas
    if not raw:
        yield {"type": "error", "message": "Nessun risultato con la domanda e i filtri scelti."}
        return
    if native:
        reranked = atlas[:top_n]
    elif config.VOYAGE_API_KEY:
        started = time.perf_counter()
        reranked = rerank(question, raw, top_n=top_n, instructions=rerank_instructions)
        timings["rerank"] = (time.perf_counter() - started) * 1000
    else:
        reranked = raw[:top_n]
    top_score = reranked[0].get("_rerank_score") if reranked else None
    retrieval = {"pipeline": search_scope.pipeline_code(pipeline, scope), "scope": scope["modes"],
                 "weights": search_scope.weights_caption(scope["modes"]), "dedup": search_scope.dedup_caption(),
                 "rerank_query": format_rerank_query(question, rerank_instructions),
                 "fused": [_passage_view(d) for d in raw],
                 "reranked_all": [_passage_view(d) for d in (atlas if native else reranked)],
                 "sent": [passage_key(d) for d in reranked], "top_score": top_score, "threshold": threshold,
                 "top_k": top_k, "top_n": top_n}
    yield {"type": "retrieval", **retrieval}
    if isinstance(top_score, (int, float)) and top_score < threshold:
        yield {"type": "result", "status": "off_topic", "question": question, "timings": timings,
               "message": f"Nessun passaggio abbastanza pertinente: il migliore ha un punteggio di reranking di {top_score:.3f}, "
                          f"sotto la soglia {threshold:.2f}. Il modello non viene chiamato.", **retrieval}
        return
    if not llm_available():
        yield {"type": "error", "message": "Il modello configurato non è disponibile."}
        return
    yield {"type": "step", "step": "generation", "label": "Il modello scrive la risposta"}
    started = time.perf_counter()
    reviewed = generate_single_answer(question, reranked)
    timings["generation"] = (time.perf_counter() - started) * 1000
    answer = reviewed["answer"].get("answer") or ""
    check = inspect_answer(answer, len(reranked), truncated=bool(reviewed["answer"].get("truncated")))
    blocked = check["status"] == "external_links"
    cited = set(check["valid_citations"])
    rows = [] if blocked else check_citations(answer, reranked)
    yield {"type": "result", "status": "blocked" if blocked else "ok", "question": question,
           "answer": "" if blocked else answer,
           "message": explain_answer_check("external_links") if blocked else
           (explain_answer_check(check["status"], check) if check["status"] not in {"cited", "insufficient"} else ""),
           "check_status": check["status"], "citation_check": rows, "citation_summary": summarize_checks(rows),
           "sources": [_passage_view(d, cited=cited, number=n) for n, d in enumerate(reranked, 1)],
           "model": reviewed["answer"].get("model"), "usage": reviewed["answer"].get("usage") or {},
           "timings": timings, **retrieval}


def query_comparisons(question: str, *, filters: dict | None, top_k: int, fuzzy: bool = False,
                      mode: str | None = None) -> dict[str, Any]:
    """Classifiche separate dei motori (solo testo, solo vettori), per spiegare la fusione."""
    coll = config.get_collection()
    mode = mode or search_scope.load_active_mode(config.EMBED_MODE)
    scope = search_scope.require_scope(mode)
    modes = scope["modes"]
    vectors = search_scope.prepare_vectors(question, modes)
    vector_modes = [m for m in modes if m != "multimodal"]
    common = {"filters": filters, "top_k": top_k}
    with ThreadPoolExecutor(max_workers=1 + len(vector_modes)) as pool:
        text = pool.submit(search, coll, question, embed_mode="all", retrieval_mode="text", fuzzy=fuzzy,
                           modes=modes, **common)
        vec = {m: pool.submit(search, coll, question, embed_mode=m, retrieval_mode="vector", modes=[m],
                              query_vectors=vectors, **common) for m in vector_modes}
        return {"text": [_passage_view(d) for d in text.result()[:20]],
                "vector": {m: [_passage_view(d) for d in f.result()[:20]] for m, f in vec.items()}}


# ─── Ricerca agentica: vista di una ricerca salvata ──────────────────────

def run_sources(run: dict) -> tuple[str, list[dict]]:
    result = run.get("result") or {}
    answer_event = next((e for e in reversed(run.get("events", [])) if e.get("type") == "answer"), {})
    return result.get("answer") or "", result.get("sources") or answer_event.get("sources") or []


def run_verdict(run: dict, answer: str, sources: list[dict]) -> dict[str, Any]:
    """Un solo verdetto: rischio (da verificare), lacuna (parziale/assente) o verificata."""
    events = run.get("events", [])
    grade = next((e for e in reversed(events) if e.get("type") == "grading"), None)
    answer_event = next((e for e in reversed(events) if e.get("type") == "answer"), {})
    # Ricalcolato dalle fonti salvate: le ricerche già fatte beneficiano dei controlli aggiornati.
    rows = check_citations(answer, [{**s, "page_number": s.get("page")} for s in sources]) if sources \
        else (answer_event.get("citation_check") or [])
    risks = []
    if not grade or not grade.get("evaluated", True):
        risks.append("la valutazione delle prove non è riuscita")
    answer_check = answer_event.get("answer_check")
    if answer_check and answer_check not in {"cited", "insufficient"}:
        risks.append({"uncited": "la risposta non cita i passaggi",
                      "invalid_citations": "la risposta cita passaggi che non esistono"}.get(answer_check, "il controllo di forma non è passato"))
    not_found = [r["claim"] for r in rows if r["status"] == "not_found"]
    uncited = [r["claim"] for r in rows if r["status"] == "uncited"]
    if not_found:
        risks.append("Numeri non trovati nel passaggio citato: " + ", ".join(not_found[:4]))
    if uncited:
        risks.append("Numeri senza citazione: " + ", ".join(uncited[:4]))
    elements = (grade or {}).get("elements") or []
    found_el = [e for e in elements if e.get("found")]
    gaps = [e.get("element") for e in elements if not e.get("found")]
    searches = sum(e.get("type") == "retrieval" for e in events)
    checked = len(rows) - len(uncited)
    found_numbers = sum(r["status"] in {"found", "rounded", "computed"} for r in rows)
    if risks:
        return {"kind": "risk", "label": "Da verificare", "notes": risks, "rows": rows}
    if grade and elements and not found_el and found_numbers:
        return {"kind": "partial", "label": "Parziale: dati verificati, risposta forse incompleta", "rows": rows,
                "notes": [f"I {found_numbers} numeri riportati sono nelle fonti citate; il valutatore non li considera "
                          "una risposta completa a: " + "; ".join(gaps)]}
    if grade and elements and not found_el:
        return {"kind": "absent", "label": "Non trovato nelle ricerche", "rows": rows,
                "notes": [f"L’agente ha cercato {searches} {'volta' if searches == 1 else 'volte'} senza trovare questi dati "
                          "nei passaggi recuperati, e non risponde a memoria. Non significa che i documenti non li contengano."]}
    if gaps:
        return {"kind": "partial", "label": f"Parziale · copertura {len(found_el)}/{len(elements)}",
                "rows": rows, "notes": [f"I numeri riportati sono verificati. Non trovato nelle {searches} "
                                        f"{'ricerca' if searches == 1 else 'ricerche'} eseguite: " + "; ".join(gaps)
                                        + ". Con più ricerche supplementari l’agente potrebbe trovarlo."]}
    label = (f"Riscontri numerici nelle fonti citate · {found_numbers}/{checked}" if checked
             else "Evidenze confermate dal valutatore")
    computed = [f"{r['claim']} = {r['formula']}" for r in rows if r["status"] == "computed"]
    if computed:
        return {"kind": "ok", "label": label, "rows": rows,
                "notes": ["Valori calcolati dalla risposta a partire dalle fonti: " + "; ".join(computed)]}
    return {"kind": "ok", "label": label, "notes": [], "rows": rows}


def run_view(run: dict, parent: dict | None = None) -> dict[str, Any]:
    answer, sources = run_sources(run)
    status = run.get("status")
    view = {"run": run, "answer": answer, "sources": sources, "status": status}
    if status in {"complete"} and answer:
        cited = set(inspect_answer(answer, len(sources))["valid_citations"])
        view["cited"] = sorted(n for n in cited if 1 <= n <= len(sources))
        view["verdict"] = run_verdict(run, answer, sources)
    # Il valore dell'agente si mostra solo con la risposta definitiva, non quando chiede di scegliere.
    if status in {"complete", "blocked"}:
        view["value"] = summarize_run(run, parent=parent)
    view["clarification"] = next((e for e in reversed(run.get("events", [])) if e.get("type") == "clarification"), None)
    return view


# ─── Upload ──────────────────────────────────────────────────────────────

def chunking_check(values: dict[str, Any]) -> dict[str, Any]:
    """Valida i parametri di chunking prima di aprire il flusso del caricamento."""
    from .ingestion import chunking_settings
    return chunking_settings(values)


def ingest_files(files: list[tuple[str, bytes]], *, mode: str, metadata: dict[str, Any],
                 storage: str = "gridfs", chunking: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    from .ingestion import chunking_settings, ingest_pdf
    if mode not in EMBED_MODES:
        raise ValueError(f"Modalità sconosciuta: {mode}")
    chunking = chunking_settings(chunking)
    yield {"type": "start", "total": len(files), "mode": mode, "chunking": chunking}
    for number, (name, data) in enumerate(files, start=1):
        filename = Path(name.replace("\\", "/")).name
        started = time.perf_counter()
        row = {"file": filename, "outcome": "error", "pages": 0, "chunks": 0, "detail": ""}
        yield {"type": "file_start", "number": number, "file": filename}
        try:
            if filename in {"", ".", ".."}:
                raise ValueError("Nome file non valido")
            with tempfile.TemporaryDirectory(prefix="rag-upload-") as temp_dir:
                path = Path(temp_dir) / filename
                path.write_bytes(data)
                summary = ingest_pdf(path, db=config.get_db(), collection=config.get_collection(), storage=storage,
                                     mode=mode, chunking=chunking, user_metadata={**metadata, "file_type": path.suffix.lstrip(".").lower() or "pdf"})
            skipped = summary.get("skipped", False)
            chunks = summary.get("chunks_inserted", 0)
            row.update({"outcome": "skipped" if skipped else "inserted" if chunks else "error",
                        "pages": summary.get("pages_processed", 0), "chunks": chunks,
                        "detail": summary.get("reason", "") if skipped else "" if chunks else
                        "Nessun contenuto indicizzabile estratto dal documento."})
        except Exception as exc:
            row["detail"] = public_error(exc)
        row["seconds"] = round(time.perf_counter() - started, 1)
        yield {"type": "file_done", "number": number, **row}
    yield {"type": "done"}


# ─── Admin ───────────────────────────────────────────────────────────────

def documents() -> list[dict[str, Any]]:
    """Un documento per riga, con passaggi e data di caricamento per ogni modalità (`by_mode`)."""
    rows = config.get_collection().aggregate([
        {"$group": {"_id": {"doc": "$document_id", "mode": "$embedding_mode"},
                    "filename": {"$first": "$filename"}, "title": {"$first": "$title"},
                    "author": {"$first": "$author"}, "ingested_at": {"$max": "$ingested_at"}, "chunks": {"$sum": 1},
                    "gridfs_file_id": {"$first": "$gridfs_file_id"}, "file_type": {"$first": "$file_type"},
                    "total_pages": {"$first": "$total_pages"}}},
        {"$group": {"_id": "$_id.doc", "filename": {"$first": "$filename"}, "title": {"$first": "$title"},
                    "author": {"$first": "$author"}, "ingested_at": {"$max": "$ingested_at"}, "chunks": {"$sum": "$chunks"},
                    "gridfs_file_id": {"$first": "$gridfs_file_id"}, "file_type": {"$first": "$file_type"},
                    "total_pages": {"$first": "$total_pages"}, "modes": {"$addToSet": "$_id.mode"},
                    "by_mode": {"$push": {"mode": "$_id.mode", "chunks": "$chunks", "ingested_at": "$ingested_at"}}}},
        {"$sort": {"ingested_at": -1}},
    ])
    return [{**r, "gridfs_file_id": str(r["gridfs_file_id"]) if r.get("gridfs_file_id") else None} for r in rows]


def delete_document(document_id: str, mode: str | None = None) -> dict[str, Any]:
    """Elimina i passaggi del documento (solo di una modalità, se indicata).
    Il file originale in GridFS si elimina solo quando non resta nessun passaggio che lo usa."""
    coll = config.get_collection()
    if mode is not None and mode not in EMBED_MODES:
        raise ValueError(f"Modalità sconosciuta: {mode}")
    first = coll.find_one({"document_id": document_id}, {"gridfs_file_id": 1}) or {}
    query = {"document_id": document_id, **({"embedding_mode": mode} if mode else {})}
    deleted = coll.delete_many(query).deleted_count
    if first.get("gridfs_file_id") and not coll.count_documents({"document_id": document_id}, limit=1):
        try:
            GridFS(config.get_db(), collection=config.MONGODB_GRIDFS_BUCKET).delete(ObjectId(str(first["gridfs_file_id"])))
        except Exception:
            pass
    return {"chunks_deleted": deleted}


def delete_all_documents() -> dict[str, Any]:
    coll = config.get_collection()
    deleted = coll.delete_many({}).deleted_count
    fs = GridFS(config.get_db(), collection=config.MONGODB_GRIDFS_BUCKET)
    files = 0
    for f in fs.find():
        fs.delete(f._id)
        files += 1
    return {"chunks_deleted": deleted, "files_deleted": files}


def setup_database(recreate: bool = False) -> Iterator[dict[str, Any]]:
    from .indexes import ensure_indexes
    from .init_db import initialize_database, repair_document_page_references
    yield {"type": "log", "text": f"Verifico la connessione al database {config.MONGODB_DB}."}
    config.get_db().command("ping")
    result = initialize_database()
    yield {"type": "log", "text": "Collection e indici GridFS pronti. Create: "
                                   + (", ".join(result["created"]) or "nessuna (già presenti)") + "."}
    repaired = repair_document_page_references(config.get_collection())
    yield {"type": "log", "text": f"Riferimenti di pagina controllati: {repaired}."}
    logs: list[str] = []
    if recreate:
        yield {"type": "log", "text": "Elimino e ricreo gli indici Atlas Search (può richiedere qualche secondo per indice)."}
    report = ensure_indexes(config.get_collection(), modes=tuple(EMBED_MODES), recreate=recreate,
                            on_progress=logs.append)
    for line in logs:
        yield {"type": "log", "text": line}
    yield {"type": "done", "created": report.get("created", []), "updated": report.get("updated", []),
           "unchanged": report.get("already_present", []),
           "errors": [op for op in report.get("operations", []) if op.get("error")]}


def reset_all(*, sessions: bool, recreate_indexes: bool) -> Iterator[dict[str, Any]]:
    """Riparti da zero: documenti, file GridFS, a scelta sessioni dell'agente e indici."""
    yield {"type": "log", "text": "Cancello tutti i passaggi e i file originali…"}
    removed = delete_all_documents()
    yield {"type": "log", "text": f"Cancellati {removed['chunks_deleted']} passaggi e {removed['files_deleted']} file GridFS."}
    if sessions:
        from .agentic_memory import get_agent_collections
        agent_sessions, agent_runs = get_agent_collections()
        runs = agent_runs.delete_many({}).deleted_count
        count = agent_sessions.delete_many({}).deleted_count
        yield {"type": "log", "text": f"Cancellate {count} sessioni e {runs} ricerche della ricerca agentica."}
    yield from setup_database(recreate=recreate_indexes)
