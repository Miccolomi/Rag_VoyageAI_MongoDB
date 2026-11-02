"""Banco di prova: le domande di FinanceBench con risposta di riferimento, misurate sulla demo.

Tre percorsi, gli stessi che il cliente vede nelle pagine:
  retrieval  la sola ricerca in Atlas (ibrida + rerank): trova la pagina giusta?
  query      la RAG classica della pagina Query: risposta corretta, sbagliata o astensione?
  agent      la Ricerca agentica, senza memoria tra domande e senza fermarsi a chiedere l'anno.

La risposta di riferimento non arriva mai al modello che risponde: serve solo al
valutatore, che confronta la risposta data con quella attesa. Ogni esecuzione è
salvata in MongoDB (collection `benchmark_runs`), così la pagina mostra subito
l'ultima misura senza rifarla davanti al cliente.
"""
from __future__ import annotations

import json
import statistics
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import agentic_search, config, search_scope

QUESTIONS_PATH = Path(__file__).resolve().parent.parent / "data" / "financebench_100" / "questions_reference.jsonl"
PATHS = ("retrieval", "query", "agent")
TYPE_LABELS = {"metrics-generated": "Dati estratti dai bilanci", "domain-relevant": "Domande da analista",
               "novel-generated": "Domande nuove e ragionamento"}
RETRIEVAL_TOP_K = 50
RETRIEVAL_TOP_N = 10

_lock = threading.Lock()
_job: dict[str, Any] = {"id": None, "stop": False}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def runs_collection():
    coll = config.get_db()["benchmark_runs"]
    coll.create_index([("created_at", -1)])
    return coll


# ─── Domande ─────────────────────────────────────────────────────────────

def load_questions(path: Path | None = None) -> list[dict[str, Any]]:
    """Domande con risposta e pagine di evidenza (FinanceBench numera da 0: qui da 1)."""
    path = path or QUESTIONS_PATH
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        pages = set()
        for evidence in row.get("evidence") or []:
            page = str(evidence.get("evidence_page_num", "")).strip()
            if page.isdigit() and evidence.get("doc_name") in (None, row.get("doc_name")):
                pages.add(int(page) + 1)
        out.append({"id": row.get("financebench_id"), "company": row.get("company"),
                    "question": row.get("question", ""), "answer": row.get("answer", ""),
                    "type": row.get("question_type"), "filename": f"{row.get('doc_name', '')}.pdf",
                    "pages": sorted(pages)})
    return out


def available_questions() -> tuple[list[dict[str, Any]], str]:
    """Solo le domande il cui documento è caricato nella modalità attiva."""
    mode = search_scope.load_active_mode(config.EMBED_MODE)
    loaded = set(config.get_collection().distinct("filename", {"embedding_mode": mode}))
    return [q for q in load_questions() if q["filename"] in loaded], mode


def pick(questions: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    """Campione ripetibile e bilanciato tra i tre tipi di domanda."""
    by_type: dict[str, list[dict[str, Any]]] = {}
    for q in sorted(questions, key=lambda q: q["id"] or ""):
        by_type.setdefault(q["type"] or "", []).append(q)
    picked: list[dict[str, Any]] = []
    while len(picked) < min(size, len(questions)):
        for bucket in by_type.values():
            if bucket and len(picked) < size:
                picked.append(bucket.pop(0))
    return picked


# ─── Valutazione di un percorso ──────────────────────────────────────────

def _rank(sources: list[dict[str, Any]], filename: str, pages: list[int] | None) -> int | None:
    for rank, doc in enumerate(sources, start=1):
        page = doc.get("page_number", doc.get("page"))
        if doc.get("filename") == filename and (pages is None or page in pages):
            return rank
    return None


def _hits(sources: list[dict[str, Any]], item: dict[str, Any]) -> dict[str, Any]:
    return {"doc_rank": _rank(sources, item["filename"], None),
            "page_rank": _rank(sources, item["filename"], item["pages"]) if item["pages"] else None,
            "sources": [{"filename": s.get("filename"), "page": s.get("page_number", s.get("page"))}
                        for s in sources[:10]]}


def _retrieve(question: str) -> list[dict[str, Any]]:
    from .rerank import rerank
    from .retrieval import build_native_rerank_pipeline, search
    mode = search_scope.load_active_mode(config.EMBED_MODE)
    scope = search_scope.require_scope(mode)
    vectors = search_scope.prepare_vectors(question, scope["modes"])
    coll = config.get_collection()
    if config.RERANK_BACKEND == "native":
        return list(coll.aggregate(build_native_rerank_pipeline(
            question, top_k=RETRIEVAL_TOP_K, top_n=RETRIEVAL_TOP_N, embed_mode=mode,
            modes=scope["modes"], query_vectors=vectors)))
    candidates = search(coll, question, top_k=RETRIEVAL_TOP_K, embed_mode=mode, retrieval_mode="hybrid",
                        modes=scope["modes"], query_vectors=vectors)
    return rerank(question, candidates, top_n=min(RETRIEVAL_TOP_N, len(candidates)))


def eval_retrieval(item: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    docs = _retrieve(item["question"])
    return {"ms": round((time.perf_counter() - started) * 1000), **_hits(docs, item)}


def eval_query(item: dict[str, Any]) -> dict[str, Any]:
    from . import web_services
    started = time.perf_counter()
    result, error = None, None
    for event in web_services.run_query(item["question"]):
        if event["type"] == "result":
            result = event
        elif event["type"] == "error":
            error = event["message"]
    ms = round((time.perf_counter() - started) * 1000)
    if error and not result:
        raise RuntimeError(error)
    sources = [{"filename": s["filename"], "page": s["page"]} for s in result.get("sources") or []]
    if result["status"] in {"off_topic", "blocked"}:
        return {"ms": ms, "answer": "", "status": "abstained", **_hits(sources, item)}
    return {"ms": ms, "answer": (result.get("answer") or "")[:2000], "status": "answered",
            "tokens": (result.get("usage") or {}).get("input_tokens"), **_hits(sources, item)}


def eval_agent(item: dict[str, Any]) -> dict[str, Any]:
    from . import agent, agentic_memory
    db = config.get_db()
    sessions, runs = db["benchmark_agent_sessions"], db["benchmark_agent_runs"]
    session = agentic_memory.create_session(sessions)
    req = agent.AgentRequest(session_id=session["_id"], question=item["question"], skip_ambiguity=True,
                             embed_mode=search_scope.load_active_mode(config.EMBED_MODE))
    started = time.perf_counter()
    run_id = None
    for event in agent.run_agent(req, runs=runs, sessions=sessions, collection=config.get_collection()):
        if event["type"] == "start":
            run_id = event["run_id"]
    ms = round((time.perf_counter() - started) * 1000)
    run = runs.find_one({"_id": run_id}) or {}
    result = run.get("result") or {}
    sources = result.get("sources") or []
    searches = sum(e.get("type") == "retrieval" for e in run.get("events", []))
    if run.get("status") == "failed":
        error = next((e.get("comment") for e in reversed(run.get("events", [])) if e.get("type") == "error"), "")
        raise RuntimeError(error or "ricerca agentica non riuscita")
    status = "abstained" if run.get("status") == "blocked" or not result.get("answer") else "answered"
    return {"ms": ms, "answer": (result.get("answer") or "")[:2000], "status": status, "searches": searches,
            "run_id": run_id, **_hits(sources, item)}


_JUDGE_SYSTEM = """You grade answers to questions about company financial filings.
You receive the question, the reference answer and the candidate answer (often in Italian).
Rules:
- correct: the candidate states the same facts or value as the reference. Formatting, units written
  differently (1577 vs 1.577 milioni di dollari), language and rounding within about 1% are fine.
  Extra correct context is fine.
- incorrect: the candidate gives a different value or conclusion, contradicts the reference,
  or misses a required part of it.
- abstained: the candidate does not answer, says the documents do not contain the information,
  or asks for clarification.
Return only JSON: {"verdict": "correct" | "incorrect" | "abstained", "reason": "<max 25 words, in Italian>"}"""


def judge(question: str, reference: str, answer: str) -> dict[str, str]:
    if not answer.strip():
        return {"verdict": "abstained", "reason": "Nessuna risposta: il sistema si è astenuto."}
    prompt = json.dumps({"question": question, "reference_answer": reference, "candidate_answer": answer},
                        ensure_ascii=False)
    try:
        data = agentic_search._read_json(agentic_search._complete(_JUDGE_SYSTEM, prompt, max_tokens=200))
        verdict = data.get("verdict")
        if verdict in {"correct", "incorrect", "abstained"}:
            return {"verdict": verdict, "reason": str(data.get("reason") or "")[:300]}
    except Exception as exc:
        return {"verdict": "unjudged", "reason": f"Valutatore non disponibile: {type(exc).__name__}"}
    return {"verdict": "unjudged", "reason": "Risposta del valutatore non leggibile."}


EVALUATORS = {"retrieval": eval_retrieval, "query": eval_query, "agent": eval_agent}


def evaluate(item: dict[str, Any], paths: list[str]) -> dict[str, Any]:
    row = {"id": item["id"], "company": item["company"], "type": item["type"], "question": item["question"],
           "reference": item["answer"], "filename": item["filename"], "pages": item["pages"]}
    for path in paths:
        try:
            outcome = EVALUATORS[path](item)
            if path != "retrieval":
                outcome.update(judge(item["question"], item["answer"], outcome["answer"])
                               if outcome["status"] == "answered" else
                               {"verdict": "abstained", "reason": "Il sistema si è astenuto: nessun passaggio pertinente."})
        except Exception as exc:
            outcome = {"error": f"{type(exc).__name__}: {str(exc)[:240]}"}
        row[path] = outcome
    return row


# ─── Sintesi ─────────────────────────────────────────────────────────────

def summarize(rows: list[dict[str, Any]], paths: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for path in paths:
        results = [r[path] for r in rows if path in r]
        ok = [r for r in results if "error" not in r]
        entry: dict[str, Any] = {"measured": len(ok), "errors": len(results) - len(ok),
                                 "median_ms": round(statistics.median(r["ms"] for r in ok)) if ok else None}
        with_pages = [r for r in ok if r.get("sources") is not None]
        n = len(with_pages) or 1
        entry["page_found"] = sum(r.get("page_rank") is not None for r in with_pages) / n if with_pages else None
        entry["doc_found"] = sum(r.get("doc_rank") is not None for r in with_pages) / n if with_pages else None
        if path == "retrieval":
            entry["page_at_1"] = sum((r.get("page_rank") or 99) <= 1 for r in ok) / len(ok) if ok else None
            entry["page_at_5"] = sum((r.get("page_rank") or 99) <= 5 for r in ok) / len(ok) if ok else None
        else:
            judged = [r for r in ok if r.get("verdict") in {"correct", "incorrect", "abstained"}]
            m = len(judged) or 1
            for verdict in ("correct", "incorrect", "abstained"):
                entry[verdict] = sum(r["verdict"] == verdict for r in judged) / m if judged else None
            entry["judged"] = len(judged)
            by_type: dict[str, dict[str, Any]] = {}
            for row in rows:
                res = row.get(path) or {}
                if res.get("verdict") in {"correct", "incorrect", "abstained"}:
                    t = by_type.setdefault(row["type"], {"n": 0, "correct": 0})
                    t["n"] += 1
                    t["correct"] += res["verdict"] == "correct"
            entry["by_type"] = {k: {**v, "label": TYPE_LABELS.get(k, k)} for k, v in by_type.items()}
            if path == "agent":
                searches = [r.get("searches") or 0 for r in ok]
                entry["multi_search"] = sum(s > 1 for s in searches) / len(searches) if searches else None
        out[path] = entry
    return out


# ─── Esecuzione in background ────────────────────────────────────────────

def running_id() -> str | None:
    return _job["id"]


def start(paths: list[str], size: int, workers: int = 2) -> str:
    paths = [p for p in PATHS if p in paths]
    if not paths:
        raise ValueError("Scegli almeno un percorso.")
    with _lock:
        if _job["id"]:
            raise RuntimeError("C'è già una misura in corso.")
        questions, mode = available_questions()
        if not questions:
            raise RuntimeError("Nessuna domanda di FinanceBench ha il suo documento caricato nella modalità attiva.")
        items = pick(questions, size)
        run_id = str(uuid.uuid4())
        runs_collection().insert_one({
            "_id": run_id, "created_at": _now(), "status": "running", "paths": paths, "mode": mode,
            "settings": {"size": len(items), "available": len(questions), "llm": config.LLM_PROVIDER,
                         "agent_llm": config.AGENT_LLM_PROVIDER, "rerank": config.VOYAGE_RERANK_MODEL,
                         "embed": config.VOYAGE_EMBED_MODEL},
            "progress": {"done": 0, "total": len(items)}, "rows": [], "summary": {}})
        _job.update({"id": run_id, "stop": False})
    threading.Thread(target=_work, args=(run_id, items, paths, workers), daemon=True).start()
    return run_id


def stop() -> None:
    _job["stop"] = True


def _work(run_id: str, items: list[dict[str, Any]], paths: list[str], workers: int) -> None:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    coll = runs_collection()
    rows: list[dict[str, Any]] = []
    status = "done"
    try:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            pending = {}
            queue = list(items)
            while queue or pending:
                while queue and len(pending) < max(1, workers) and not _job["stop"]:
                    item = queue.pop(0)
                    pending[pool.submit(evaluate, item, paths)] = item
                if not pending:
                    break
                done = next(as_completed(pending))
                pending.pop(done)
                row = done.result()
                rows.append(row)
                coll.update_one({"_id": run_id}, {"$push": {"rows": row},
                                                   "$set": {"progress.done": len(rows),
                                                            "summary": summarize(rows, paths)}})
        if _job["stop"]:
            status = "stopped"
    except Exception as exc:
        status = "failed"
        coll.update_one({"_id": run_id}, {"$set": {"error": f"{type(exc).__name__}: {exc}"}})
    finally:
        coll.update_one({"_id": run_id}, {"$set": {"status": status, "finished_at": _now(),
                                                   "summary": summarize(rows, paths)}})
        _job.update({"id": None, "stop": False})


def list_runs(limit: int = 20) -> list[dict[str, Any]]:
    coll = runs_collection()
    # Una misura "in corso" che non è più in esecuzione (server riavviato) viene chiusa.
    coll.update_many({"status": "running", "_id": {"$ne": _job["id"]}},
                     {"$set": {"status": "interrupted", "finished_at": _now()}})
    return list(coll.find({}, {"rows": 0}).sort("created_at", -1).limit(limit))


def get_run(run_id: str) -> dict[str, Any] | None:
    return runs_collection().find_one({"_id": run_id})


def delete_run(run_id: str) -> int:
    return runs_collection().delete_one({"_id": run_id, "status": {"$ne": "running"}}).deleted_count


def overview() -> dict[str, Any]:
    try:
        questions, mode = available_questions()
    except Exception as exc:
        return {"error": str(exc), "available": 0, "total": len(load_questions())}
    types: dict[str, int] = {}
    for q in questions:
        types[q["type"]] = types.get(q["type"], 0) + 1
    return {"available": len(questions), "total": len(load_questions()), "mode": mode, "running": _job["id"],
            "types": {k: {"n": v, "label": TYPE_LABELS.get(k, k)} for k, v in types.items()},
            "questions_file": Path(QUESTIONS_PATH).exists()}

