"""Demo RAG su MongoDB Atlas: server web.

Avvio:
    .venv/bin/python -m uvicorn server:app --port 8000 --reload
poi apri http://localhost:8000

Le API restituiscono JSON; Query, Ricerca agentica, Upload e preparazione del
database restituiscono un flusso di eventi (una riga JSON per evento), così la
pagina mostra ogni passo mentre avviene.
"""
from __future__ import annotations

import json
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterator

from bson import ObjectId
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, Response, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from src import agent, agentic_memory, config, field_encryption, search_scope, web_services

WEB = Path(__file__).resolve().parent / "web"


def _default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    return str(value)


def _json(data: Any, status: int = 200) -> Response:
    return Response(json.dumps(data, default=_default, ensure_ascii=False), status_code=status,
                    media_type="application/json")


def _stream(events: Iterator[dict[str, Any]]) -> StreamingResponse:
    """Un evento per riga (NDJSON). Un errore non gestito diventa un evento, non una pagina rotta."""
    def lines():
        try:
            for event in events:
                if event.get("type") == "error" and event.get("message"):
                    event = {**event, "message": web_services.public_error(str(event["message"]), 600)}
                yield json.dumps(event, default=_default, ensure_ascii=False) + "\n"
        except Exception as exc:
            yield json.dumps({"type": "error", "message": web_services.public_error(exc)}, ensure_ascii=False) + "\n"
    return StreamingResponse(lines(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _collections():
    sessions, runs = agentic_memory.get_agent_collections()
    return sessions, runs


# ─── Pagine ──────────────────────────────────────────────────────────────

async def index(request: Request) -> Response:
    return FileResponse(WEB / "index.html", headers={"Cache-Control": "no-store"})


# ─── Generali ────────────────────────────────────────────────────────────

def api_info(request: Request) -> Response:
    data = web_services.info()
    data["demo_questions"] = web_services.DEMO_QUESTIONS
    return _json(data)


def api_health(request: Request) -> Response:
    return _json(web_services.health(force=request.query_params.get("force") == "1"))


def api_filters(request: Request) -> Response:
    try:
        return _json(web_services.available_filters())
    except Exception as exc:
        return _json({"error": web_services.public_error(exc)}, 500)


def api_file(request: Request) -> Response:
    found = web_services.pdf_bytes(request.path_params["file_id"])
    if not found:
        return _json({"error": "File non trovato in GridFS"}, 404)
    data, name = found
    media = "application/pdf" if name.lower().endswith(".pdf") else "application/octet-stream"
    return Response(data, media_type=media, headers={"Content-Disposition": f'inline; filename="{name}"'})


# ─── Query ───────────────────────────────────────────────────────────────

async def api_query(request: Request) -> Response:
    body = await request.json()
    question = (body.get("question") or "").strip()
    if not question:
        return _json({"error": "Scrivi una domanda."}, 400)
    return _stream(web_services.run_query(
        question, filters=web_services.build_filters(body.get("filters")), top_k=body.get("top_k"),
        top_n=body.get("top_n"), rerank_instructions=body.get("rerank_instructions") or "",
        fuzzy=bool(body.get("fuzzy")), threshold=body.get("threshold")))


async def api_query_compare(request: Request) -> Response:
    body = await request.json()
    try:
        return _json(web_services.query_comparisons(
            body["question"], filters=web_services.build_filters(body.get("filters")),
            top_k=body.get("top_k") or config.RETRIEVAL_TOP_K, fuzzy=bool(body.get("fuzzy"))))
    except Exception as exc:
        return _json({"error": web_services.public_error(exc)}, 500)


# ─── Ricerca agentica ────────────────────────────────────────────────────

def api_sessions(request: Request) -> Response:
    sessions, runs = _collections()
    rows = list(sessions.find({}, {"title": 1, "created_at": 1, "updated_at": 1, "memory.turn_count": 1,
                                   "first_question": 1, "memory.turn_summaries": {"$slice": 1}})
                .sort("created_at", -1).limit(50))
    out = []
    for r in rows:
        memory = r.get("memory") or {}
        stored = r.get("first_question")
        encrypted = field_encryption.is_encrypted(stored)
        first = field_encryption.decrypt_text(stored) if encrypted else stored
        if encrypted and first is None:
            first = "domanda cifrata: chiave non disponibile"
        first = first or next((t.get("question") for t in memory.get("turn_summaries") or []), None)
        out.append({"id": r["_id"], "title": r.get("title"), "created_at": r.get("created_at"),
                    "updated_at": r.get("updated_at"), "turns": memory.get("turn_count", 0),
                    "first_question": first, "encrypted": encrypted})
    return _json(out)


_PROTECTION_FIELDS = {"title": 1, "created_at": 1, "memory.turn_count": 1, "first_question": 1,
                      "encryption_error": 1}


def api_session_protection(request: Request) -> Response:
    """Lo stesso documento visto dall'applicazione (con la chiave) e com'è salvato in Atlas."""
    sessions, runs = _collections()
    doc = sessions.find_one({"_id": request.path_params["session_id"]}, _PROTECTION_FIELDS)
    if not doc:
        return _json({"error": "Sessione non trovata"}, 404)
    return _json(_protection_view(doc))


def api_latest_protection(request: Request) -> Response:
    """L'ultima sessione con la prima domanda cifrata: il confronto dal vivo della home."""
    sessions, runs = _collections()
    doc = sessions.find_one({"first_question": {"$type": "binData"}}, _PROTECTION_FIELDS,
                            sort=[("created_at", -1)])
    if not doc:
        return _json({"empty": True, "configured": field_encryption.is_configured()})
    return _json(_protection_view(doc))


def _protection_view(doc: dict[str, Any]) -> dict[str, Any]:
    stored = doc.get("first_question")
    return {
        "id": doc["_id"], "title": doc.get("title"), "created_at": doc.get("created_at"),
        "turns": (doc.get("memory") or {}).get("turn_count", 0),
        "field": field_encryption.ENCRYPTED_FIELD,
        "has_question": stored is not None,
        "encrypted": field_encryption.is_encrypted(stored),
        "app_view": field_encryption.decrypt_text(stored),
        "stored_view": field_encryption.stored_view(stored),
        "algorithm": field_encryption.ALGORITHM_NAME,
        "key_alt_name": config.CSFLE_KEY_ALT_NAME,
        "key_vault": config.CSFLE_KEY_VAULT_NAMESPACE,
        "configured": field_encryption.is_configured(),
        "encryption_error": doc.get("encryption_error"),
    }


def api_new_session(request: Request) -> Response:
    sessions, runs = _collections()
    # Riusa una sessione mai usata invece di riempire l'elenco di sessioni vuote.
    empty = sessions.find_one({"$or": [{"memory.turn_count": 0}, {"memory.turn_count": {"$exists": False}}],
                               "active_run": None}, sort=[("updated_at", -1)])
    if empty:
        # La sessione vuota riusata prende data e ora di adesso.
        now = agentic_memory.utc_now()
        sessions.update_one({"_id": empty["_id"]}, {"$set": {
            "created_at": now, "updated_at": now, "title": agentic_memory.session_title(now)}})
        empty = sessions.find_one({"_id": empty["_id"]})
    session = empty or agentic_memory.create_session(sessions)
    return _json({"id": session["_id"], "title": session.get("title"), "created_at": session.get("created_at")})


def api_session_runs(request: Request) -> Response:
    sessions, runs = _collections()
    rows = list(runs.find({"session_id": request.path_params["session_id"]},
                          {"question": 1, "status": 1, "created_at": 1, "parent_run_id": 1})
                .sort("created_at", -1).limit(30))
    return _json([{"id": r["_id"], "question": r.get("question"), "status": r.get("status"),
                   "created_at": r.get("created_at"), "parent_run_id": r.get("parent_run_id")} for r in rows])


def api_run(request: Request) -> Response:
    sessions, runs = _collections()
    run = runs.find_one({"_id": request.path_params["run_id"]})
    if not run:
        return _json({"error": "Ricerca non trovata"}, 404)
    parent = runs.find_one({"_id": run["parent_run_id"]}) if run.get("parent_run_id") else None
    return _json(web_services.run_view(run, parent))


async def api_agent(request: Request) -> Response:
    body = await request.json()
    sessions, runs = _collections()
    session_id = body.get("session_id")
    if not session_id or not sessions.find_one({"_id": session_id}, {"_id": 1}):
        session_id = agentic_memory.create_session(sessions)["_id"]
    question = (body.get("question") or "").strip()
    if not question:
        return _json({"error": "Scrivi una domanda."}, 400)
    clarification = body.get("clarification") or {}
    started = time.perf_counter()
    memory, summaries = agent.planning_context(sessions, session_id)
    memory_ms = (time.perf_counter() - started) * 1000
    filters = web_services.build_filters(body.get("filters"))
    if clarification.get("filters"):
        filters = {**(filters or {}), **clarification["filters"]}
    req = agent.AgentRequest(
        session_id=session_id, question=question,
        top_k=int(body.get("top_k") or config.AGENT_TOP_K), top_n=int(body.get("top_n") or config.RERANK_TOP_N),
        retries=int(body.get("retries", 1)), filters=filters,
        rerank_instructions=body.get("rerank_instructions") or "",
        feedback_context=clarification.get("context") or body.get("feedback_context") or "",
        parent_run_id=clarification.get("parent_run_id") or body.get("parent_run_id"),
        is_clarification=bool(clarification), skip_ambiguity=bool(clarification.get("skip_ambiguity")),
        embed_mode=search_scope.load_active_mode(config.EMBED_MODE),
        planning_memory=memory, planning_summaries=summaries, memory_read_ms=memory_ms)

    def events():
        yield {"type": "session", "session_id": session_id}
        yield from agent.run_agent(req, runs=runs, sessions=sessions, collection=config.get_collection())
    return _stream(events())


# ─── Upload ──────────────────────────────────────────────────────────────

async def api_upload(request: Request) -> Response:
    form = await request.form()
    mode = form.get("mode") or "auto"
    metadata = json.loads(form.get("metadata") or "{}")
    files = [(f.filename, await f.read()) for f in form.getlist("files") if getattr(f, "filename", None)]
    if not files:
        return _json({"error": "Nessun file selezionato."}, 400)
    return _stream(web_services.ingest_files(files, mode=mode, metadata=metadata,
                                             storage=form.get("storage") or "gridfs"))


# ─── Admin ───────────────────────────────────────────────────────────────

def api_documents(request: Request) -> Response:
    return _json(web_services.documents())


def api_delete_document(request: Request) -> Response:
    return _json(web_services.delete_document(request.path_params["document_id"]))


def api_delete_all(request: Request) -> Response:
    return _json(web_services.delete_all_documents())


def api_delete_sessions(request: Request) -> Response:
    sessions, runs = _collections()
    return _json({"runs_deleted": runs.delete_many({}).deleted_count,
                  "sessions_deleted": sessions.delete_many({}).deleted_count})


def api_count_sessions(request: Request) -> Response:
    sessions, runs = _collections()
    return _json({"sessions": sessions.count_documents({"memory.turn_count": {"$gt": 0}}),
                  "runs": runs.estimated_document_count()})


async def api_active_mode(request: Request) -> Response:
    body = await request.json()
    search_scope.save_active_mode(body["mode"])
    return _json({"active_mode": body["mode"]})


async def api_setup(request: Request) -> Response:
    body = await request.json() if await request.body() else {}
    return _stream(web_services.setup_database(recreate=bool(body.get("recreate"))))


async def api_reset(request: Request) -> Response:
    body = await request.json()
    if body.get("confirm") != "AZZERA":
        return _json({"error": "Conferma mancante."}, 400)
    return _stream(web_services.reset_all(sessions=bool(body.get("sessions")),
                                          recreate_indexes=bool(body.get("recreate_indexes"))))


routes = [
    Route("/", index),
    Route("/api/info", api_info),
    Route("/api/filters", api_filters),
    Route("/api/health", api_health),
    Route("/api/files/{file_id}", api_file),
    Route("/api/query", api_query, methods=["POST"]),
    Route("/api/query/compare", api_query_compare, methods=["POST"]),
    Route("/api/sessions", api_sessions),
    Route("/api/sessions", api_new_session, methods=["POST"]),
    Route("/api/sessions/{session_id}/runs", api_session_runs),
    Route("/api/sessions/{session_id}/protection", api_session_protection),
    Route("/api/protection/latest", api_latest_protection),
    Route("/api/runs/{run_id}", api_run),
    Route("/api/agent", api_agent, methods=["POST"]),
    Route("/api/upload", api_upload, methods=["POST"]),
    Route("/api/admin/documents", api_documents),
    Route("/api/admin/documents", api_delete_all, methods=["DELETE"]),
    Route("/api/admin/documents/{document_id}", api_delete_document, methods=["DELETE"]),
    Route("/api/admin/sessions", api_count_sessions),
    Route("/api/admin/sessions", api_delete_sessions, methods=["DELETE"]),
    Route("/api/admin/mode", api_active_mode, methods=["POST"]),
    Route("/api/admin/setup", api_setup, methods=["POST"]),
    Route("/api/admin/reset", api_reset, methods=["POST"]),
    Mount("/web", StaticFiles(directory=WEB), name="web"),
]

app = Starlette(routes=routes)
