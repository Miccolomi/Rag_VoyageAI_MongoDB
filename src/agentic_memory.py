"""MongoDB persistence for agentic session memory and flexible run traces."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pymongo.collection import Collection

from . import config, field_encryption

logger = logging.getLogger(__name__)

MAX_SESSION_MESSAGES = 24


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


_INDEXED: set[tuple[int, str]] = set()


def get_agent_collections() -> tuple[Collection, Collection]:
    db = config.get_db()
    sessions = db["agentic_sessions"]
    runs = db["agentic_runs"]
    # Gli indici si creano una volta per processo, non a ogni richiesta dell'interfaccia.
    if (id(db.client), db.name) not in _INDEXED:
        sessions.create_index([("updated_at", -1)])
        sessions.create_index([("created_at", -1)])
        runs.create_index([("session_id", 1), ("created_at", -1)])
        _INDEXED.add((id(db.client), db.name))
    return sessions, runs


def session_title(when: datetime | None = None) -> str:
    """Titolo con data e ora locali: le sessioni si riconoscono nell'elenco."""
    local = (when or utc_now()).astimezone()
    return f"Sessione del {local:%d/%m/%Y} alle {local:%H:%M}"


def create_session(sessions: Collection) -> dict[str, Any]:
    now = utc_now()
    doc = {
        "_id": str(uuid4()),
        "schema_version": 1,
        "title": session_title(now),
        "status": "active",
        "created_at": now,
        "updated_at": now,
        "memory": {
            "kind": "session_long_term",
            "messages": [],
            "turn_summaries": [],
            "feedback": [],
            "turn_count": 0,
        },
        "active_run": None,
    }
    sessions.insert_one(doc)
    return doc


def start_run(runs: Collection, sessions: Collection, session_id: str, question: str, *,
              parent_run_id: str | None = None, feedback_id: str | None = None,
              settings: dict[str, Any] | None = None, feedback_context: str = "") -> str:
    # The feedback ID also identifies the follow-up run, preventing a second
    # submission of the same saved feedback from creating another run.
    run_id = feedback_id or str(uuid4())
    now = utc_now()
    runs.insert_one({
        "_id": run_id,
        "schema_version": 1,
        "session_id": session_id,
        "question": question,
        "status": "running",
        "created_at": now,
        "updated_at": now,
        "state": {"step": "planning", "iteration": 0, "candidate_count": 0},
        "events": [],
        "parent_run_id": parent_run_id,
        "feedback_id": feedback_id,
        "feedback_context": feedback_context,
        "settings": settings or {},
    })
    sessions.update_one(
        {"_id": session_id},
        {"$set": {
            "active_run": {"run_id": run_id, "status": "running", "step": "planning", "started_at": now},
            "updated_at": now,
        }},
    )
    return run_id


def add_event(
    runs: Collection,
    sessions: Collection,
    session_id: str,
    run_id: str,
    event: dict[str, Any],
    *,
    state: dict[str, Any] | None = None,
) -> None:
    now = utc_now()
    runs.update_one(
        {"_id": run_id, "session_id": session_id},
        {"$push": {"events": {**event, "at": now}},
         "$set": {"updated_at": now, **({"state": state} if state is not None else {})}},
    )
    if state is not None:
        sessions.update_one(
            {"_id": session_id},
            {"$set": {"active_run.step": state.get("step"), "active_run.iteration": state.get("iteration"), "updated_at": now}},
        )


def finish_run(
    runs: Collection,
    sessions: Collection,
    *,
    session_id: str,
    run_id: str,
    question: str,
    answer: str,
    sources: list[dict[str, Any]],
    status: str,
    feedback_context: str = "",
    parent_run_id: str | None = None,
) -> None:
    now = utc_now()
    runs.update_one(
        {"_id": run_id, "session_id": session_id},
        {"$set": {"status": status, "updated_at": now, "state.step": "complete",
                  "result": {"answer": answer, "sources": sources}}},
    )
    messages = [
        {"turn_id": run_id, "role": "user", "text": question, "at": now,
         **({"feedback_context": feedback_context, "parent_run_id": parent_run_id} if feedback_context else {})},
        {"turn_id": run_id, "role": "assistant", "text": answer[:6000], "sources": sources[:20], "at": now},
    ]
    turn_summary = {
        "turn_id": run_id,
        "question": question[:500],
        "answer_summary": " ".join(answer.split())[:700],
        "sources": sources[:5],
        **({"feedback_context": feedback_context, "parent_run_id": parent_run_id} if feedback_context else {}),
        "at": now,
    }
    sessions.update_one(
        {"_id": session_id},
        {
            "$push": {
                "memory.messages": {"$each": messages, "$slice": -MAX_SESSION_MESSAGES},
                "memory.turn_summaries": {"$each": [turn_summary], "$slice": -30},
            },
            "$inc": {"memory.turn_count": 1},
            "$set": {
                "active_run": None,
                "updated_at": now,
                "last_run_id": run_id,
                "last_question": question[:200],
            },
        },
    )
    # La prima domanda è cifrata lato client (CSFLE): in Atlas resta un dato binario.
    # Se la cifratura non è disponibile si salva in chiaro: la risposta non va mai persa.
    # Il motivo del fallimento resta nella sessione e nel log: mai un fallimento silenzioso.
    fields: dict[str, Any] = {}
    try:
        fields["first_question"] = field_encryption.encrypt_text(question[:200])
        if not field_encryption.is_encrypted(fields["first_question"]):
            fields["encryption_error"] = "Chiave master locale non trovata su questo computer."
    except Exception as exc:
        fields["first_question"] = question[:200]
        fields["encryption_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    if "encryption_error" in fields:
        logger.warning("Prima domanda salvata in chiaro: %s", fields["encryption_error"])
    sessions.update_one({"_id": session_id, "first_question": {"$exists": False}}, {"$set": fields})


def fail_run(
    runs: Collection,
    sessions: Collection,
    *,
    session_id: str,
    run_id: str,
    message: str,
) -> None:
    now = utc_now()
    # Solo una ricerca ancora in corso: una già conclusa non diventa "fallita" se la pagina si chiude dopo.
    runs.update_one(
        {"_id": run_id, "session_id": session_id, "status": "running"},
        {"$set": {"status": "failed", "updated_at": now, "state.step": "failed"},
         "$push": {"events": {"type": "error", "comment": message[:500], "at": now}}},
    )
    sessions.update_one(
        {"_id": session_id},
        {"$set": {"active_run": None, "updated_at": now}},
    )
