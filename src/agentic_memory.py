"""MongoDB persistence for agentic session memory and flexible run traces."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pymongo.collection import Collection

from . import config

MAX_SESSION_MESSAGES = 24
MAX_SESSION_FEEDBACK = 30


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def get_agent_collections() -> tuple[Collection, Collection]:
    db = config.get_db()
    sessions = db["agentic_sessions"]
    runs = db["agentic_runs"]
    sessions.create_index([("updated_at", -1)])
    runs.create_index([("session_id", 1), ("created_at", -1)])
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


def record_feedback(runs: Collection, sessions: Collection, *, session_id: str,
                    run_id: str, action: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    """Keep the user's decision with both its run and the session's recent memory."""
    if action not in {"accepted", "needs_refinement", "refinement_submitted"}:
        raise ValueError("Azione di feedback non valida.")
    entry = {**(details or {}), "id": str(uuid4()), "run_id": run_id,
             "action": action, "at": utc_now()}
    outcome = runs.update_one(
        {"_id": run_id, "session_id": session_id, "status": {"$in": ["complete", "blocked"]}},
        {"$push": {"feedback": entry}, "$set": {"updated_at": entry["at"]}},
    )
    if outcome.matched_count != 1:
        raise ValueError("Ricerca non trovata nella sessione o non ancora conclusa.")
    result = sessions.update_one(
        {"_id": session_id},
        {"$push": {"memory.feedback": {"$each": [entry], "$slice": -MAX_SESSION_FEEDBACK}},
         "$set": {"updated_at": entry["at"]}},
    )
    if result.matched_count != 1:
        raise RuntimeError("Feedback salvato nella ricerca, ma sessione non disponibile: impossibile aggiornare la memoria.")
    return entry


def validate_refinement(previous_question: str, question: str, *, top_k: int, top_n: int) -> str:
    """Require an explicit new question and settings that can be executed."""
    question = question.strip()
    if not question:
        raise ValueError("Scrivi la nuova domanda completa prima di avviare l’approfondimento.")
    if " ".join(question.casefold().split()) == " ".join(previous_question.casefold().split()):
        raise ValueError("Riformula la domanda: deve essere diversa dalla domanda precedente.")
    if not 5 <= top_k <= 100 or not 1 <= top_n <= 20 or top_n > top_k:
        raise ValueError("Usa 5–100 candidati e 1–20 fonti, con fonti non superiori ai candidati.")
    return question


def refinement_history(parent: dict[str, Any], answer: str,
                       summaries: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Give a follow-up the selected run's context, even if newer turns exist."""
    previous = {"role": "user", "text": parent.get("question", "")}
    if parent.get("feedback_context"):
        previous["feedback_context"] = parent["feedback_context"]
    history = [previous]
    if answer:
        history.append({"role": "assistant", "text": answer})
    matching = [item for item in summaries if item.get("turn_id") == parent.get("_id")]
    return history, matching


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
    sessions.update_one({"_id": session_id, "first_question": {"$exists": False}},
                        {"$set": {"first_question": question[:200]}})


def fail_run(
    runs: Collection,
    sessions: Collection,
    *,
    session_id: str,
    run_id: str,
    message: str,
) -> None:
    now = utc_now()
    runs.update_one(
        {"_id": run_id, "session_id": session_id},
        {"$set": {"status": "failed", "updated_at": now, "state.step": "failed"},
         "$push": {"events": {"type": "error", "comment": message[:500], "at": now}}},
    )
    sessions.update_one(
        {"_id": session_id},
        {"$set": {"active_run": None, "updated_at": now}},
    )
