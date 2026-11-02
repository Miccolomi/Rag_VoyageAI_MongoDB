"""Ciclo della ricerca agentica, indipendente dall'interfaccia.

`run_agent` è un generatore: esegue un passo alla volta, salva ogni evento in
MongoDB (`agentic_runs`) e lo restituisce subito, così l'interfaccia web può
mostrarlo in diretta. L'ultimo evento è sempre {"type": "done", ...}.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterator

from . import agentic_memory, agentic_search, answer_quality, config, generation, search_scope
from .citation_check import check_citations
from .retrieval_details import passage_key

STOP_REASONS = {
    "evidence_sufficient": "Il valutatore ritiene sufficienti i passaggi raccolti.",
    "evaluation_unavailable": "Il valutatore non ha prodotto un esito utilizzabile.",
    "limit_reached": "È stato raggiunto il limite di ricerche configurato.",
    "awaiting_feedback": "La ricerca si ferma dopo il primo tentativo e attende il tuo feedback.",
    "no_followup_query": "Il valutatore non ha proposto una query di approfondimento.",
    "duplicate_query": "La query proposta ripete una ricerca già effettuata.",
    "rephrased_duplicate": "La ricerca proposta ripete gli stessi termini sostanziali di una ricerca precedente.",
    "off_topic": "Il passaggio più pertinente è sotto la soglia: la domanda sembra fuori tema rispetto ai documenti.",
    "no_results": "La ricerca non ha restituito passaggi.",
}


@dataclass
class AgentRequest:
    session_id: str
    question: str
    top_k: int = config.AGENT_TOP_K
    top_n: int = config.RERANK_TOP_N
    retries: int = 1
    filters: dict[str, Any] | None = None
    rerank_instructions: str = ""
    feedback_context: str = ""
    parent_run_id: str | None = None
    feedback_id: str | None = None
    is_clarification: bool = False
    skip_ambiguity: bool = False
    embed_mode: str = config.EMBED_MODE
    planning_memory: list[dict[str, Any]] = field(default_factory=list)
    planning_summaries: list[dict[str, Any]] = field(default_factory=list)
    memory_read_ms: float = 0.0


def planning_context(sessions, session_id: str) -> tuple[list[dict], list[dict]]:
    """Ultimi messaggi e riepiloghi della sessione: servono a capire «E nel 2017?»."""
    session = sessions.find_one({"_id": session_id}) or {}
    memory = session.get("memory") or {}
    return (memory.get("messages") or [])[-8:], memory.get("turn_summaries") or []


def _fuse(rounds: list[list[dict]]) -> dict[str, dict[str, Any]]:
    fused: dict[str, dict[str, Any]] = {}
    for round_idx, ranked in enumerate(rounds, start=1):
        for rank, doc in enumerate(ranked, start=1):
            key = str(doc.get("_id", passage_key(doc)))
            entry = fused.setdefault(key, {"doc": doc, "score": 0.0, "ranks": []})
            entry["score"] += 1 / (60 + rank)
            entry["ranks"].append({"iteration": round_idx, "rank": rank})
    return fused


def run_agent(req: AgentRequest, *, runs, sessions, collection) -> Iterator[dict[str, Any]]:
    settings = {"interaction_mode": "Automatica", "top_k": req.top_k, "top_n": req.top_n,
                "retries": req.retries, "filters": req.filters or {},
                "rerank_instructions": req.rerank_instructions, "skip_ambiguity": req.skip_ambiguity}
    run_id = agentic_memory.start_run(
        runs, sessions, req.session_id, req.question, parent_run_id=req.parent_run_id,
        feedback_id=req.feedback_id, settings=settings, feedback_context=req.feedback_context)
    yield {"type": "start", "run_id": run_id, "question": req.question, "max_searches": req.retries + 1}

    model = agentic_search.model_label()

    def record(event: dict[str, Any], *, step: str, iteration: int = 0, candidate_count: int = 0) -> dict[str, Any]:
        event = {**event, "model": model}
        agentic_memory.add_event(runs, sessions, req.session_id, run_id, event,
                                 state={"step": step, "iteration": iteration, "candidate_count": candidate_count})
        return event

    def finish(answer: str, sources: list[dict], status: str) -> dict[str, Any]:
        agentic_memory.finish_run(runs, sessions, session_id=req.session_id, run_id=run_id,
                                  question=req.question, answer=answer, sources=sources,
                                  feedback_context=req.feedback_context, parent_run_id=req.parent_run_id,
                                  status=status)
        return {"type": "done", "run_id": run_id, "status": status}

    try:
        scope = search_scope.require_scope(req.embed_mode)
        modes = scope["modes"]

        # 1. Interpretazione: solo se c'è una conversazione da cui risolvere riferimenti.
        started = time.perf_counter()
        if (req.planning_memory or req.planning_summaries or req.feedback_context) and not req.is_clarification:
            plan = agentic_search.plan_search(req.question, req.planning_memory, req.planning_summaries)
            plan["skipped"] = False
        else:
            plan = {"search_query": req.question, "comment": "", "error": None, "skipped": True}
        plan_ms = (time.perf_counter() - started) * 1000
        initial_query = agentic_search.compose_search_query(
            plan["search_query"], "" if req.is_clarification else req.feedback_context)
        yield record({
            "type": "plan", "search_query": initial_query, "planned_query": plan["search_query"],
            "planner_skipped": plan["skipped"], "duration_ms": round(plan_ms), "comment": plan["comment"],
            "memory_messages_used": len(req.planning_memory),
            "memory": {"collection": "agentic_sessions", "session_id": req.session_id,
                       "read_ms": round(req.memory_read_ms, 1),
                       "turns": len(req.planning_summaries),
                       "previous_questions": [t.get("question") for t in req.planning_summaries[-3:]]},
            "filters": req.filters or {}, "modes": modes, "settings": settings,
            "feedback_context": req.feedback_context, "parent_run_id": req.parent_run_id,
            "database_search": {
                "database": scope["database"], "collection": scope["collection"],
                "text_index": config.SEARCH_INDEX_NAME, "text_path": "content",
                "vector_indexes": [{"mode": m, "index": config.vector_index_name(m)} for m in modes],
            },
            "max_searches": req.retries + 1, "planner_error": plan.get("error"),
        }, step="retrieval")

        rounds: list[list[dict]] = []
        executed: list[str] = []
        union: dict[str, dict] = {}
        search_query = initial_query
        seen = {req.question.casefold(), plan["search_query"].casefold(), initial_query.casefold()}
        stop_reason = "Ho usato tutte le ricerche configurate."
        final_query = initial_query
        pool: list[dict] = []
        fused: dict[str, dict] = {}
        not_answerable: str | None = None
        ambiguity = None
        threshold = config.RERANK_RELEVANCE_THRESHOLD

        for iteration in range(req.retries + 1):
            # 2. Ricerca ibrida + reranking in Atlas.
            started = time.perf_counter()
            candidates, ranked = agentic_search.retrieve_candidates(
                collection, search_query, filters=req.filters, modes=modes,
                top_k=req.top_k, instructions=req.rerank_instructions)
            retrieval_ms = (time.perf_counter() - started) * 1000
            rounds.append(ranked)
            executed.append(search_query)
            for doc in candidates:
                union.setdefault(str(doc.get("_id", passage_key(doc))), doc)
            refs = [{"rank": r, "filename": d.get("filename"), "page": d.get("page_number"),
                     "chunk_id": str(d.get("_id", "")), "score": d.get("_rerank_score")}
                    for r, d in enumerate(ranked[:10], start=1)]
            yield record({"type": "retrieval", "iteration": iteration + 1, "search_query": search_query,
                          "candidate_count": len(candidates), "ranked_sources": refs, "search_modes": modes,
                          "duration_ms": round(retrieval_ms), "focused": iteration > 0},
                         step="grading", iteration=iteration + 1, candidate_count=len(candidates))

            # Stessa soglia fuori tema della Query, sulla prima ricerca.
            if iteration == 0:
                top = ranked[0].get("_rerank_score") if ranked else None
                if not ranked or (isinstance(top, (int, float)) and top < threshold):
                    not_answerable = "no_results" if not ranked else "off_topic"
                    stop_reason = STOP_REASONS[not_answerable]
                    if not_answerable == "off_topic":
                        stop_reason += f" (miglior passaggio {top:.3f} < soglia {threshold:.2f})"
                    yield record({"type": "decision", "iteration": 1, "reason": not_answerable, "action": "stop",
                                  "top_rerank": top, "threshold": threshold}, step="decision", iteration=1)
                    break

            # Insieme cumulativo riordinato rispetto alla domanda completa.
            fused = _fuse(rounds)
            if len(rounds) == 1:
                pool = list(ranked)
            else:
                ordered = sorted(fused.values(), key=lambda item: item["score"], reverse=True)
                started = time.perf_counter()
                pool = agentic_search.rerank_union(
                    collection, final_query, [item["doc"] for item in ordered[:config.AGENT_UNION_RERANK]],
                    top_n=max(req.top_n, 8), instructions=req.rerank_instructions)
                yield record({"type": "agent_rank_fusion", "unique_candidates": len(fused),
                              "reranked": min(len(ordered), config.AGENT_UNION_RERANK),
                              "duration_ms": round((time.perf_counter() - started) * 1000),
                              "comment": "Passaggi di tutte le ricerche riuniti e riordinati sulla domanda."},
                             step="grading", iteration=iteration + 1, candidate_count=len(fused))

            # 3. Valutazione delle prove, elemento per elemento.
            grade_question = (req.question if plan.get("skipped") or req.is_clarification
                              else f"{req.question} (riferita a: {plan['search_query']})")
            started = time.perf_counter()
            grade = agentic_search.assess_evidence(grade_question, search_query, pool, executed)
            grade_ms = (time.perf_counter() - started) * 1000
            pool_refs = [{"rank": r, "filename": d.get("filename"), "page": d.get("page_number"),
                          "chunk_id": str(d.get("_id", "")),
                          "excerpt": agentic_search.compact_text(d.get("content", ""))[:3000]}
                         for r, d in enumerate(pool[:8], start=1)]
            yield record({
                "type": "grading", "iteration": iteration + 1, "cumulative": len(rounds) > 1,
                "duration_ms": round(grade_ms), "elements": grade.get("elements", []),
                "unverified_elements": grade.get("unverified_elements", []),
                "evaluated": grade.get("evaluated", True), "sufficient": grade["sufficient"],
                "comment": grade.get("comment", ""), "followup_query": grade.get("followup_query"),
                "evaluated_sources": len(pool[:8]), "evaluated_source_refs": pool_refs,
                "missing_elements": grade.get("missing_elements", []),
                "supporting_quotes": grade.get("supporting_quotes", []),
                "failure_reason": grade.get("failure_reason"), "failure_detail": grade.get("failure_detail"),
            }, step="decision", iteration=iteration + 1, candidate_count=len(union))

            # Valori diversi in anni diversi e domanda senza anno: si ferma e chiede.
            ambiguity = None if req.skip_ambiguity else agentic_search.find_year_ambiguity(
                grade, req.question, plan["search_query"], req.feedback_context)
            if ambiguity:
                not_answerable = "needs_clarification"
                stop_reason = "Valori diversi in anni diversi: serve sapere quale anno."
                yield record({"type": "clarification", "iteration": iteration + 1, "reason": "year", **ambiguity},
                             step="decision", iteration=iteration + 1)
                yield record({"type": "decision", "iteration": iteration + 1, "reason": "needs_clarification",
                              "action": "stop"}, step="decision", iteration=iteration + 1)
                break

            decision = agentic_search.decide_next_search(grade, iteration=iteration, retries=req.retries,
                                                         seen_queries=seen)
            yield record({"type": "decision", "iteration": iteration + 1, "reason": decision["reason"],
                          "action": decision["action"], "search_query": decision.get("query")},
                         step="rewrite" if decision["action"] == "continue" else "decision", iteration=iteration + 1)
            if decision["action"] == "stop":
                stop_reason = STOP_REASONS[decision["reason"]]
                break
            # Ricerca mirata al solo elemento mancante.
            search_query = decision["query"]
            seen.add(search_query.casefold())
            yield record({"type": "rewrite", "iteration": iteration + 1, "search_query": search_query,
                          "reason": grade.get("comment", "")}, step="rewrite", iteration=iteration + 1)

        final_sources = pool[:req.top_n]

        if not_answerable == "needs_clarification":
            years = ", ".join(str(o["year"]) for o in ambiguity["options"])
            answer = (f"Ho trovato valori diversi in anni diversi ({years}) per: "
                      + "; ".join(ambiguity["elements"]) + ". Quale anno ti interessa?")
            yield record({"type": "answer", "status": "needs_input", "answer_preview": answer, "sources": [],
                          "attempts": 0, "stop_reason": stop_reason}, step="complete")
            yield finish(answer, [], "needs_input")
            return

        if not_answerable or not final_sources:
            block = not_answerable or "no_results"
            answer = "Risposta non generata. " + stop_reason
            yield record({"type": "answer", "status": "blocked", "answer_preview": answer, "sources": [],
                          "attempts": 0, "stop_reason": stop_reason,
                          "diagnostics": {"block_reason": block, "attempts": [], "retry_error_type": None}},
                         step="complete")
            yield finish(answer, [], "blocked")
            return

        # 4. Risposta, con gli stessi controlli della Query.
        if not generation.llm_available():
            raise RuntimeError("Il provider LLM configurato non è disponibile.")
        generation_question = req.question
        if plan["search_query"].casefold() != req.question.casefold():
            generation_question = (
                f"Domanda dell'utente: {req.question}\n\n"
                f"Interpretazione autonoma della domanda, ricavata anche dalla memoria recente: {plan['search_query']}\n\n"
                "Usa questa interpretazione solo per risolvere i riferimenti conversazionali. "
                "Basa i fatti della risposta esclusivamente sui passaggi recuperati.")
        if req.feedback_context:
            generation_question += (
                "\n\nIndicazioni dell'utente (non sono evidenze): "
                f"{req.feedback_context}\nUsa queste indicazioni per capire la richiesta; "
                "verifica i fatti esclusivamente nei passaggi recuperati.")
        yield {"type": "generating"}
        started = time.perf_counter()
        reviewed = answer_quality.generate_reviewed_answer(generation_question, final_sources)
        answer_ms = (time.perf_counter() - started) * 1000
        answer = reviewed["answer"].get("answer", "")
        diagnostics = answer_quality.answer_diagnostics(reviewed)
        blocked = bool(reviewed.get("blocked"))
        answer_check = (((reviewed.get("attempts") or [{}])[-1].get("check") or {}).get("status", "cited")
                        if not blocked else diagnostics.get("block_reason"))
        if blocked:
            fallback = answer_quality.best_publishable_attempt(reviewed)
            if fallback:
                answer, answer_check = fallback
                blocked = False
            else:
                last = (diagnostics["attempts"][-1].get("check") or {}) if diagnostics["attempts"] else {}
                answer = ("Risposta bloccata. " + answer_quality.explain_answer_check(diagnostics.get("block_reason"), last)
                          + " Consulta i passaggi recuperati.")
        sources = [{"filename": d.get("filename"), "page": d.get("page_number"), "chunk_id": str(d.get("_id", "")),
                    "content": (d.get("content") or "")[:8000],
                    "gridfs_file_id": str(d.get("gridfs_file_id")) if d.get("gridfs_file_id") else None}
                   for d in final_sources]
        status = "blocked" if blocked else "complete"
        yield record({
            "type": "answer", "status": "blocked" if blocked else "generated", "answer_check": answer_check,
            "answer_preview": answer[:700], "sources": sources, "attempts": len(reviewed.get("attempts", [])),
            "stop_reason": stop_reason, "diagnostics": diagnostics, "duration_ms": round(answer_ms),
            "citation_check": [] if blocked else check_citations(answer, final_sources),
        }, step="complete")
        yield finish(answer, sources, status)
    except GeneratorExit:
        # La pagina si è chiusa a metà ricerca: senza questo la ricerca resterebbe "in corso" per sempre.
        agentic_memory.fail_run(runs, sessions, session_id=req.session_id, run_id=run_id,
                                message="Ricerca interrotta: la pagina è stata chiusa o ricaricata.")
        raise
    except Exception as exc:  # lo stato e l'errore restano in MongoDB
        agentic_memory.fail_run(runs, sessions, session_id=req.session_id, run_id=run_id,
                                message=f"{type(exc).__name__}: {exc}")
        yield {"type": "error", "run_id": run_id, "message": f"{type(exc).__name__}: {exc}"}
        yield {"type": "done", "run_id": run_id, "status": "failed"}
