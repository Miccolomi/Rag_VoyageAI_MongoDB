"""Riquadro «Cosa ha fatto l'agente in più»: ricerca standard e agente a confronto.

Tutto viene dagli eventi salvati della ricerca agentica. La prima ricerca dell'agente
usa la stessa pipeline della pagina Query, quindi lo stato dopo la prima ricerca è
quello che la ricerca standard avrebbe passato al modello: il confronto è onesto e
non richiede ricerche in più.
"""
from __future__ import annotations

import html
import uuid
from typing import Any


def _fmt_ms(ms: float | None) -> str:
    if not ms:
        return ""
    return f"{ms:.0f} ms" if ms < 1000 else f"{ms / 1000:.1f} s".replace(".", ",")


def _short(text: str, limit: int = 90) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _dati(n: int, total: int) -> str:
    return f"{n} {'dato' if n == 1 else 'dati'} su {total}"


def summarize_run(run: dict[str, Any], parent: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Scenario, colonne e passi del confronto. None se non c'è nulla da mostrare.

    `parent` è la ricerca precedente quando questa risponde a una scelta (l'anno):
    il valore va raccontato qui, con la risposta definitiva, non sulla domanda di scelta.
    """
    events = run.get("events") or []
    parent_events = (parent or {}).get("events") or []
    asked = next((e for e in parent_events if e.get("type") == "clarification"), None)
    plan = next((e for e in events if e.get("type") == "plan"), {})
    searches = [e for e in events if e.get("type") == "retrieval"]
    gradings = [e for e in events if e.get("type") == "grading" and e.get("evaluated", True)]
    clarification = next((e for e in events if e.get("type") == "clarification"), None)
    decision = next((e for e in reversed(events) if e.get("type") == "decision" and e.get("action") == "stop"), {})
    answer = next((e for e in reversed(events) if e.get("type") == "answer"), {})
    if not searches:
        return None
    question = run.get("question") or ""
    first = gradings[0] if gradings else {}
    last = gradings[-1] if gradings else {}
    first_found = {e.get("element") for e in first.get("elements") or [] if e.get("found")}
    elements = [e.get("element") for e in (last.get("elements") or first.get("elements") or [])]
    last_found = {e.get("element") for e in last.get("elements") or [] if e.get("found")}
    planned = plan.get("planned_query") or ""
    understood = bool(plan) and not plan.get("planner_skipped") and planned.strip().casefold() != question.strip().casefold()
    memory = plan.get("memory") or {}
    chosen = None
    if asked:
        import re
        match = re.search(r"anno (\d{4})", f"{plan.get('feedback_context') or ''} {question}")
        chosen = match.group(1) if match else None
    absent = decision.get("reason") in {"off_topic", "no_results"} or (elements and not last_found and not first_found)

    # Scenario principale: cosa l'agente ha fatto che la ricerca standard non fa.
    if clarification:
        years = [str(o.get("year")) for o in clarification.get("options") or []]
        scenario = ("Ha chiesto quale anno usare",
                    "Risponderebbe con un anno solo, senza dire che negli altri anni i valori cambiano",
                    f"Ha trovato valori diversi in {len(years)} anni ({', '.join(years)}) e ti fa scegliere")
    elif asked:
        years = [str(o.get("year")) for o in asked.get("options") or []]
        scenario = ("Ti ha chiesto l’anno, poi ha verificato",
                    "Avrebbe risposto con l’anno che il reranker mette in cima, senza dire che negli altri anni i valori cambiano",
                    (f"Ha visto valori diversi in {len(years)} anni ({', '.join(years)}), ti ha fatto scegliere"
                     + (f" e ha risposto con i dati del {chosen}" if chosen else " e ha confrontato gli anni")
                     + f": {_dati(len(last_found), len(elements) or len(last_found))} verificati"))
    elif understood and not absent:
        turns = memory.get("turns") or 0
        scenario = ("Ha ricordato la conversazione, salvata in MongoDB",
                    f"Cercherebbe «{_short(question, 40)}» alla lettera: senza memoria non sa di cosa stai parlando",
                    f"Ha letto da MongoDB le {turns} domande precedenti e ha cercato «{_short(planned, 70)}»"
                    if turns else f"Ha usato la conversazione: «{_short(planned, 70)}»")
    elif absent:
        scenario = ("Non ha risposto a memoria",
                    "Chiamerebbe comunque il modello sui passaggi trovati",
                    "Ha controllato le fonti: i dati richiesti non ci sono e lo dice")
    elif len(searches) > 1 and len(last_found) > len(first_found):
        scenario = ("Ha cercato il dato mancante",
                    f"Risponderebbe con {_dati(len(first_found), len(elements))}",
                    f"Ha fatto una ricerca mirata ed è arrivato a {_dati(len(last_found), len(elements))}")
    else:
        scenario = ("Ha verificato prima di rispondere",
                    "Risponderebbe senza controllare se le fonti bastano",
                    f"Ha controllato che le fonti coprano {_dati(len(last_found), len(elements) or len(last_found))}"
                    if elements else "Ha controllato che le fonti bastino")

    steps: list[dict[str, str]] = []
    if asked:
        steps.append({"icon": "❓", "text": "Valori diversi per anno: ti ha chiesto quale"
                      + (f" · hai scelto il {chosen}" if chosen else " · hai chiesto il confronto"), "ms": ""})
    if understood and memory.get("turns"):
        steps.append({"icon": "🍃", "text": f"Legge la sessione da MongoDB: {memory['turns']} domande precedenti",
                      "ms": _fmt_ms(memory.get("read_ms")) if memory.get("read_ms", 0) >= 1 else "< 1 ms"})
    if understood:
        steps.append({"icon": "🧭", "text": f"Interpreta «{_short(question, 30)}» → «{_short(planned, 60)}»",
                      "ms": _fmt_ms(plan.get("duration_ms"))})
    grade_by_iteration = {g.get("iteration"): g for g in gradings}
    for search in searches:
        it = search.get("iteration", 1)
        label = "Ricerca nei documenti" if it == 1 else f"Ricerca mirata: «{_short(search.get('search_query'), 50)}»"
        steps.append({"icon": "🔍", "text": label, "ms": _fmt_ms(search.get("duration_ms"))})
        grade = grade_by_iteration.get(it)
        if grade:
            found = [e for e in grade.get("elements") or [] if e.get("found")]
            total = len(grade.get("elements") or [])
            text = (f"Controlla le prove: {_dati(len(found), total)} trovati" if total
                    else ("Controlla le prove: sufficienti" if grade.get("sufficient") else "Controlla le prove: incomplete"))
            steps.append({"icon": "✅" if grade.get("sufficient") else "⚠️", "text": text,
                          "ms": _fmt_ms(grade.get("duration_ms"))})
    if clarification:
        steps.append({"icon": "❓", "text": "Valori diversi per anno: si ferma e ti chiede quale", "ms": ""})
    elif decision.get("reason") == "off_topic":
        steps.append({"icon": "🚫", "text": "Pertinenza sotto la soglia: nessuna risposta inventata", "ms": ""})
    elif answer and answer.get("status") != "needs_input":
        steps.append({"icon": "💬", "text": "Scrive la risposta citando le fonti", "ms": _fmt_ms(answer.get("duration_ms"))})

    total_ms = sum(e.get("duration_ms") or 0 for e in events)
    model_calls = (0 if plan.get("planner_skipped") or not plan else 1) + len(gradings) + (answer.get("attempts") or 0)
    return {
        "title": scenario[0], "standard": scenario[1], "agent": scenario[2],
        "elements": elements, "standard_found": first_found, "agent_found": last_found,
        # Con la memoria la ricerca standard cercherebbe un'altra cosa: i suoi dati trovati non sono confrontabili.
        "standard_chips": not (understood and not absent and not clarification and not asked),
        "steps": steps,
        "memory": ({"collection": memory.get("collection", "agentic_sessions"), "session_id": memory.get("session_id"),
                    "previous_questions": memory.get("previous_questions") or [], "question": question,
                    "planned": planned, "read_ms": memory.get("read_ms")} if understood and not absent else None),
        "cost": f"{len(searches)} {'ricerca' if len(searches) == 1 else 'ricerche'} · "
                f"{model_calls} {'chiamata' if model_calls == 1 else 'chiamate'} al modello" + (f" · {_fmt_ms(total_ms)}" if total_ms else ""),
    }


def agent_value_html(run: dict[str, Any], *, dark: bool = False) -> tuple[str, int] | None:
    data = summarize_run(run)
    if not data:
        return None
    uid = "av" + uuid.uuid4().hex[:8]
    fg, muted, border, card = (("#fafafa", "#a3a8b8", "rgba(250,250,250,0.18)", "rgba(255,255,255,0.04)") if dark
                               else ("#31333F", "#6b6f7b", "rgba(49,51,63,0.16)", "#ffffff"))
    ok_bg, ok_fg = ("rgba(29,158,117,0.22)", "#9FE1CB") if dark else ("rgba(29,158,117,0.14)", "#0F6E56")
    ko_bg, ko_fg = ("rgba(239,159,39,0.22)", "#FAC775") if dark else ("rgba(239,159,39,0.16)", "#854F0B")
    accent = "#7F77DD"

    def chips(found: set, animated: bool) -> str:
        out = []
        for i, name in enumerate(data["elements"]):
            on = name in found
            style = f"background:{ok_bg};color:{ok_fg}" if on else f"background:{ko_bg};color:{ko_fg}"
            mark = "✓" if on else "✗"
            cls = f' class="{uid}-chip" style="animation-delay:{0.6 + 0.9 * len(data["steps"]) + 0.2 * i:.1f}s;{style}"' if animated \
                else f' style="{style}"'
            out.append(f'<span{cls}>{mark} {html.escape(_short(name, 46))}</span>')
        return "".join(out)

    steps = "".join(
        f'<div class="{uid}-step" style="animation-delay:{0.4 + 0.9 * i:.1f}s">'
        f'<span>{s["icon"]}</span><span>{html.escape(s["text"])}</span>'
        f'<span style="margin-left:auto;color:{muted};white-space:nowrap">{html.escape(s["ms"])}</span></div>'
        for i, s in enumerate(data["steps"]))
    final_delay = 0.4 + 0.9 * len(data["steps"])
    page = f"""
<div id="{uid}" style="font-family:'Source Sans Pro',sans-serif;color:{fg};font-size:14px">
<style>
#{uid} .{uid}-step{{display:flex;gap:8px;align-items:baseline;padding:4px 0;opacity:0;animation:{uid}in .5s ease forwards}}
#{uid} .{uid}-chip,#{uid} span.{uid}-static{{display:inline-block;margin:3px 6px 0 0;padding:3px 10px;border-radius:6px;font-size:12.5px}}
#{uid} .{uid}-chip{{opacity:0;animation:{uid}in .5s ease forwards}}
#{uid} .{uid}-final{{opacity:0;animation:{uid}in .6s ease forwards;animation-delay:{final_delay:.1f}s}}
@keyframes {uid}in{{from{{opacity:0;transform:translateY(4px)}}to{{opacity:1;transform:none}}}}
#{uid} .{uid}-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:12px}}
</style>
<div style="display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:10px">
  <div style="font-size:16px;font-weight:600">✨ Cosa ha fatto l’agente in più: {html.escape(data["title"].lower())}</div>
  <button onclick="var r=document.getElementById('{uid}');var c=r.cloneNode(true);r.parentNode.replaceChild(c,r);"
    style="font-size:12px;border:1px solid {border};background:transparent;color:{fg};border-radius:6px;padding:3px 10px;cursor:pointer">↻ Riproduci</button>
</div>
<div class="{uid}-grid">
  <div style="border:1px solid {border};border-radius:10px;padding:10px 12px;background:{card}">
    <div style="color:{muted};font-size:12.5px;margin-bottom:6px">🔎 Ricerca standard · una ricerca, risponde subito</div>
    <div>{chips(data["standard_found"], False).replace("<span style", f"<span class='{uid}-static' style")}</div>
    <div style="margin-top:8px;font-weight:600">{html.escape(data["standard"])}</div>
  </div>
  <div style="border:2px solid {accent};border-radius:10px;padding:10px 12px;background:{card}">
    <div style="color:{accent};font-size:12.5px;margin-bottom:6px">🧭 Agente · controlla prima di rispondere</div>
    {steps}
    <div>{chips(data["agent_found"], True)}</div>
    <div class="{uid}-final" style="margin-top:8px;font-weight:600">{html.escape(data["agent"])}</div>
  </div>
</div>
<div style="color:{muted};font-size:12.5px;margin-top:8px">{html.escape(data["cost"])}</div>
</div>"""
    height = 150 + 30 * len(data["steps"]) + (30 if data["elements"] else 0) + (24 if len(data["elements"]) > 2 else 0)
    return page, height
