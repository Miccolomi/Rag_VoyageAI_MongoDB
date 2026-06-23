"""Mini frontend Streamlit per la demo RAG.

Layout:
- 📖 How it Works → architettura + valore della pipeline
- 📤 Upload  → ingestion documenti (PDF, DOCX, XLSX, PPTX, TXT, ...)
- 🔎 Query   → ricerca hybrid + rerank + grounded answer
- 🧪 Confronto → (disabilitato) modalità avanzata
- ⚙ Admin   → indici Atlas, metriche, modalità embedding

Run:  streamlit run app.py
"""
from __future__ import annotations

import html
import json
import re
import tempfile
import time
from pathlib import Path
import tiktoken
import streamlit as st
from bson import ObjectId
from gridfs import GridFS

from src import config
from src.generation import generate_answer, generate_answer_stream, llm_available
from src.indexes import ensure_indexes
from src.ingestion import ingest_pdf
from src.rerank import rerank
from src.retrieval import (
    build_native_rerank_pipeline,
    build_pipeline,
    search,
    search_with_native_rerank,
)

_TOKENIZER = tiktoken.get_encoding("cl100k_base")


# =================================================================== setup ====

st.set_page_config(page_title="RAG Demo — MongoDB + Voyage", layout="wide")

# Custom CSS for Premium Design & Fonts
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap');

/* Global Font Override */
.stApp {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
}

/* Custom cards styling */
.custom-card {
    background-color: rgba(255, 255, 255, 0.02);
    border-radius: 12px;
    border: 1px solid rgba(128, 128, 128, 0.15);
    padding: 16px;
    margin-bottom: 12px;
    transition: all 0.25s ease-in-out;
}

.custom-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);
    border-color: rgba(126, 87, 194, 0.4);
}

/* Header Gradient customizer */
.gradient-header {
    background: linear-gradient(135deg, #00684A 0%, #7e57c2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)

if "embed_mode" not in st.session_state:
    st.session_state.embed_mode = config.EMBED_MODE


# ─── HEALTH CHECK ──────────────────────────────────────────────────────────────

_STATUS_EMOJI = {"ok": "🟢", "warning": "🟡", "error": "🔴"}


@st.cache_data(ttl=15, show_spinner=False)
def _health_check(embed_mode: str) -> dict[str, tuple[str, str]]:
    """Stato real-time dei componenti. Cached 15s per non martellare i servizi.

    Ritorna {nome_componente: (status, message)} dove status ∈ {ok, warning, error}.
    """
    from pymongo import MongoClient as _MC
    out: dict[str, tuple[str, str]] = {}

    # 1. MongoDB Atlas — ping con timeout corto
    mongo_ok = False
    try:
        c = _MC(
            config.MONGODB_URI,
            serverSelectionTimeoutMS=3000,
            socketTimeoutMS=3000,
            connectTimeoutMS=3000,
        )
        c.admin.command("ping")
        c.close()
        out["MongoDB Atlas"] = ("ok", f"raggiungibile · DB `{config.MONGODB_DB}`")
        mongo_ok = True
    except Exception as e:
        msg = str(e).splitlines()[0][:80] if str(e) else type(e).__name__
        out["MongoDB Atlas"] = ("error", f"non raggiungibile · {msg}")

    # 2. Atlas Search Indexes — solo se Mongo è up
    if mongo_ok:
        try:
            coll = config.get_collection()
            all_idx = list(coll.list_search_indexes())
            present = {idx["name"]: idx for idx in all_idx}
            missing = [
                n for n in (config.VECTOR_INDEX_NAME, config.SEARCH_INDEX_NAME)
                if n not in present
            ]
            if missing:
                out["Atlas Search Indexes"] = (
                    "warning",
                    f"mancanti: `{', '.join(missing)}` — crea dal tab Admin",
                )
            else:
                statuses = {n: idx.get("status", "?") for n, idx in present.items()}
                non_ready = [f"`{n}`={s}" for n, s in statuses.items()
                             if s.upper() != "READY"]
                if non_ready:
                    out["Atlas Search Indexes"] = (
                        "warning",
                        f"non READY: {', '.join(non_ready)}",
                    )
                else:
                    out["Atlas Search Indexes"] = (
                        "ok",
                        f"READY: `{config.VECTOR_INDEX_NAME}`, "
                        f"`{config.SEARCH_INDEX_NAME}`",
                    )
                # Coerenza schema indice ↔ embed_mode (anche se status READY)
                vidx = present.get(config.VECTOR_INDEX_NAME)
                if vidx:
                    fields = (vidx.get("latestDefinition", {})
                              .get("fields", []))
                    vec_field = next(
                        (f for f in fields
                         if f.get("type") in ("autoEmbed", "vector")),
                        None,
                    )
                    if vec_field:
                        idx_type = vec_field.get("type")
                        idx_path = vec_field.get("path")
                        expected = ("autoEmbed", "content") if embed_mode == "auto" \
                            else ("vector", "embedding")
                        if (idx_type, idx_path) != expected:
                            out["Schema indice vector"] = (
                                "error",
                                f"incoerente: indice ha `type={idx_type}` "
                                f"`path={idx_path}` ma `EMBED_MODE={embed_mode}` "
                                f"richiede `type={expected[0]}` `path={expected[1]}`. "
                                f"Drop & recreate dal tab Admin + re-ingest.",
                            )
                        else:
                            expected_dims = (
                                config.VOYAGE_CONTEXT_DIMENSIONS
                                if embed_mode == "context"
                                else config.VOYAGE_EMBED_DIMENSIONS
                            )
                            idx_dims = vec_field.get("numDimensions")
                            if embed_mode != "auto" and idx_dims != expected_dims:
                                out["Schema indice vector"] = (
                                    "error",
                                    f"dimensione indice `{idx_dims}` ma "
                                    f"`EMBED_MODE={embed_mode}` richiede "
                                    f"`{expected_dims}`. Drop & recreate + re-ingest.",
                                )
                            else:
                                dims_label = (
                                    f" · `{idx_dims}` dimensioni"
                                    if idx_dims else ""
                                )
                                out["Schema indice vector"] = (
                                    "ok",
                                    f"`type={idx_type}` `path={idx_path}`"
                                    f"{dims_label} coerente con "
                                    f"`EMBED_MODE={embed_mode}`",
                                )
        except Exception as e:
            out["Atlas Search Indexes"] = (
                "warning",
                f"errore lettura: {type(e).__name__}",
            )

    # 3. Embed mode — coerenza tra modalità e chiave
    needs_voyage = embed_mode in ("manual", "context")
    if needs_voyage and not config.VOYAGE_API_KEY:
        out["Embed mode"] = (
            "error",
            f"`{embed_mode}` selezionata ma `VOYAGE_API_KEY` mancante",
        )
    else:
        mode_label = {
            "auto":    "autoEmbed (Atlas)",
            "manual":  "manual (app)",
            "context": "context-4 page-aware",
        }.get(embed_mode, embed_mode)
        used_model = (
            config.VOYAGE_CONTEXT_MODEL if embed_mode == "context"
            else config.VOYAGE_EMBED_MODEL
        )
        out["Embed mode"] = ("ok", f"{mode_label} · model `{used_model}`")

    # 4. Voyage API (per rerank + embed manual)
    if config.VOYAGE_API_KEY:
        endpoint_host = config.VOYAGE_API_BASE.replace("https://", "").split("/")[0]
        out["Voyage AI"] = (
            "ok",
            f"chiave configurata · endpoint `{endpoint_host}` · "
            f"rerank model `{config.VOYAGE_RERANK_MODEL}`",
        )
    else:
        out["Voyage AI"] = (
            "warning",
            "`VOYAGE_API_KEY` mancante — il rerank verrà saltato in Query",
        )

    # 5. LLM provider — usa l'helper esistente
    provider = config.LLM_PROVIDER
    model_per_provider = {
        "anthropic": config.ANTHROPIC_MODEL,
        "ollama": config.OLLAMA_MODEL,
        "gemini": config.GEMINI_MODEL,
    }
    if llm_available():
        model = model_per_provider.get(provider, "?")
        out[f"LLM ({provider})"] = ("ok", f"raggiungibile · model `{model}`")
    else:
        reason = {
            "anthropic": "`ANTHROPIC_API_KEY` mancante",
            "gemini": "`GEMINI_API_KEY` mancante",
            "ollama": f"Ollama non raggiungibile su `{config.OLLAMA_BASE_URL}` "
                      f"(avvia `ollama serve`)",
        }.get(provider, f"provider sconosciuto `{provider}`")
        out[f"LLM ({provider})"] = ("error", reason)

    return out


def _overall_status(health: dict[str, tuple[str, str]]) -> str:
    levels = [s for s, _ in health.values()]
    if "error" in levels:
        return "error"
    if "warning" in levels:
        return "warning"
    return "ok"


# ─── TITOLO + STATUS ──────────────────────────────────────────────────────────

_health = _health_check(st.session_state.embed_mode)
_overall = _overall_status(_health)
_overall_label = {
    "ok": "Tutto operativo",
    "warning": "Funziona con limitazioni",
    "error": "Componente critico KO",
}[_overall]

title_col, status_col = st.columns([0.78, 0.22], vertical_alignment="bottom")
with title_col:
    st.title("RAG Demo — MongoDB Atlas + Voyage AI")
with status_col:
    with st.popover(
        f"{_STATUS_EMOJI[_overall]} {_overall_label}",
        use_container_width=True,
    ):
        st.markdown("##### Stato dei componenti monitorati")
        for _name, (_status, _msg) in _health.items():
            st.markdown(f"{_STATUS_EMOJI[_status]} **{_name}** — {_msg}")
        st.divider()
        st.caption("⏱ Stato in cache per 15 secondi. Click per rinfrescare ora.")
        if st.button("🔄 Ricontrolla", key="health_refresh",
                     use_container_width=True):
            _health_check.clear()
            st.rerun()

mode_badge = {
    "auto":    "autoEmbed",
    "manual":  "manual",
    "context": "context-4 page-aware",
}.get(st.session_state.embed_mode, st.session_state.embed_mode)
active_embed_model = (
    config.VOYAGE_CONTEXT_MODEL if st.session_state.embed_mode == "context"
    else config.VOYAGE_EMBED_MODEL
)
llm_label = {
    "anthropic": f"Anthropic `{config.ANTHROPIC_MODEL}`",
    "ollama":    f"Ollama `{config.OLLAMA_MODEL}`",
    "gemini":    f"Gemini `{config.GEMINI_MODEL}`",
}.get(config.LLM_PROVIDER, f"Sconosciuto `{config.LLM_PROVIDER}`")
st.caption(
    f"DB `{config.MONGODB_DB}` · Collection `{config.MONGODB_COLLECTION}` · "
    f"Embed mode **{mode_badge}** · Model `{active_embed_model}` · "
    f"Rerank `{config.VOYAGE_RERANK_MODEL}` · LLM {llm_label}"
)


@st.cache_resource
def _coll():
    return config.get_collection()


@st.cache_resource
def _db():
    return config.get_db()


_DEFAULT_QUERY = (
    "Quali azioni sono previste per il riciclo delle acque piovane e "
    "dove si localizza l'elettrodotto Doganella?"
)

# Scenari pre-cucinati per la demo: dimostrano i 3 casi narrativi chiave
# (text vince / vector vince / hybrid combina). Modificabili a piacere.
DEMO_PRESETS = [
    {
        "label": "Query con codici/identificatori esatti",
        "query": "Norma CEI 11-4",
        "description": "Query contenente un termine tecnico/normativo specifico. "
                       "Caso d'uso tipico: ricerca di codici normativi, P.IVA, "
                       "riferimenti puntuali.",
        "demo_focus": "Dimostra perche' il branch full-text resta fondamentale: codici, norme e identificativi non vanno affidati solo alla semantica.",
        "badges": ["$search", "BM25", "codici esatti"],
    },
    {
        "label": "Query con parafrasi/sinonimi",
        "query": "piattaforma per la misurazione dei consumi elettrici",
        "description": "Query che esprime un concetto senza usare i termini "
                       "esatti del documento (es. il documento dice 'metering', "
                       "la query lo riformula in italiano).",
        "demo_focus": "Dimostra il valore della ricerca vettoriale: recupera concetti anche quando domanda e documento usano parole diverse.",
        "badges": ["$vectorSearch", "sinonimi", "semantic recall"],
    },
    {
        "label": "Query mista (termini tecnici + concetti)",
        "query": "audit e tracciabilità del sistema",
        "description": "Query con mix di parole tecniche presenti nei documenti "
                       "e termini concettuali ampi.",
        "demo_focus": "Dimostra il valore di $rankFusion: MongoDB combina ranking lessicale e semantico in una singola aggregation.",
        "badges": ["$rankFusion", "hybrid", "metadata"],
    },
    {
        "label": "Query off-topic (rifiuto del sistema)",
        "query": "Quali sono le ricette tipiche della cucina napoletana?",
        "description": "Query non coperta dal corpus. Dimostra l'off-topic "
                       "detection: rerank score basso → no chiamata LLM, "
                       "zero hallucination.",
        "demo_focus": "Dimostra il guardrail anti-allucinazione: se il corpus non contiene la risposta, il sistema evita una generazione elegante ma inventata.",
        "badges": ["rerank threshold", "no hallucination", "guardrail"],
    },
]


def _on_preset_change() -> None:
    """Callback: al cambio preset, popola il campo query.

    - Su un preset reale → usa la query del preset.
    - Su "(scrivi una domanda libera)" → pre-popola con la query di esempio
      _DEFAULT_QUERY, così l'utente può vedere subito un caso ben formulato
      che funziona (e modificarla a piacere).
    """
    choice = st.session_state.get("preset_choice", "")
    if choice == "(scrivi una domanda libera)":
        st.session_state.query_simple = _DEFAULT_QUERY
        return
    for p in DEMO_PRESETS:
        if p["label"] == choice:
            st.session_state.query_simple = p["query"]
            return


# ================================================================ helpers ====

_STOPWORDS = {
    # italiano
    "che", "del", "della", "dei", "delle", "una", "uno", "con", "per", "non",
    "nel", "nella", "alla", "allo", "agli", "alle", "sul", "sulla", "sui",
    "lui", "lei", "tra", "fra", "come", "cosa", "ecc", "qui", "qua", "loro",
    "questo", "questa", "quello", "quella", "essere", "stata", "stato",
    # inglese
    "the", "and", "for", "with", "from", "this", "that", "into", "your",
    "have", "has", "are", "was", "were", "but", "not", "you", "all",
}

_CITATION_RE = re.compile(r"#(\d+)")

# Regex per "neutralizzare" caratteri markdown a inizio riga (heading, list,
# blockquote, ecc.) che st.markdown interpreterebbe come sintassi attiva.
# Applicare DOPO html.escape, perché interviene solo all'inizio della riga.
_MD_LINESTART_RE = re.compile(r"^([ \t]*)([#>\-*+_~`|=])", re.MULTILINE)


def _escape_markdown_linestart(text: str) -> str:
    """Antepone backslash ai char markdown all'inizio di ogni riga, per
    evitare che testo del chunk venga renderizzato come heading/lista/quote."""
    return _MD_LINESTART_RE.sub(r"\1\\\2", text)


def _safe_html(text: str) -> str:
    """HTML escape + markdown-linestart escape, SENZA evidenziazione query.

    Rendering pulito del testo del chunk nelle card / snippet — niente
    `<mark>` gialli, niente decorazioni: solo testo neutro safe da iniettare
    in `st.markdown(unsafe_allow_html=True)`.
    """
    return _escape_markdown_linestart(html.escape(text or ""))


def _highlight(text: str, query: str) -> str:
    """Evidenzia i termini della query nel testo (HTML <mark>).

    Applica HTML escape + markdown-linestart escape per evitare che il
    contenuto del chunk venga interpretato come sintassi markdown.
    """
    safe = _escape_markdown_linestart(html.escape(text or ""))
    if not query:
        return safe
    terms = [t for t in re.findall(r"\w{3,}", query.lower()) if t not in _STOPWORDS]
    if not terms:
        return safe
    terms.sort(key=len, reverse=True)
    pattern = re.compile(r"(" + "|".join(re.escape(t) for t in terms) + r")", re.IGNORECASE)
    return pattern.sub(
        r'<mark style="background:#ffd54f;color:#000;padding:1px 4px;'
        r'border-radius:3px;font-weight:600">\1</mark>',
        safe,
    )


def _chunk_source(c: dict) -> str:
    """Da quale branch del rankFusion è arrivato il chunk: both | vector_only | text_only."""
    details = (c.get("scoreDetails") or {}).get("details") or []
    # MongoDB $rankFusion scoreDetails usa "rank" (non "inputPipelineRank")
    has_vec = any(
        d.get("inputPipelineName") == "vector" and d.get("rank")
        for d in details
    )
    has_txt = any(
        d.get("inputPipelineName") == "text" and d.get("rank")
        for d in details
    )
    if has_vec and has_txt:
        return "both"
    if has_vec:
        return "vector_only"
    if has_txt:
        return "text_only"
    return "unknown"


_SOURCE_STYLE: dict[str, dict[str, str]] = {
    "both":        {"label": "🔀 entrambe",      "bg": "#7e57c2"},
    "vector_only": {"label": "🧠 solo semantica", "bg": "#1e88e5"},
    "text_only":   {"label": "🔤 solo lessicale", "bg": "#fb8c00"},
    "unknown":     {"label": "—",                  "bg": "#757575"},
}


def _source_badge_html(source: str) -> str:
    s = _SOURCE_STYLE[source]
    return (
        f'<span style="background:{s["bg"]};color:white;padding:2px 10px;'
        f'border-radius:12px;font-weight:600;font-size:0.85em">{s["label"]}</span>'
    )


def _demo_badges_html(labels: list[str]) -> str:
    return " ".join(
        f'<span style="display:inline-block;background:rgba(0,237,100,0.12);'
        f'border:1px solid rgba(0,237,100,0.35);color:inherit;'
        f'padding:3px 9px;border-radius:999px;font-size:0.82em;'
        f'font-weight:650;margin:2px 4px 2px 0">{html.escape(label)}</span>'
        for label in labels
    )


def _filter_summary(filters: dict[str, Any] | None) -> str:
    if not filters:
        return "nessun filtro metadata"
    parts = []
    for key, value in filters.items():
        if isinstance(value, dict) and "$in" in value:
            val = ", ".join(str(v) for v in value["$in"])
            parts.append(f"{key} in [{val}]")
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts)


def _timing_summary(timings: dict[str, float]) -> str:
    if not timings:
        return "n/d"
    return " · ".join(f"{k}: {v:.0f} ms" for k, v in timings.items())


def _source_counts(chunks: list[dict]) -> dict[str, int]:
    counts = {"both": 0, "vector_only": 0, "text_only": 0, "unknown": 0}
    for c in chunks:
        counts[_chunk_source(c)] += 1
    return counts


def _result_key(c: dict) -> tuple:
    """Identità stabile di un chunk tra ranking keyword, vector e hybrid."""
    return (c.get("filename"), c.get("page_number"), c.get("chunk_index"))


def _render_trust_panel(
    *,
    raw_results: list[dict],
    reranked: list[dict],
    citations: list[int],
    filters: dict[str, Any] | None,
    top_k: int,
    timings: dict[str, float],
    top_rerank: float | None,
    threshold: float,
    provider: str,
    model: str,
    usage: dict,
) -> None:
    cited_files = {
        (reranked[n - 1].get("filename") or "?")
        for n in citations
        if 1 <= n <= len(reranked)
    }
    top_s = f"{top_rerank:.3f}" if isinstance(top_rerank, (int, float)) else "n/d"
    st.markdown("### Trust panel")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Top rerank", top_s, f"soglia {threshold:.2f}")
    c2.metric("Passaggi citati", str(len(citations)), f"{len(cited_files)} file")
    c3.metric("Risultati fusi", str(len(raw_results)), f"{len(reranked)} passati al LLM")
    c4.metric("LLM", provider, model)
    st.markdown(
        f'<div style="font-size:0.86em;opacity:0.78;margin:6px 0 12px 0">'
        f'Top-K impostato: <b>{top_k}</b> · '
        f'Filtri: <b>{html.escape(_filter_summary(filters))}</b>. '
        f'La composizione della fusione è spiegata nel passaggio '
        f'<b>3. Fusione Ibrida (RRF)</b>.</div>',
        unsafe_allow_html=True,
    )
    if usage:
        st.caption(
            f"Token LLM: input `{usage.get('input_tokens')}` · "
            f"output `{usage.get('output_tokens')}`"
        )


def _build_pipeline_receipt(
    *,
    query: str,
    filters: dict[str, Any] | None,
    top_k: int,
    top_n: int,
    fuzzy: bool,
    timings: dict[str, float],
    top_rerank: float | None,
    threshold: float,
    rerank_mode: str,
    provider: str,
    model: str,
    citations: list[int],
    reranked: list[dict],
) -> str:
    cited_sources = []
    for n in citations:
        if 1 <= n <= len(reranked):
            c = reranked[n - 1]
            page = c.get("page_number")
            page_s = f" p.{page}" if page not in (None, "") else ""
            cited_sources.append(f"[#{n}] {c.get('filename', '?')}{page_s}")
    top_s = f"{top_rerank:.3f}" if isinstance(top_rerank, (int, float)) else "n/d"
    lines = [
        "Demo summary - MongoDB Atlas RAG",
        f"Query: {query}",
        f"Filtri metadata: {_filter_summary(filters)}",
        f"Retrieval: MongoDB Atlas hybrid search con $vectorSearch + $search + $rankFusion",
        f"Parametri: top_k={top_k}, top_n_llm={top_n}, fuzzy_text={fuzzy}",
        f"Rerank: Voyage {config.VOYAGE_RERANK_MODEL}, "
        f"top_score={top_s}, soglia_off_topic={threshold:.2f}",
        f"LLM: {provider} / {model}",
        f"Timing: {_timing_summary(timings)}",
        f"Fonti citate: {', '.join(cited_sources) if cited_sources else 'nessuna'}",
        "Valore demo: una sola piattaforma per chunk, metadati, full-text search, vector search, ranking e fonti verificabili.",
    ]
    return "\n".join(lines)


def _looks_incomplete_answer(text: str) -> bool:
    if not text or len(text.strip()) < 40:
        return True
    stripped = text.strip()
    if stripped.endswith(("...", "…")):
        return True
    if stripped[-1] not in ".!?)]»\"'":
        return True
    # Caso tipico osservato con Gemini streaming: frase aperta o citazione aperta.
    if stripped.count('"') % 2 == 1:
        return True
    return False


def _needs_generation_repair(text: str, top_rerank: float | None) -> bool:
    if _looks_incomplete_answer(text):
        return True
    if isinstance(top_rerank, (int, float)) and top_rerank >= 0.35:
        return not _parse_citations(text)
    return False


def _extractive_fallback_answer(reranked: list[dict], *, max_sources: int = 2) -> str:
    """Risposta deterministica se il LLM produce testo incompleto/non citato."""
    if not reranked:
        return "Non ho informazioni sufficienti nei passaggi forniti per rispondere."
    lines = ["Ho trovato queste informazioni nei passaggi più rilevanti:"]
    for i, c in enumerate(reranked[:max_sources], start=1):
        content = re.sub(r"\s+", " ", (c.get("content") or "")).strip()
        if not content:
            continue
        snippet = content[:420].rstrip()
        if len(content) > 420:
            snippet += "..."
        lines.append(f"- {snippet} [#{i}]")
    return "\n".join(lines)


def _page_label(page, *, prefix: str = "p.", missing: str = "🤖 chunk auto") -> str:
    """Etichetta pagina per UI con fallback per documenti senza paginazione."""
    if page is None or page == "":
        return missing
    return f"{prefix}{page}"


def _render_chunk(i: int, c: dict, query: str, source: str | None = None) -> None:
    rerank_score = c.get("_rerank_score")
    rerank_s = f"{rerank_score:.3f}" if isinstance(rerank_score, (int, float)) else "—"
    badge = _source_badge_html(source) if source else ""
    st.markdown(
        f"{badge}  **[#{i}]** `{c.get('filename')}` · {_page_label(c.get('page_number'))} · "
        f"🎯 rerank=`{rerank_s}`",
        unsafe_allow_html=True,
    )
    snippet = (c.get("content") or "").strip()[:2000]
    with st.container(border=True):
        if not snippet:
            st.caption("_(chunk vuoto)_")
        elif len(snippet) < 40:
            st.markdown(
                f'<div style="font-size:1.1em;padding:4px 0">{_safe_html(snippet)}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div style="white-space:pre-wrap;line-height:1.55;font-size:0.92em">'
                f'{_safe_html(snippet)}</div>',
                unsafe_allow_html=True,
            )


@st.cache_data(show_spinner=False)
def _gridfs_pdf_bytes(file_id_str: str) -> bytes | None:
    """Recupera il PDF originale da GridFS data la stringa dell'ObjectId. Cached."""
    try:
        fs = GridFS(config.get_db(), collection=config.MONGODB_GRIDFS_BUCKET)
        return fs.get(ObjectId(file_id_str)).read()
    except Exception:
        return None


def _render_cited_sources_downloads(
    reranked: list[dict],
    answer_text: str,
) -> None:
    """Mostra un download button per ogni FILE PDF citato (dedupe per file,
    non per chunk: se il LLM cita 3 chunk dello stesso PDF, mostra 1 solo
    bottone). Le citazioni `[#N]` accumulano sotto a quel bottone.
    """
    citations = _parse_citations(answer_text)
    if not citations:
        return

    # Dedupe per FILE (gridfs_file_id se presente, altrimenti filename)
    # Per ogni file unico, raccolgo tutti i numeri di citazione associati.
    file_groups: dict[str, dict] = {}
    for cit_n in citations:
        if not (1 <= cit_n <= len(reranked)):
            continue
        c = reranked[cit_n - 1]
        gridfs_id = c.get("gridfs_file_id")
        fname = c.get("filename") or "documento.pdf"
        # Chiave: gridfs_id se c'è, altrimenti filename (fallback)
        file_key = str(gridfs_id) if gridfs_id else f"file::{fname}"
        if file_key not in file_groups:
            file_groups[file_key] = {
                "filename": fname,
                "gridfs_id": gridfs_id,
                "citations": [],
            }
        file_groups[file_key]["citations"].append(cit_n)

    if not file_groups:
        return

    st.markdown("##### Scarica le fonti citate")
    n_files = len(file_groups)
    plural = "file unici" if n_files > 1 else "file unico"
    st.caption(
        f"**{n_files} {plural}** referenziati nella risposta. "
        f"I link scaricabili sono qui sotto."
    )
    # 2 colonne per compattezza
    items = list(file_groups.values())
    for i in range(0, len(items), 2):
        row = items[i:i + 2]
        cols = st.columns(2)
        for col, info in zip(cols, row):
            with col:
                fname = info["filename"]
                gridfs_id = info["gridfs_id"]
                cit_list = sorted(set(info["citations"]))
                cit_label = ", ".join(f"[#{n}]" for n in cit_list)
                if not gridfs_id:
                    st.caption(f"📄 {fname} {cit_label} — non in GridFS")
                    continue
                pdf_bytes = _gridfs_pdf_bytes(str(gridfs_id))
                if pdf_bytes is None:
                    st.caption(f"📄 {fname} {cit_label} — errore caricamento")
                    continue
                size_kb = len(pdf_bytes) // 1024
                st.download_button(
                    label=f"📄 **{fname}**  ·  {cit_label}  ·  {size_kb} KB",
                    data=pdf_bytes,
                    file_name=fname,
                    mime="application/pdf",
                    key=f"dl_file_{gridfs_id}",
                    width="stretch",
                )



def _parse_citations(text: str) -> list[int]:
    if not text:
        return []
    return sorted({int(n) for n in _CITATION_RE.findall(text)})


# Regex per normalizzare link Markdown del tipo [label](#N) in citazioni compatte
# [#N]. Il download del PDF vive nella sezione dedicata sotto la risposta.
_MD_CITATION_RE = re.compile(r"\s*\[[^\]]+\]\(\s*#[^)]*\)")


def _normalize_inline_citations(text: str) -> str:
    """Converte citazioni verbose del LLM in marker compatti [#N]."""
    if not text:
        return ""
    cleaned = re.sub(
        r"\s*\[[^\]]+\]\(\s*#(\d+)\s*\)",
        lambda m: f" [#{m.group(1)}]",
        text,
    )
    # Evita ripetizioni immediate dello stesso marker: "[#8]. ... [#8]."
    cleaned = re.sub(r"(\[#(\d+)\])(\s+\1)+", r"\1", cleaned)
    cleaned = re.sub(r" {2,}", " ", cleaned)
    cleaned = re.sub(r" ([.,;:!?])", r"\1", cleaned)
    return cleaned.strip()


def _display_answer_text(text: str) -> str:
    """Rimuove citazioni e link dalla risposta visibile.

    Il testo originale resta invariato per associare le fonti scaricabili.
    """
    if not text:
        return ""

    cleaned = re.sub(r"\s*\[[^\]]+\]\(\s*#\d+\s*\)", "", text)
    cleaned = re.sub(r"\s*\[#\d+\]", "", cleaned)
    cleaned = re.sub(r" {2,}", " ", cleaned)
    cleaned = re.sub(r" ([.,;:!?])", r"\1", cleaned)
    return cleaned.strip()




@st.cache_data(ttl=10, show_spinner=False)
def _available_metadata_filters() -> dict[str, list]:
    """Per ogni campo metadata, valori distinti non vuoti presenti in collection.

    Solo i campi presenti (con almeno 1 valore valorizzato) finiscono nel dict.
    La UI dei filtri usa questo per mostrare solo dropdown utili.
    """
    coll = config.get_collection()
    result: dict[str, list] = {}
    for field in ("filename", "category", "status", "author", "year", "file_type"):
        distinct = [v for v in coll.distinct(field) if v not in (None, "", [])]
        if distinct:
            result[field] = sorted(distinct, key=lambda x: str(x))
    # tags: lista, serve unwind
    tags = [
        d["_id"] for d in coll.aggregate([
            {"$unwind": "$tags"},
            {"$group": {"_id": "$tags"}},
            {"$sort": {"_id": 1}},
        ]) if d.get("_id")
    ]
    if tags:
        result["tags"] = tags
    return result


@st.cache_data(ttl=60, show_spinner=False)
def _distinct_filenames() -> list[str]:
    """Lista dei filename presenti in collection (cached 60s)."""
    return sorted(config.get_collection().distinct("filename"))


@st.cache_data(ttl=60, show_spinner=False)
def _dominant_doc_langs() -> dict[str, str]:
    """Mappa filename → lingua dominante (la più rappresentata nei suoi chunk).

    Più affidabile della lingua per-chunk: langdetect sbaglia spesso su chunk
    corti / tecnici / con tabelle (es. il-piccolo-libro-di-mongodb ha 18 chunk
    taggati 'en' perché contengono codice MongoDB, ma il libro è italiano).
    """
    pipeline = [
        {"$group": {"_id": {"f": "$filename", "l": "$language"}, "n": {"$sum": 1}}},
        {"$sort": {"_id.f": 1, "n": -1}},
        {"$group": {"_id": "$_id.f", "lang": {"$first": "$_id.l"}}},
    ]
    return {
        d["_id"]: d["lang"]
        for d in config.get_collection().aggregate(pipeline)
        if d.get("lang")
    }


@st.cache_data(ttl=10, show_spinner=False)
def _document_list() -> list[dict]:
    """Aggregazione: 1 entry per `document_id` con metadati e conteggio chunk.

    Usata dal tab Admin per la gestione documenti (lista + delete singolo/all).
    TTL corto (10s) perché vogliamo vedere subito ingest/delete; la cache
    viene comunque invalidata esplicitamente dopo upload e dopo ogni delete.
    """
    coll = config.get_collection()
    pipeline = [
        {"$group": {
            "_id": "$document_id",
            "filename":       {"$first": "$filename"},
            "title":          {"$first": "$title"},
            "author":         {"$first": "$author"},
            "ingested_at":    {"$max":   "$ingested_at"},
            "chunks":         {"$sum":   1},
            "gridfs_file_id": {"$first": "$gridfs_file_id"},
            "file_type":      {"$first": "$file_type"},
            "total_pages":    {"$first": "$total_pages"},
        }},
        {"$sort": {"ingested_at": -1}},
    ]
    return list(coll.aggregate(pipeline))


def _build_adaptive_filters(values: dict[str, Any]) -> dict | None:
    """Costruisce il dict MQL filter partendo dai valori dei dropdown adattivi.

    `values` è una dict {field: selected_value} dove "(tutti)"/"(tutte)" = no filter.
    Filtra direttamente su ciascun campo per pipeline pulita e leggibile.
    """
    SENT = ("(tutti)", "(tutte)", "(tutti gli anni)", "")
    f: dict = {}
    for field in ("filename", "category", "status", "author", "file_type", "language"):
        v = values.get(field)
        if v and v not in SENT:
            f[field] = v
    year = values.get("year")
    if isinstance(year, (int, float)) and year not in SENT:
        f["year"] = int(year)
    tag = values.get("tags")
    if tag and tag not in SENT:
        f["tags"] = {"$in": [tag]}
    return f or None


# ================================================================== tabs =====

tab_story, tab_upload, tab_query, tab_admin = st.tabs(
    ["📖 How it Works", "📤 Upload", "🔎 Query", "⚙ Admin"]
)


# ─── HOW IT WORKS (home) ───────────────────────────────────────────────────────
with tab_story:
    st.markdown("<h2 class=\"gradient-header\">RAG Enterprise con MongoDB Atlas</h2>", unsafe_allow_html=True)
    st.image(
        str(Path(__file__).resolve().parent / "assets" / "how_it_works_before_after.png"),
        width="stretch",
    )
    st.markdown(
        "**Questa demo mostra MongoDB Atlas come piattaforma unica per una soluzione RAG:** "
        "documenti, chunk, metadati, file originali, ricerca full-text, ricerca vettoriale, "
        "filtri, ranking e fonti verificabili convivono nello stesso layer dati."
    )

    v1, v2, v3 = st.columns(3)
    with v1:
        st.markdown(
            """
            <div class="custom-card" style="height:100%; border-left:4px solid #00ED64;">
                <h4 style="margin-top:0;">Meno componenti</h4>
                <p style="margin-bottom:0; line-height:1.55;">
                Atlas gestisce dati, metadati, file, search e vector search.
                Niente vector database separato da sincronizzare.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with v2:
        st.markdown(
            """
            <div class="custom-card" style="height:100%; border-left:4px solid #00684A;">
                <h4 style="margin-top:0;">Ricerca piu' precisa</h4>
                <p style="margin-bottom:0; line-height:1.55;">
                Keyword search, semantic search, $rankFusion e Voyage rerank
                lavorano insieme per trovare il passaggio giusto.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with v3:
        st.markdown(
            """
            <div class="custom-card" style="height:100%; border-left:4px solid #7e57c2;">
                <h4 style="margin-top:0;">Risposte verificabili</h4>
                <p style="margin-bottom:0; line-height:1.55;">
                Il LLM risponde solo sui chunk recuperati, cita le fonti e
                permette di scaricare il documento originale da GridFS.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Il flusso end-to-end")
    st.graphviz_chart('''
        digraph flow_v2 {
            rankdir=LR
            bgcolor="transparent"
            node [fontname="Helvetica" fontsize=10 style="filled,rounded" shape=box penwidth=1.5]
            edge [color="#888888" penwidth=1.5]

            doc      [label="Documenti\\nPDF, Office, TXT" fillcolor="#FFFFFF" color="#00684a" fontcolor="#00684a"]
            step1    [label="1. Ingestion\\nParsing, OCR, tabelle, immagini" fillcolor="#F2FAF7" color="#00684a" fontcolor="#00684a"]
            step2    [label="2. MongoDB Atlas\\nChunk, metadati, GridFS" fillcolor="#E8F5EF" color="#00684a" fontcolor="#00684a"]
            step3    [label="3. Hybrid Search\\n$vectorSearch + $search + $rankFusion" fillcolor="#00684a" color="#00684a" fontcolor="#FFFFFF"]
            step4    [label="4. Voyage Rerank\\nCross-encoder di precisione" fillcolor="#7e57c2" color="#7e57c2" fontcolor="#FFFFFF"]
            step5    [label="5. LLM Grounded\\nRisposta solo sulle fonti" fillcolor="#F3E5F5" color="#7e57c2" fontcolor="#7e57c2"]
            answer   [label="Risposta verificabile\\nfonti + download PDF" fillcolor="#FFFFFF" color="#7e57c2" fontcolor="#7e57c2"]

            doc -> step1 -> step2 -> step3 -> step4 -> step5 -> answer
        }
    ''')

    st.markdown("### Perche' l'architettura unica conta")
    st.markdown(
        """
        <div class="custom-card" style="border-left:4px solid #00ED64;">
            <p style="font-size:16px; line-height:1.65; margin:0;">
            In una RAG tradizionale il dato vive spesso in piu' sistemi: object storage,
            database applicativo, motore full-text, vector database, job di embedding e
            pipeline di sincronizzazione. Qui MongoDB Atlas diventa il layer unico:
            la stessa aggregation combina ricerca semantica, ricerca lessicale,
            filtri sui metadati e ranking. Meno copie del dato significa meno codice,
            meno drift tra sistemi e debug piu' semplice.
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )

    before_col, after_col = st.columns(2)
    with before_col:
        st.markdown("#### RAG tradizionale")
        st.markdown(
            """
            <div class="custom-card" style="border-left:4px solid #fb8c00;">
                <ul style="margin:0; padding-left:18px; line-height:1.75;">
                    <li>File in object storage</li>
                    <li>Metadati in un database separato</li>
                    <li>Vector database dedicato</li>
                    <li>Search engine full-text separato</li>
                    <li>Job esterni per embedding e re-sync</li>
                    <li>Ranking e fusione nel codice applicativo</li>
                    <li>Governance distribuita tra piu' sistemi</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with after_col:
        st.markdown("#### MongoDB Atlas RAG")
        st.markdown(
            """
            <div class="custom-card" style="border-left:4px solid #00ED64;">
                <ul style="margin:0; padding-left:18px; line-height:1.75;">
                    <li>Chunk e metadati nella collection</li>
                    <li>PDF originali in GridFS</li>
                    <li>Embedding automatici con autoEmbed</li>
                    <li>Vector Search e Search nello stesso cluster</li>
                    <li>$rankFusion nativo dentro aggregation</li>
                    <li>Filtri metadata prima del retrieval</li>
                    <li>Ruoli, backup, audit e scaling Atlas</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Cosa succede quando fai una domanda")
    q1, q2, q3 = st.columns(3)
    with q1:
        st.markdown(
            """
            <div class="custom-card">
                <h4 style="margin-top:0;">1. Retrieval ampio</h4>
                <p style="line-height:1.55; margin-bottom:0;">
                MongoDB cerca in parallelo per significato con $vectorSearch e
                per parole esatte con $search. I filtri restringono subito il dominio.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with q2:
        st.markdown(
            """
            <div class="custom-card">
                <h4 style="margin-top:0;">2. Fusione e precisione</h4>
                <p style="line-height:1.55; margin-bottom:0;">
                $rankFusion combina le classifiche. Voyage rerank rilegge query e
                chunk come cross-encoder e promuove le fonti piu' rispondenti.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with q3:
        st.markdown(
            """
            <div class="custom-card">
                <h4 style="margin-top:0;">3. Risposta grounded</h4>
                <p style="line-height:1.55; margin-bottom:0;">
                Il LLM riceve solo i passaggi migliori. Se il rerank score e'
                troppo basso, la demo blocca la generazione off-topic.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Perche' hybrid search")
    st.markdown(
        """
        | Tipo domanda | Cosa serve | Perche' MongoDB aiuta |
        |---|---|---|
        | Codici, sigle, articoli, nomi propri | Ricerca lessicale precisa | `$search` trova termini esatti e varianti controllate |
        | Parafrasi e sinonimi | Ricerca semantica | `$vectorSearch` recupera chunk anche quando le parole non coincidono |
        | Domande miste | Fusione robusta | `$rankFusion` combina BM25 e vettori nello stesso database |
        | Domande fuori dominio | Controllo qualita' | Voyage rerank + soglia off-topic evitano risposte inventate |
        """
    )

    st.markdown("### Cosa dimostra questa demo")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown(
            """
            - Upload di PDF, Office, CSV, JSON, Markdown e testo.
            - Parsing con PyMuPDF, pdfplumber, MarkItDown e OCR.
            - Tabelle e immagini trasformate in contenuto ricercabile.
            - Metadati filtrabili: categoria, stato, anno, lingua, autore, tag.
            """
        )
    with d2:
        st.markdown(
            """
            - Hybrid retrieval dentro MongoDB Atlas.
            - Ranking spiegabile con `scoreDetails`.
            - Rerank Voyage per ordinamento finale di precisione.
            - Risposta grounded, fonti citate e download del PDF originale.
            """
        )

    st.markdown("### KPI osservabili")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Componenti core", "1", "MongoDB Atlas")
    m2.metric("Branch retrieval", "2", "vector + text")
    m3.metric("Round-trip DB", "1", "$rankFusion")
    m4.metric("Fonti", "verificabili", "chunk + PDF")

    st.markdown("### Approfondimenti Tecnici")

    with st.expander("La pipeline MongoDB Atlas: una sola aggregation"):
        st.markdown(
            "La ricerca ibrida viene eseguita nativamente in MongoDB con un solo round-trip "
            "applicativo. Il branch vector lavora sul significato, il branch text lavora sui "
            "termini esatti, `$rankFusion` fonde le classifiche."
        )
        st.code(
            '''db.pdf_chunks.aggregate([
  {
    $rankFusion: {
      input: { pipelines: {

        vector: [{
          $vectorSearch: {
            index: "pdf_chunks_vector_idx",
            path: "content",
            query: "<domanda utente>",
            numCandidates: 500,
            limit: 50,
            filter: { language: "it", category: "Manuale" }
          }
        }],

        text: [{
          $search: {
            index: "pdf_chunks_text_idx",
            compound: {
              must: [
                { text: { query: "<domanda utente>", path: "content" } }
              ],
              filter: [
                { equals: { path: "language", value: "it" } },
                { equals: { path: "category", value: "Manuale" } }
              ]
            }
          }
        }, { $limit: 50 }]

      }},
      combination: { weights: { vector: 1.0, text: 1.0 } },
      scoreDetails: true
    }
  },
  { $limit: 50 },
  {
    $project: {
      content: 1,
      filename: 1,
      page_number: 1,
      score: { $meta: "score" },
      scoreDetails: { $meta: "scoreDetails" }
    }
  }
])''',
            language="javascript",
        )

    with st.expander("autoEmbed: perche' e' importante"):
        st.markdown(
            """
            Con `autoEmbed`, l'applicazione inserisce testo e metadati nella collection.
            L'indice Atlas Vector Search genera gli embedding lato server tramite Voyage AI.
            Questo riduce codice applicativo, job batch, code di sincronizzazione e rischio
            che chunk, metadati e vettori vadano fuori allineamento.

            In questa demo la modalita' `auto` e' la piu' rappresentativa del messaggio
            MongoDB: scrivi dati operativi, Atlas mantiene l'indice semantico.
            """
        )

    with st.expander("I Modelli Voyage AI (Dettaglio)"):
        st.markdown(
            """
            I modelli Voyage AI permettono di scegliere il profilo piu' adatto al corpus:
            
            ##### General purpose
            - **`voyage-4-large`** (Default): Qualità massima multilingue per retrieval.
            - **`voyage-4-lite`**: Ottimizzato per bassa latenza e massima economicità.
            
            ##### Domain-specific
            - **`voyage-code-3`**: Ottimizzato per codice sorgente e documentazione tecnica.
            - **`voyage-finance-2`**: Ideale per report finanziari e bilanci.
            - **`voyage-law-2`**: Ottimizzato per contratti e testi normativi.
            
            ##### Reranking
            - **`rerank-2.5`** (Default): cross-encoder usato dopo `$rankFusion` per riordinare i candidati rispetto alla domanda.
            """
        )

    with st.expander("Anatomia di un Chunk & voyage-context-4 (L'evoluzione del Retrieval)"):
        st.markdown(
            """
            > **Recall ≠ Answerability**  
            > *Recall*: il chunk recuperato è quello giusto?  
            > *Answerability*: il chunk contiene la risposta *completa* (eccezioni e regole incluse)?
            
            Gli embedding classici elaborano ogni chunk in **isolamento**. Se un chunk contiene una regola e l'altro le eccezioni, finiscono lontani nello spazio vettoriale.
            
            ##### Come `voyage-context-4` risolve questo problema:
            Passando a Voyage tutti i chunk dello stesso documento come un unico gruppo, il modello genera vettori che **incorporano il contesto globale degli altri chunk** senza cambiare i confini page-aware prodotti dall'ingestion.
            - Pronomi e riferimenti non vanno più persi.
            - Pagina, tabella e tipo di contenuto rimangono associati al chunk originale.
            - Chunk size e overlap restano controllabili e verificabili nell'applicazione.
            
            ##### Confronto: Voyage context-4 vs Anthropic Contextual Retrieval
            """
        )
        st.markdown(
            """
            | Aspetto | Anthropic Contextual Retrieval | Voyage `context-4` |
            |---|---|---|
            | **Dove vive il contesto** | Aggiunto come **testo in chiaro** al chunk | **Compresso nello spazio latente** del vettore |
            | **Testo aggiunto al chunk** | Riassunto contestuale esplicito | Nessun testo aggiuntivo |
            | **Costo Ingestion** | 1 chiamata LLM per chunk | 1 chiamata Voyage per documento |
            | **Costo Generazione** | Prompt ingrossato dai riassunti | Solo i chunk originali |
            | **Architettura** | Augmentation testuale + embedding standard | Modello dedicato che codifica il contesto nel vettore |
            """,
            unsafe_allow_html=True
        )

    with st.expander("Automated Embedding & Motore Vettoriale Atlas"):
        st.markdown(
            """
            ##### Algoritmo HNSW & Pre-filtering:
            - **HNSW (Hierarchical Navigable Small Worlds)**: Naviga un grafo di vicinanze per trovare il chunk più rilevante in millisecondi senza scansioni complete.
            - **Pre-filtering nativo**: Filtri MQL (es: anno, autore, categoria) applicati *durante* la navigazione del grafo, non dopo, restringendo istantaneamente il dominio.
            """
        )


# ─── UPLOAD ────────────────────────────────────────────────────────────────────
with tab_upload:
    st.subheader("Carica un documento")
    st.caption(f"Modalità embedding attiva: **{st.session_state.embed_mode}** (cambia in Admin)")

    # Versioning delle key dei widget: incrementare = nuovi widget vuoti.
    # Permette di resettare il form senza perdere embed_mode (che vive
    # su una chiave session_state separata, non versionata).
    if "upload_form_v" not in st.session_state:
        st.session_state.upload_form_v = 0

    # ─── Caso A: ingestion già completata → riepilogo + pulsante reset ────────
    if "last_ingestion" in st.session_state:
        summary = st.session_state.last_ingestion
        if summary.get("skipped"):
            st.info(
                f"Documento gia' presente: nessun nuovo chunk inserito. "
                f"document_id `{summary['document_id'][:12]}...`"
            )
            st.caption(
                f"Policy duplicati `{summary.get('duplicate_policy', 'skip')}` · "
                f"{summary.get('reason', 'skip')}"
            )
            if st.button(
                "Carica un altro documento",
                type="primary",
                use_container_width=True,
                key="reset_upload_form_skipped",
            ):
                st.session_state.upload_form_v += 1
                st.session_state.pop("last_ingestion", None)
                st.rerun()
            st.stop()
        img_info = (
            f" · 🖼 {summary['images_processed']} immagini descritte"
            if summary.get("images_processed") else ""
        )
        st.success(
            f"✅ **Ultima ingestion completata** — Inseriti "
            f"**{summary['chunks_inserted']}** chunk · "
            f"mode `{summary['mode']}` · "
            f"document_id `{summary['document_id'][:12]}…`{img_info}"
        )
        if st.button(
            "🆕 Carica un altro documento",
            type="primary",
            use_container_width=True,
            key="reset_upload_form",
            help="Pulisce il form di upload mantenendo la modalità embedding attiva.",
        ):
            st.session_state.upload_form_v += 1
            st.session_state.pop("last_ingestion", None)
            st.rerun()

        # Render esplicito dei metadati effettivi
        st.markdown("### 📋 Metadati applicati")
        eff = summary["effective_metadata_applied_to_chunks"]
        user_provided = summary.get("user_metadata_provided") or {}

        def _fmt(v):
            if v is None or v == [] or v == "":
                return "_—_"
            if isinstance(v, list):
                return ", ".join(f"`{x}`" for x in v)
            return f"**{v}**"

        with st.container(border=True):
            md_lines = ["| Campo | Valore applicato | Origine |", "|---|---|---|"]
            field_order = [
                "title", "author", "language", "category",
                "status", "year", "tags", "file_type",
            ]
            origin_labels = {True: "👤 user override", False: "📄 da PDF/default"}
            for k in field_order:
                v_ = eff.get(k)
                is_user = k in user_provided and user_provided.get(k) is not None
                origin = origin_labels[is_user]
                md_lines.append(f"| **{k}** | {_fmt(v_)} | {origin} |")
            st.markdown("\n".join(md_lines))

        with st.expander("🔍 Dettaglio tecnico (raw summary JSON)"):
            st.json(summary)

    # ─── Caso B: form di upload ────────────────────────────────────────────────
    else:
        v = st.session_state.upload_form_v  # suffix delle key

        storage = st.radio(
            "Storage del file originale",
            options=["gridfs", "s3"],
            format_func=lambda x: {
                "gridfs": "GridFS (dentro Atlas)",
                "s3": "S3 esterno (non configurato)",
            }[x],
            horizontal=True,
            key=f"storage_v{v}",
        )
        if storage == "s3":
            st.info("Upload S3 non configurato. Verrà inserito solo il riferimento nei chunk.")
        uploaded = st.file_uploader(
            "Documento (PDF, DOCX, XLSX, PPTX, TXT, HTML, CSV, JSON, MD)",
            type=["pdf", "docx", "doc", "pptx", "ppt", "xlsx", "xls",
                  "txt", "html", "htm", "csv", "json", "xml", "md"],
            help="PDF parsato da PyMuPDF + pdfplumber + Tesseract (tabelle/OCR). "
                 "Altri formati via MarkItDown (Microsoft): markdown structured output.",
            key=f"file_v{v}",
        )

        with st.expander("📋 Metadati documento (opzionali — abilitano filtri smart in Query)",
                         expanded=True):
            st.caption(
                "Tutti i campi sono opzionali. Quelli compilati vengono ereditati da "
                "TUTTI i chunk del documento e diventano filtrabili nella tab Query "
                "(i filtri appaiono automaticamente solo se ci sono dati popolati)."
            )
            m_c1, m_c2 = st.columns(2)
            u_title = m_c1.text_input("Titolo", value="", key=f"u_title_v{v}",
                                       placeholder="es: Avviso consultazione metering Terna 2025")
            u_author = m_c2.text_input("Autore", value="", key=f"u_author_v{v}",
                                        placeholder="es: Terna SpA")

            m_c3, m_c4 = st.columns(2)
            u_category = m_c3.selectbox(
                "Categoria",
                options=["", "Contratto", "Manuale", "Report", "Specifica tecnica",
                         "Slide", "Articolo", "Email", "Avviso/Gara", "Altro"],
                index=0, key=f"u_cat_v{v}",
            )
            u_status = m_c4.selectbox(
                "Stato",
                options=["", "Bozza", "In revisione", "Approvato", "Pubblicato", "Archiviato"],
                index=0, key=f"u_status_v{v}",
            )

            m_c5, m_c6 = st.columns(2)
            u_language = m_c5.selectbox(
                "Lingua (override del detect automatico)",
                options=["", "it", "en", "fr", "de", "es", "pt"],
                format_func=lambda x: "(auto-detect)" if x == "" else {
                    "it": "🇮🇹 Italiano", "en": "🇬🇧 English", "fr": "🇫🇷 Français",
                    "de": "🇩🇪 Deutsch", "es": "🇪🇸 Español", "pt": "🇵🇹 Português",
                }.get(x, x),
                index=0, key=f"u_lang_v{v}",
            )
            u_year = m_c6.number_input(
                "Anno edizione", min_value=1900, max_value=2100, value=None,
                step=1, key=f"u_year_v{v}",
                placeholder="es: 2025",
                help="Anno di pubblicazione/edizione del documento (NON la data di ingest). "
                     "Lascia vuoto se non lo conosci o non vuoi filtrare per anno.",
            )

            u_tags = st.text_input(
                "Tag (separati da virgola)", value="", key=f"u_tags_v{v}",
                placeholder="es: metering, energia, terna, gara",
            )

            u_duplicate_policy = st.radio(
                "Se il documento e' gia' presente",
                options=["skip", "replace", "append"],
                index=["skip", "replace", "append"].index(config.INGESTION_DUPLICATE_POLICY)
                if config.INGESTION_DUPLICATE_POLICY in ("skip", "replace", "append") else 0,
                format_func=lambda x: {
                    "skip": "Salta upload duplicati",
                    "replace": "Sostituisci chunk e file esistenti",
                    "append": "Inserisci comunque una nuova copia",
                }[x],
                horizontal=True,
                key=f"u_duplicate_policy_v{v}",
                help="Il documento e' riconosciuto tramite hash SHA-256 del file.",
            )

            st.divider()
            # Vision support dipende dal LLM_PROVIDER attivo
            _vision_ok_per_provider = {
                "anthropic": bool(config.ANTHROPIC_API_KEY),
                "gemini":    bool(config.GEMINI_API_KEY),
                "ollama":    True,  # richiede OLLAMA_VISION_MODEL pullato (default llava:7b)
            }
            _vision_provider_label = {
                "anthropic": "Claude Haiku Vision",
                "gemini":    f"Gemini Vision ({config.GEMINI_MODEL})",
                "ollama":    f"Ollama Vision ({config.OLLAMA_VISION_MODEL}, locale)",
            }
            _vision_ok = _vision_ok_per_provider.get(config.LLM_PROVIDER, False)
            _vision_label = _vision_provider_label.get(config.LLM_PROVIDER, config.LLM_PROVIDER)

            u_process_images = st.toggle(
                f"🖼 Estrai immagini dal PDF e descrivile ({_vision_label})",
                value=False, key=f"u_process_images_v{v}",
                help="Estrae le immagini significative dal PDF (≥150px), genera una "
                     "descrizione testuale via vision LLM, e le inserisce come "
                     "chunk addizionali con `content_type='image'`. Provider attivo "
                     "= quello configurato in `LLM_PROVIDER`. Utile per disegni "
                     "tecnici, schemi, foto.",
            )
            if u_process_images and not _vision_ok:
                key_name = {"gemini": "GEMINI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}.get(
                    config.LLM_PROVIDER, "API_KEY"
                )
                st.warning(
                    f"⚠ Toggle attivo ma `{key_name}` mancante nel `.env` → "
                    "le immagini verranno estratte ma NON descritte."
                )
            elif u_process_images and config.LLM_PROVIDER == "ollama":
                st.caption(
                    f"ℹ Userai il modello locale `{config.OLLAMA_VISION_MODEL}` per le "
                    f"descrizioni. Se non è installato, esegui: "
                    f"`ollama pull {config.OLLAMA_VISION_MODEL}`. "
                    f"Latenza: ~3-10s per immagine su Mac M-series."
                )

        if uploaded and st.button("Avvia ingestion", type="primary"):
            user_md: dict = {}
            if u_title.strip():    user_md["title"] = u_title.strip()
            if u_author.strip():   user_md["author"] = u_author.strip()
            if u_category:         user_md["category"] = u_category
            if u_status:           user_md["status"] = u_status
            if u_language:         user_md["language"] = u_language
            if u_year is not None: user_md["year"] = int(u_year)
            if u_tags.strip():
                user_md["tags"] = [t.strip() for t in u_tags.split(",") if t.strip()]
            if u_process_images:
                user_md["process_images"] = True
            user_md["file_type"] = Path(uploaded.name).suffix.lstrip(".").lower() or "pdf"
            user_md["duplicate_policy"] = u_duplicate_policy

            with st.status("Ingestion in corso…", expanded=True) as status:
                tmp = Path(tempfile.mkdtemp()) / uploaded.name
                tmp.write_bytes(uploaded.getvalue())
                st.write(f"File: `{tmp.name}`")
                if user_md:
                    st.write(f"Metadati applicati: `{user_md}`")
                summary = ingest_pdf(
                    tmp, db=_db(), collection=_coll(),
                    storage=storage, mode=st.session_state.embed_mode,
                    user_metadata=user_md,
                )
                status.update(label="Completata", state="complete")

            # Invalida cache filtri/lista così Query e Admin li ricaricano
            _available_metadata_filters.clear()
            _distinct_filenames.clear()
            _dominant_doc_langs.clear()
            _document_list.clear()

            # Persisto il summary e ricarico: alla prossima render verrà
            # mostrato il blocco "Caso A" con il pulsante "Carica un altro
            # documento" — il form sparisce finché non l'utente non resetta.
            st.session_state.last_ingestion = summary
            st.rerun()



# ─── QUERY — helper di rendering ────────────────────────────────────────────
#
# Pipeline retrieval/rerank/generation. Sotto la risposta:
#   1. Risposta del LLM (citazioni Markdown strippate)
#   2. Bottoni di download per le fonti citate
#   3. Per ogni fonte CITATA: card con Rank Journey (BM25 → Vector → Hybrid → Rerank)
#   4. Matrice di rank di tutto il top-N
#   5. Vista pulita (3 colonne: Lessicale · Semantica · Entrambi)
#   6. Expander "Analisi Tecnica" con 2 sotto-tab: Top-N completi · Latenze+MQL
#
# Le ricerche per-modalità sono profonde (top-50) per Rank Journey/Matrice.

_JOURNEY_TOP_K = 50

# Colori per i marker delle fonti citate dal LLM (Rank Journey, vista pulita)
_CITATION_BORDER_COLOR = "#8e24aa"  # viola
_CITATION_BG = "rgba(142, 36, 170, 0.12)"


def _position_in(results: list[dict], target_key: tuple) -> int | None:
    """Trova la posizione 1-based di `target_key` in `results`, o None se non c'è."""
    for i, c in enumerate(results, start=1):
        k = (c.get("filename"), c.get("page_number"), c.get("chunk_index"))
        if k == target_key:
            return i
    return None


def _render_rank_journey(
    cited_chunks_with_cit: list[tuple[dict, list[int]]],
    text_deep: list[dict],
    vector_deep: list[dict],
    hybrid_deep: list[dict],
    rerank_results: list[dict],
    query: str,
) -> None:
    """Per ogni chunk citato mostra la sua "storia" lungo la pipeline.

    `cited_chunks_with_cit`: lista di (chunk_dict, lista_numeri_citazione).
    """
    if not cited_chunks_with_cit:
        return

    st.markdown("### Rank Journey delle fonti citate")
    st.caption(
        "Per ogni fonte effettivamente usata dal LLM, una **storia individuale**: "
        f"vedi la posizione in cui ognuno dei 4 stadi della pipeline ha collocato "
        f"questa specifica fonte. Da sinistra a destra: lessicale → semantico → "
        f"hybrid → rerank. Le frecce ➔ rappresentano la **promozione** lungo gli "
        f"stadi. Le ricerche per stadio sono profonde fino a top-{_JOURNEY_TOP_K} "
        f"per mostrarti **posizioni reali** anche per fonti inizialmente lontane "
        f"(es. `#27 BM25 → #4 Vector → #2 Hybrid → #1 Rerank` racconta meglio "
        f"di \"non trovata in top-5\")."
    )

    for chunk, cit_nums in cited_chunks_with_cit:
        key = (chunk.get("filename"), chunk.get("page_number"),
               chunk.get("chunk_index"))
        pos_text   = _position_in(text_deep,   key)
        pos_vector = _position_in(vector_deep, key)
        pos_hybrid = _position_in(hybrid_deep, key)
        pos_rerank = _position_in(rerank_results, key)

        # Snippet
        content = (chunk.get("content") or "").strip().replace("\n", " ")
        snippet = content[:200] + ("…" if len(content) > 200 else "")
        snippet_safe = _escape_markdown_linestart(html.escape(snippet))

        fname = chunk.get("filename", "?")
        page_lbl = _page_label(chunk.get("page_number"))
        cit_label = ", ".join(f"[#{n_}]" for n_ in sorted(cit_nums))

        # Header card
        st.markdown(
            f'<div style="margin-top:14px;padding:10px 14px;'
            f'border-left:5px solid {_CITATION_BORDER_COLOR};'
            f'background:{_CITATION_BG};border-radius:4px">'
            f'<div style="display:flex;align-items:center;gap:10px;'
            f'flex-wrap:wrap;font-size:1.02em">'
            f'<span style="background:{_CITATION_BORDER_COLOR};color:white;'
            f'padding:3px 10px;border-radius:10px;font-weight:700">'
            f'💬 {cit_label}</span>'
            f'<span style="font-weight:600">{html.escape(fname)}</span>'
            f'<span style="opacity:0.65;font-size:0.9em">· {page_lbl}</span>'
            f'</div>'
            f'<div style="opacity:0.7;font-size:0.88em;margin-top:6px;'
            f'font-style:italic;line-height:1.4">"{snippet_safe}"</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Journey badges con frecce
        def _badge(label: str, pos: int | None, highlight: bool = False) -> str:
            if pos is None:
                color = "#9e9e9e"
                bg = "rgba(158,158,158,0.15)"
                txt = f"<span style='color:{color}'>{label}: <b>—</b></span>"
            elif highlight:
                bg = f"{_CITATION_BORDER_COLOR}"
                txt = f"<span style='color:white'>{label}: <b>#{pos}</b></span>"
            else:
                pal = "#1e88e5" if pos <= 5 else ("#fb8c00" if pos <= 15 else "#9e9e9e")
                bg = f"rgba({30 if pos <= 5 else (251 if pos <= 15 else 158)},"
                bg += f"{136 if pos <= 5 else (140 if pos <= 15 else 158)},"
                bg += f"{229 if pos <= 5 else (0 if pos <= 15 else 158)},0.15)"
                txt = f"<span style='color:{pal}'>{label}: <b>#{pos}</b></span>"
            return (
                f"<span style='padding:4px 12px;background:{bg};"
                f"border-radius:14px;font-size:0.92em'>{txt}</span>"
            )

        arrow = ("<span style='color:#bbb;font-size:1.1em;padding:0 4px'>"
                 "➔</span>")
        journey_html = (
            f"<div style='display:flex;align-items:center;gap:6px;"
            f"flex-wrap:wrap;padding:8px 14px;margin-bottom:6px'>"
            f"{_badge('🔤 Lessicale', pos_text)}{arrow}"
            f"{_badge('🧠 Semantica', pos_vector)}{arrow}"
            f"{_badge('🔀 Hybrid RRF', pos_hybrid)}{arrow}"
            f"{_badge('🎯 Rerank', pos_rerank, highlight=True)}"
            f"</div>"
        )
        st.markdown(journey_html, unsafe_allow_html=True)


def _render_comparison_matrix(
    text_deep: list[dict],
    vector_deep: list[dict],
    hybrid_deep: list[dict],
    rerank_results: list[dict],
    cited_keys: dict[tuple, list[int]],
) -> None:
    """Tabella matrix: 1 riga per chunk Top-N rerank, colonne = posizioni
    nelle modalità (BM25, Vector, Hybrid, Rerank, LLM cited)."""
    import pandas as pd

    rows = []
    for c in rerank_results:
        k = (c.get("filename"), c.get("page_number"), c.get("chunk_index"))
        fname = c.get("filename", "?")
        page = c.get("page_number")
        page_lbl = _page_label(page, missing="auto")
        rows.append({
            "Chunk": f"{fname} · {page_lbl}",
            "🔤 BM25":    _position_in(text_deep, k),
            "🧠 Vector":  _position_in(vector_deep, k),
            "🔀 Hybrid":  _position_in(hybrid_deep, k),
            "🎯 Rerank":  _position_in(rerank_results, k),
            "💬 Citato": ("[" + ", ".join(f"#{n}" for n in sorted(cited_keys[k])) + "]"
                          if k in cited_keys else "—"),
        })
    if not rows:
        st.caption("_(nessun chunk nel rerank top-N)_")
        return

    df = pd.DataFrame(rows)

    def _color_pos(val):
        if val is None or pd.isna(val):
            return "background-color: rgba(158,158,158,0.10); color: #888"
        if val == 1:
            return "background-color: #2e7d32; color: white; font-weight: 700"
        if val <= 3:
            return "background-color: #ef6c00; color: white; font-weight: 600"
        if val <= 10:
            return "background-color: #fbc02d; color: black; font-weight: 600"
        return "background-color: rgba(158,158,158,0.10)"

    rank_cols = ["🔤 BM25", "🧠 Vector", "🔀 Hybrid", "🎯 Rerank"]
    styler = df.style
    style_fn = getattr(styler, "map", None) or styler.applymap
    styled = style_fn(_color_pos, subset=rank_cols)

    st.dataframe(
        styled, use_container_width=True, hide_index=True,
        column_config={
            "Chunk": st.column_config.TextColumn(width="medium"),
            "🔤 BM25":   st.column_config.NumberColumn(format="#%d"),
            "🧠 Vector": st.column_config.NumberColumn(format="#%d"),
            "🔀 Hybrid": st.column_config.NumberColumn(format="#%d"),
            "🎯 Rerank": st.column_config.NumberColumn(format="#%d"),
            "💬 Citato": st.column_config.TextColumn(width="small"),
        },
    )
    st.caption(
        "**Legenda celle**: 🟢 #1 · 🟠 top-3 · 🟡 top-10 · ⬜ posizioni più "
        "basse · ▪️ non trovato nei top-50."
    )


def _render_clean_retrieval_view(
    text_topN: list[dict],
    vector_topN: list[dict],
    rerank_results: list[dict],
    cited_chunks_with_cit: list[tuple[dict, list[int]]],
    query: str,
) -> None:
    """Vista pulita del retrieval: 3 colonne (Solo Semantica · Solo Lessicale ·
    Entrambi) + sezione narrativa del rerank. Niente numeri di rank.

    Concetto: il numero `#10` di posizione di un chunk in un algoritmo
    intermedio non aggiunge valore di business. Qui mostriamo solo
    **da quale motore di ricerca arriva** ciascun chunk e **quale è stato
    promosso dal rerank** come fonte principale.
    """
    def _key(c: dict) -> tuple:
        return (c.get("filename"), c.get("page_number"), c.get("chunk_index"))

    text_keys   = {_key(c): c for c in text_topN}
    vector_keys = {_key(c): c for c in vector_topN}

    only_vector_keys = set(vector_keys) - set(text_keys)
    only_text_keys   = set(text_keys)   - set(vector_keys)
    both_keys        = set(text_keys)   & set(vector_keys)

    only_vector_chunks = [vector_keys[k] for k in only_vector_keys]
    only_text_chunks   = [text_keys[k]   for k in only_text_keys]
    both_chunks        = [text_keys[k]   for k in both_keys]

    st.markdown("### Vista pulita: dove ha pescato ciascun motore di ricerca")
    st.caption(
        "Stessi chunk del rerank top-N, senza numeri di rank intermedi. "
        "Solo l'**etichetta dell'origine** (chi ha trovato cosa) e gli "
        "snippet di testo reali."
    )

    def _render_clean_card(c: dict, origin_label: str, origin_color: str) -> None:
        fname = c.get("filename", "?")
        page_lbl = _page_label(c.get("page_number"), missing="🤖 auto")
        snippet = (c.get("content") or "").strip().replace("\n", " ")[:240]
        if len(c.get("content") or "") > 240:
            snippet += "…"
        # Rendering neutro: html.escape + markdown escape, niente <mark>.
        snippet_html = _safe_html(snippet)
        st.markdown(
            f'<div style="padding:10px 12px;margin-bottom:8px;'
            f'border-left:4px solid {origin_color};'
            f'background:rgba(128,128,128,0.06);border-radius:4px;'
            f'font-size:0.88em">'
            f'<div style="margin-bottom:6px">'
            f'<span style="background:{origin_color};color:white;'
            f'padding:2px 8px;border-radius:10px;font-size:0.82em;'
            f'font-weight:600">{origin_label}</span></div>'
            f'<div style="font-weight:600;margin-bottom:2px">'
            f'{html.escape(fname)} · <span style="opacity:0.7">{page_lbl}</span>'
            f'</div>'
            f'<div style="opacity:0.85;font-style:italic;line-height:1.4">'
            f'"{snippet_html}"</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    cols = st.columns(3)
    columns_data = [
        ("🔤 Solo Lessicale", only_text_chunks,
         "🔤 Recuperato dal BM25", "#fb8c00"),
        ("🧠 Solo Semantica", only_vector_chunks,
         "🧠 Recuperato dal vector search", "#1e88e5"),
        ("🔀 Entrambi", both_chunks,
         "🔀 Recuperato da entrambi", "#7e57c2"),
    ]
    for col, (label, chunks, origin_label, origin_color) in zip(cols, columns_data):
        with col:
            st.markdown(f"##### {label}")
            st.caption(f"**{len(chunks)}** chunk")
            if not chunks:
                st.caption(
                    "_(nessun chunk in questo bucket — questa modalità "
                    "non ha pescato nulla di unico)_"
                )
                continue
            for c in chunks:
                _render_clean_card(c, origin_label, origin_color)

    # ── Sezione narrativa del rerank ─────────────────────────────────────
    if cited_chunks_with_cit:
        st.markdown("### 🎯 Cosa ha selezionato il Reranker come fonte principale")
        st.caption(
            "Il **cross-encoder Voyage rerank-2.5** ha letto query + chunk "
            "insieme e ha promosso le seguenti fonti come quelle che davvero "
            "rispondono alla domanda. Sono le **uniche citate dal LLM**."
        )
        for chunk, cit_nums in cited_chunks_with_cit:
            cit_label = ", ".join(f"[#{n}]" for n in sorted(cit_nums))
            fname = chunk.get("filename", "?")
            page_lbl = _page_label(chunk.get("page_number"), missing="🤖 auto")
            content = (chunk.get("content") or "").strip().replace("\n", " ")
            snippet = content[:300] + ("…" if len(content) > 300 else "")
            snippet_html = _safe_html(snippet)

            st.markdown(
                f'<div style="margin:10px 0;padding:14px 18px;'
                f'border-left:6px solid #2e7d32;'
                f'background:rgba(46,125,50,0.08);border-radius:6px">'
                f'<div style="display:flex;align-items:center;gap:10px;'
                f'flex-wrap:wrap;margin-bottom:8px">'
                f'<span style="background:#2e7d32;color:white;padding:3px 12px;'
                f'border-radius:12px;font-weight:700;font-size:0.9em">'
                f'✅ Fonte citata {cit_label}</span>'
                f'<span style="font-weight:700;font-size:1.02em">'
                f'{html.escape(fname)}</span>'
                f'<span style="opacity:0.7">· {page_lbl}</span>'
                f'</div>'
                f'<div style="font-style:italic;opacity:0.9;'
                f'line-height:1.5;font-size:0.95em">"{snippet_html}"</div>'
                f'<div style="margin-top:8px;font-size:0.85em;opacity:0.75">'
                f'➜ Il reranker ha promosso questo chunk come fonte principale '
                f'perché contiene la risposta diretta alla tua domanda.'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ─── QUERY — la pagina ─────────────────────────────────────────────────────
with tab_query:
    preset_options_v2 = ["(scrivi una domanda libera)"] + [p["label"] for p in DEMO_PRESETS]
    if "query_v2_text" not in st.session_state:
        st.session_state.query_v2_text = _DEFAULT_QUERY

    def _on_preset_change_v2() -> None:
        choice = st.session_state.get("preset_choice_v2", "")
        if choice == "(scrivi una domanda libera)":
            st.session_state.query_v2_text = _DEFAULT_QUERY
            return
        for p in DEMO_PRESETS:
            if p["label"] == choice:
                st.session_state.query_v2_text = p["query"]
                return

    st.selectbox(
        "🎬 Scenari demo (popola automaticamente la domanda)",
        options=preset_options_v2, index=0,
        key="preset_choice_v2", on_change=_on_preset_change_v2,
    )
    for p in DEMO_PRESETS:
        if p["label"] == st.session_state.get("preset_choice_v2"):
            st.markdown(
                f'<div style="border:1px solid rgba(0,237,100,0.25);'
                f'background:rgba(0,237,100,0.06);border-radius:8px;'
                f'padding:12px 14px;margin:8px 0 12px 0">'
                f'<div style="font-weight:750;margin-bottom:4px">Cosa dimostriamo: '
                f'{html.escape(p["label"])}</div>'
                f'<div style="line-height:1.5;margin-bottom:8px">'
                f'{html.escape(p["demo_focus"])}</div>'
                f'{_demo_badges_html(p.get("badges", []))}'
                f'</div>',
                unsafe_allow_html=True,
            )
            break
    else:
        st.caption(
            "Domanda libera: la demo usera' comunque hybrid search, rerank, "
            "grounding e fonti verificabili."
        )

    query_v2 = st.text_input(
        "Domanda", value="", placeholder=_DEFAULT_QUERY, key="query_v2_text",
    )

    with st.expander("Filtri sui metadati (auto-adattivi)", expanded=False):
        col_caption2, col_refresh2 = st.columns([4, 1])
        col_caption2.caption(
            "Filtri **smart**: appaiono solo quelli che hanno almeno un valore "
            "popolato in collection. Più ricchi i metadati in upload, più "
            "filtri compaiono qui."
        )
        if col_refresh2.button("🔄 Refresh", key="refresh_filters_v2"):
            _available_metadata_filters.clear()
            _distinct_filenames.clear()
            _dominant_doc_langs.clear()
            st.rerun()
        available_v2 = _available_metadata_filters()
        doc_langs_v2 = _dominant_doc_langs()
        filter_values_v2: dict = {}
        FIELD_LABEL_V2 = {
            "filename": ("📄 Documento", "tutti"),
            "category": ("🏷 Categoria", "tutte"),
            "status":   ("📊 Stato", "tutti"),
            "author":   ("👤 Autore", "tutti"),
            "year":     ("📅 Anno", "tutti gli anni"),
            "tags":     ("🔖 Tag", "tutti"),
            "file_type": ("📦 Tipo file", "tutti"),
        }
        active_filters_v2 = [
            f for f in ("category", "status", "author", "year", "tags", "file_type")
            if f in available_v2
        ]
        widgets_v2 = list(active_filters_v2)
        if doc_langs_v2:
            widgets_v2.append("language")
        if not widgets_v2:
            st.caption("_(nessun metadato popolato)_")
        else:
            for i in range(0, len(widgets_v2), 2):
                row = widgets_v2[i:i + 2]
                cols = st.columns(2)
                for col, w in zip(cols, row):
                    if w == "language":
                        available_langs_v2 = sorted(set(doc_langs_v2.values()))
                        lang_labels = {
                            "it": "🇮🇹 Italiano", "en": "🇬🇧 English",
                            "fr": "🇫🇷 Français", "de": "🇩🇪 Deutsch",
                            "es": "🇪🇸 Español",  "pt": "🇵🇹 Português",
                        }
                        sel = col.selectbox(
                            "🌍 Lingua", options=["(tutte)"] + available_langs_v2,
                            format_func=lambda x: "(tutte le lingue)" if x == "(tutte)"
                                                  else lang_labels.get(x, x.upper()),
                            key="f_lang_v2", accept_new_options=True,
                        )
                        filter_values_v2["language"] = sel
                    else:
                        label, default_label = FIELD_LABEL_V2[w]
                        default = "(tutti)" if "tutti" in default_label else "(tutte)"
                        opts = [default] + available_v2[w]
                        allow_custom = (w != "year")
                        sel = col.selectbox(
                            label, options=opts, key=f"f_{w}_v2",
                            accept_new_options=allow_custom,
                            format_func=lambda x, dl=default_label, d=default:
                                        f"({dl})" if x == d else str(x),
                        )
                        filter_values_v2[w] = sel

    with st.expander("⚙ Parametri avanzati"):
        c6v, c7v = st.columns(2)
        with c6v:
            top_k_v2 = st.slider(
                "Top-K ($rankFusion)", 5, 100,
                config.RETRIEVAL_TOP_K, key="topk_v2",
            )
        with c7v:
            top_n_v2 = st.slider(
                "Top-N (chunk al LLM)", 1, 20,
                config.RERANK_TOP_N, key="topn_v2",
            )
        st.divider()
        fuzzy_v2 = st.toggle("🔍 Fuzzy match sul branch text", value=False, key="fuzzy_v2")
        relevance_threshold_v2 = st.slider(
            "🚫 Soglia rerank minima (off-topic)",
            0.0, 1.0, config.RERANK_RELEVANCE_THRESHOLD,
            step=0.05, key="rel_threshold_v2",
        )

    if st.button("Esegui", type="primary", disabled=not query_v2, key="run_v2"):
        filters_v2 = _build_adaptive_filters(filter_values_v2)
        timings_ms_v2: dict[str, float] = {}
        query_vector_v2: list[float] | None = None

        if st.session_state.embed_mode != "auto":
            from src.retrieval import _query_vector
            try:
                with st.spinner("Embedding della query..."):
                    query_vector_v2 = _query_vector(
                        query_v2, st.session_state.embed_mode
                    )
            except Exception as e:
                st.error("Non è stato possibile generare l'embedding della query.")
                with st.expander("Dettaglio tecnico"):
                    st.code(str(e))
                st.stop()

        # 1. Retrieval ibrido principale
        t0 = time.perf_counter()
        try:
            with st.spinner("Hybrid retrieval ($rankFusion)…"):
                raw_v2 = search(_coll(), query_v2, filters=filters_v2,
                                top_k=top_k_v2,
                                embed_mode=st.session_state.embed_mode,
                                retrieval_mode="hybrid", fuzzy=fuzzy_v2,
                                query_vector=query_vector_v2)
        except Exception as e:
            err_msg = str(e)
            if "INITIAL_SYNC" in err_msg or "while in state" in err_msg:
                st.warning("🛠 Indice in costruzione, aspetta 30-60s e riprova.")
            elif "HostUnreachable" in err_msg or ":28000" in err_msg:
                st.error("🚨 Atlas Search non raggiungibile dal cluster.")
            elif "is not indexed as vector" in err_msg or "queryVector must be present" in err_msg:
                st.error("🚨 Schema indice incoerente con la modalità embedding. Vai in Admin.")
            else:
                raise
            with st.expander("Errore tecnico"):
                st.code(err_msg)
            st.stop()
        timings_ms_v2["hybrid"] = (time.perf_counter() - t0) * 1000

        if not raw_v2:
            st.warning(
                "Nessun risultato trovato con la query e i filtri selezionati."
            )
            st.caption(
                f"Filtri applicati: `{_filter_summary(filters_v2)}`. "
                "Modifica o rimuovi uno dei filtri e riprova."
            )
            st.stop()

        # 2. Rerank: preferisce lo stage MongoDB nativo; fallback API Voyage.
        t0 = time.perf_counter()
        rerank_mode_v2 = "native"
        native_rerank_error_v2 = ""
        reranked_v2: list[dict] = []
        native_support_v2 = st.session_state.get("atlas_native_rerank_supported")

        if native_support_v2 is not False:
            try:
                with st.spinner("Reranking nativo MongoDB…"):
                    reranked_v2 = search_with_native_rerank(
                        _coll(),
                        query_v2,
                        filters=filters_v2,
                        top_k=top_k_v2,
                        top_n=top_n_v2,
                        embed_mode=st.session_state.embed_mode,
                        fuzzy=fuzzy_v2,
                        query_vector=query_vector_v2,
                    )
                st.session_state.atlas_native_rerank_supported = True
            except Exception as e:
                native_rerank_error_v2 = str(e)
                unsupported_native_v2 = (
                    getattr(e, "code", None) == 40324
                    or "Unrecognized pipeline stage name: '$rerank'" in native_rerank_error_v2
                )
                if unsupported_native_v2:
                    st.session_state.atlas_native_rerank_supported = False

        if not reranked_v2:
            rerank_mode_v2 = "voyage_api"
            if config.VOYAGE_API_KEY:
                with st.spinner("Reranking Voyage API…"):
                    reranked_v2 = rerank(query_v2, raw_v2, top_n=top_n_v2)
            else:
                rerank_mode_v2 = "disabled"
                st.warning(
                    "Reranking nativo non disponibile e VOYAGE_API_KEY mancante: "
                    "uso direttamente i primi risultati RRF."
                )
                reranked_v2 = raw_v2[:top_n_v2]
                for c in reranked_v2:
                    c["_rerank_score"] = None

        timings_ms_v2["rerank"] = (time.perf_counter() - t0) * 1000

        if not reranked_v2:
            st.warning(
                "La ricerca ha trovato candidati, ma il reranking non ha "
                "restituito passaggi utilizzabili."
            )
            st.stop()

        # 3. Off-topic detection
        top_rerank_v2 = reranked_v2[0].get("_rerank_score")
        if isinstance(top_rerank_v2, (int, float)) and top_rerank_v2 < relevance_threshold_v2:
            st.error(
                f"🚫 Off-topic (rerank `{top_rerank_v2:.3f}` < soglia `{relevance_threshold_v2:.2f}`)"
            )
            st.stop()

        # 4. Generation
        st.markdown("### Risposta")
        if llm_available():
            stream_v2 = generate_answer_stream(query_v2, reranked_v2)
            # Consumiamo lo stream senza renderizzare il Markdown parziale:
            # evita che link e marker tecnici appaiano per poi cambiare.
            answer_placeholder = st.empty()
            try:
                with st.spinner("Generazione della risposta..."):
                    for _ in stream_v2:
                        pass
            except Exception as e:
                answer_placeholder.empty()
                err_msg = str(e)
                if "temporaneamente non disponibile" in err_msg:
                    st.error(
                        "Gemini è temporaneamente sovraccarico. "
                        "Ho già effettuato più tentativi automatici: riprova tra poco."
                    )
                elif "limite di richieste" in err_msg:
                    st.error(
                        "Gemini ha raggiunto il limite di richieste. "
                        "Attendi qualche minuto prima di riprovare."
                    )
                elif "non è raggiungibile" in err_msg:
                    st.error(
                        "Non riesco a raggiungere Gemini. "
                        "Controlla la connessione e riprova."
                    )
                else:
                    st.error("La generazione della risposta non è riuscita.")
                with st.expander("Dettaglio tecnico"):
                    st.code(err_msg)
                st.stop()
            answer_text_v2 = stream_v2.full_text
            answer_provider_v2 = stream_v2.provider
            answer_model_v2 = stream_v2.model
            answer_usage_v2 = stream_v2.usage
            repaired_v2 = False
            if _needs_generation_repair(answer_text_v2, top_rerank_v2):
                with st.spinner("Rifinitura risposta..."):
                    try:
                        repaired = generate_answer(query_v2, reranked_v2)
                        repaired_text = repaired.get("answer") or ""
                        if repaired_text and not _looks_incomplete_answer(repaired_text):
                            answer_text_v2 = repaired_text
                            answer_provider_v2 = repaired.get("provider", answer_provider_v2)
                            answer_model_v2 = repaired.get("model", answer_model_v2)
                            answer_usage_v2 = repaired.get("usage") or answer_usage_v2
                            repaired_v2 = True
                    except Exception as e:
                        st.caption(f"Retry generation non riuscito: {type(e).__name__}")
            fallback_v2 = False
            if _needs_generation_repair(answer_text_v2, top_rerank_v2):
                answer_text_v2 = _extractive_fallback_answer(reranked_v2)
                repaired_v2 = False
                fallback_v2 = True
            answer_placeholder.markdown(
                _display_answer_text(answer_text_v2)
            )
            top_disp = f"{top_rerank_v2:.3f}" if isinstance(top_rerank_v2, (int, float)) else "—"
            st.caption(
                f"Provider `{answer_provider_v2}` · Model `{answer_model_v2}` · "
                f"Token in={answer_usage_v2.get('input_tokens')} "
                f"out={answer_usage_v2.get('output_tokens')} · "
                f"Top rerank `{top_disp}`"
                f"{' · risposta rifinita' if repaired_v2 else ''}"
                f"{' · fallback estrattivo' if fallback_v2 else ''}"
            )
            citations_v2 = _parse_citations(answer_text_v2)
            if not citations_v2:
                if reranked_v2 and isinstance(top_rerank_v2, (int, float)) and top_rerank_v2 >= relevance_threshold_v2:
                    citations_v2 = [1]
                    st.caption(
                        "Il modello non ha emesso marker di citazione; uso come fonte "
                        "scaricabile il chunk top rerank."
                    )
                else:
                    st.warning(
                        "Il modello ha prodotto una risposta senza citazioni. "
                        "La mostro comunque, ma considera la risposta non verificata."
                    )
            _render_trust_panel(
                raw_results=raw_v2,
                reranked=reranked_v2,
                citations=citations_v2,
                filters=filters_v2,
                top_k=top_k_v2,
                timings=timings_ms_v2,
                top_rerank=top_rerank_v2,
                threshold=relevance_threshold_v2,
                provider=answer_provider_v2,
                model=answer_model_v2,
                usage=answer_usage_v2,
            )
            # Download button per ciascuna fonte unica citata — subito sotto
            # la risposta, così l'utente può scaricare il PDF originale
            # senza dover scrollare in fondo.
            _render_cited_sources_downloads(
                reranked_v2, answer_text=answer_text_v2 if _parse_citations(answer_text_v2) else "[#1]",
            )
        else:
            citations_v2 = []
            st.info("LLM disattivato. Mostro solo retrieval.")
            st.stop()

        # 5. Mappa citazioni → chunk + dedup
        #
        # IMPORTANTE: si dedupa per FILE, non per chunk. Se il LLM cita più
        # chunk dello stesso PDF (es. [#2] = pag 3 e [#5] = pag 7 dello
        # stesso documento), nella vista Rank Journey/Vista pulita appare
        # UNA SOLA card per quel file, con tutte le citazioni aggregate
        # (es. "💬 [#2], [#5]"). Il rappresentante è il chunk con il
        # numero di citazione PIÙ BASSO (= il primo apparso nella risposta),
        # che è il più rilevante per il rerank tra i fratelli citati.
        #
        # cited_keys_v2 (per i marker viola nella matrice) resta granulare a
        # livello di chunk, perché la matrice mostra ogni chunk del top-N.
        cited_keys_v2: dict[tuple, list[int]] = {}
        for cit_n in citations_v2:
            if 1 <= cit_n <= len(reranked_v2):
                rc = reranked_v2[cit_n - 1]
                rk = (rc.get("filename"), rc.get("page_number"),
                      rc.get("chunk_index"))
                cited_keys_v2.setdefault(rk, []).append(cit_n)

        # cited_ordered: una entry per FILE unico (dedupe per filename), ordinati
        # per primo numero di citazione apparso.
        file_to_entry: dict[str, tuple[dict, list[int]]] = {}
        for cit_n in citations_v2:
            if not (1 <= cit_n <= len(reranked_v2)):
                continue
            rc = reranked_v2[cit_n - 1]
            file_key = rc.get("gridfs_file_id") or rc.get("filename") or f"_{cit_n}"
            if file_key in file_to_entry:
                # Aggiungo la citazione alla entry esistente (chunk repr resta il primo)
                existing_chunk, existing_cits = file_to_entry[file_key]
                if cit_n not in existing_cits:
                    file_to_entry[file_key] = (
                        existing_chunk, sorted(existing_cits + [cit_n])
                    )
            else:
                file_to_entry[file_key] = (rc, [cit_n])
        cited_ordered: list[tuple[dict, list[int]]] = list(file_to_entry.values())

        # 6. Ricerche profonde per Rank Journey e confronto con gli input RRF.
        journey_depth_v2 = max(_JOURNEY_TOP_K, top_k_v2)
        t0 = time.perf_counter()
        with st.spinner(f"Calcolo Rank Journey (top-{journey_depth_v2} per modalità)…"):
            text_deep_v2 = search(
                _coll(), query_v2, filters=filters_v2, top_k=journey_depth_v2,
                embed_mode=st.session_state.embed_mode, retrieval_mode="text",
                fuzzy=fuzzy_v2,
            )
            timings_ms_v2["text"] = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            vector_deep_v2 = search(
                _coll(), query_v2, filters=filters_v2, top_k=journey_depth_v2,
                embed_mode=st.session_state.embed_mode, retrieval_mode="vector",
                query_vector=query_vector_v2,
            )
            timings_ms_v2["vector"] = (time.perf_counter() - t0) * 1000

        # Riorganizzazione interattiva a Step per la demo (Semplificata & Visiva)
        st.divider()
        st.markdown("<h3 style='margin-top:0;'>🛠️ Come è stata costruita questa risposta?</h3>", unsafe_allow_html=True)
        st.caption("Esplora i singoli passaggi della nostra pipeline di ricerca e generazione per capire come i motori lavorano sotto il cofano.")
        receipt_v2 = _build_pipeline_receipt(
            query=query_v2,
            filters=filters_v2,
            top_k=top_k_v2,
            top_n=top_n_v2,
            fuzzy=fuzzy_v2,
            timings=timings_ms_v2,
            top_rerank=top_rerank_v2,
            threshold=relevance_threshold_v2,
            rerank_mode=rerank_mode_v2,
            provider=answer_provider_v2,
            model=answer_model_v2,
            citations=citations_v2,
            reranked=reranked_v2,
        )
        with st.expander("Query summary", expanded=False):
            st.markdown(
                "Parametri passati alla query e riepilogo copiabile della pipeline "
                "usata per generare questa risposta."
            )
            st.text_area(
                "Riepilogo copiabile",
                value=receipt_v2,
                height=210,
                key="pipeline_receipt_v2",
                label_visibility="collapsed",
            )
        
        tab_step1, tab_step2, tab_step3, tab_step4, tab_step5 = st.tabs([
            "🎯 1. Analisi Query", 
            "🔍 2. Ricerca Parallela", 
            "🔀 3. Fusione Ibrida (RRF)", 
            "🎯 4. Reranking Voyage", 
            "💬 5. Generazione LLM"
        ])
        
        with tab_step1:
            st.markdown("#### 🎯 Analisi della Query")
            st.markdown("La query dell'utente viene analizzata e tradotta in vettori numerici per la ricerca semantica.")
            
            query_tokens = len(_TOKENIZER.encode(query_v2))
            st.markdown(
                f'<div style="display:flex; gap:24px; flex-wrap:wrap; font-size:0.92em; margin-bottom:14px; background:rgba(128,128,128,0.03); padding:10px 14px; border-radius:8px; border:1px solid rgba(128,128,128,0.15);">'
                f'<div><b>Testo Query:</b> <span>"{html.escape(query_v2)}"</span></div>'
                f'<div><b>Lunghezza:</b> {query_tokens} Token ({len(query_v2)} char)</div>'
                f'<div><b>Modello Embedding:</b> Voyage AI ({st.session_state.embed_mode.upper()})</div>'
                f'</div>',
                unsafe_allow_html=True
            )

            if rerank_mode_v2 == "native":
                shell_pipeline_v2 = build_native_rerank_pipeline(
                    query_v2,
                    filters=filters_v2,
                    top_k=top_k_v2,
                    top_n=top_n_v2,
                    embed_mode=st.session_state.embed_mode,
                    fuzzy=fuzzy_v2,
                    query_vector=query_vector_v2,
                )
            else:
                shell_pipeline_v2 = build_pipeline(
                    query_v2,
                    filters=filters_v2,
                    top_k=top_k_v2,
                    embed_mode=st.session_state.embed_mode,
                    retrieval_mode="hybrid",
                    fuzzy=fuzzy_v2,
                    query_vector=query_vector_v2,
                )
            shell_query_v2 = (
                f"db.getCollection({json.dumps(config.MONGODB_COLLECTION)}).aggregate(\n"
                f"{json.dumps(shell_pipeline_v2, ensure_ascii=False, indent=2)}\n"
                f")"
            )

            with st.expander("👁️ Mostra query", expanded=False):
                if not query_vector_v2:
                    st.caption(
                        "In modalità auto MongoDB genera l'embedding direttamente "
                        "nell'indice a partire dal testo della domanda."
                    )
                if rerank_mode_v2 == "native":
                    st.success(
                        "Reranking eseguito dentro MongoDB con lo stage `$rerank`."
                    )
                st.markdown("**Pipeline MongoDB eseguita, pronta per `mongosh`**")
                st.code(shell_query_v2, language="javascript")

        with tab_step2:
            st.markdown("#### 🔍 Ricerca Parallela (BM25 vs Vector)")
            st.markdown("MongoDB Atlas esegue due ricerche simultanee sul database: una per parole esatte, una per concetto.")
            
            col_search_l, col_search_r = st.columns(2)
            with col_search_l:
                st.markdown('<h5 style="color:#fb8c00; margin-top:0;">🔤 Lessicale (BM25)</h5>', unsafe_allow_html=True)
                st.caption("Cerca corrispondenze esatte di parole. Ideale per codici e termini specifici.")
                if text_deep_v2:
                    for i, c in enumerate(text_deep_v2[:3], start=1):
                        score_val = c.get("score", 0.0)
                        fname = c.get("filename", "?")
                        page_lbl = _page_label(c.get("page_number"), missing="auto")
                        snippet = (c.get("content") or "").strip()
                        
                        st.markdown(
                            f'<div style="padding:8px 12px; background:rgba(251,140,0,0.05); border-left:3px solid #fb8c00; border-radius:6px; margin-bottom:8px">'
                            f'<span style="float:right; font-weight:bold; color:#fb8c00;">#{i} (Score: {score_val:.2f})</span>'
                            f'📄 <span style="font-size:0.9em; font-weight:600;">{html.escape(fname[:35])}...</span> · <small>{page_lbl}</small>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                        with st.expander(f"Visualizza testo #{i}", expanded=False):
                            highlighted_snippet = _highlight(snippet, query_v2)
                            st.markdown(f'<div style="font-size:0.9em; line-height:1.45; white-space:pre-wrap;">{highlighted_snippet}</div>', unsafe_allow_html=True)
                else:
                    st.caption("Nessun risultato.")
                    
            with col_search_r:
                st.markdown('<h5 style="color:#1e88e5; margin-top:0;">🧠 Semantica (Vector Search)</h5>', unsafe_allow_html=True)
                st.caption("Cerca per concetto e significato (Cosine Similarity HNSW). Ideale per sinonimi.")
                if vector_deep_v2:
                    for i, c in enumerate(vector_deep_v2[:3], start=1):
                        score_val = c.get("score", 0.0)
                        fname = c.get("filename", "?")
                        page_lbl = _page_label(c.get("page_number"), missing="auto")
                        snippet = (c.get("content") or "").strip()
                        
                        st.markdown(
                            f'<div style="padding:8px 12px; background:rgba(30,136,229,0.05); border-left:3px solid #1e88e5; border-radius:6px; margin-bottom:8px">'
                            f'<span style="float:right; font-weight:bold; color:#1e88e5;">#{i} (Affinità: {score_val * 100:.0f}% - Cosine Similarity)</span>'
                            f'📄 <span style="font-size:0.9em; font-weight:600;">{html.escape(fname[:35])}...</span> · <small>{page_lbl}</small>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                        with st.expander(f"Visualizza testo #{i}", expanded=False):
                            highlighted_snippet = _highlight(snippet, query_v2)
                            st.markdown(f'<div style="font-size:0.9em; line-height:1.45; white-space:pre-wrap;">{highlighted_snippet}</div>', unsafe_allow_html=True)
                else:
                    st.caption("Nessun risultato.")

        with tab_step3:
            st.markdown("#### 🔀 Fusione Ibrida Nativa ($rankFusion)")
            st.markdown("Combina i due canali (Lessicale e Semantico) usando la formula **Reciprocal Rank Fusion** nativa di MongoDB Atlas.")

            text_input_v2 = text_deep_v2[:top_k_v2]
            vector_input_v2 = vector_deep_v2[:top_k_v2]
            text_input_keys_v2 = {_result_key(c) for c in text_input_v2}
            vector_input_keys_v2 = {_result_key(c) for c in vector_input_v2}
            common_input_v2 = len(text_input_keys_v2 & vector_input_keys_v2)
            counts_v2 = _source_counts(raw_v2)

            st.markdown("##### Cosa ha cambiato `$rankFusion`")
            st.markdown(
                f'<div style="display:grid;grid-template-columns:minmax(140px,1fr) auto '
                f'minmax(140px,1fr) auto minmax(210px,1.35fr);gap:10px;align-items:stretch;'
                f'margin:6px 0 14px 0">'
                f'<div style="border-left:3px solid #fb8c00;background:rgba(251,140,0,0.06);'
                f'padding:10px;border-radius:6px"><b>Keyword</b><br>'
                f'<span style="font-size:1.15em">{len(text_input_v2)} candidati</span></div>'
                f'<div style="display:flex;align-items:center;font-size:1.3em">+</div>'
                f'<div style="border-left:3px solid #1e88e5;background:rgba(30,136,229,0.06);'
                f'padding:10px;border-radius:6px"><b>Semantica</b><br>'
                f'<span style="font-size:1.15em">{len(vector_input_v2)} candidati</span></div>'
                f'<div style="display:flex;align-items:center;font-size:1.3em">→</div>'
                f'<div style="border-left:3px solid #7e57c2;background:rgba(126,87,194,0.07);'
                f'padding:10px;border-radius:6px"><b>Top-{len(raw_v2)} fusa</b><br>'
                f'<span style="font-size:0.9em">'
                f'{counts_v2["text_only"]} solo keyword · '
                f'{counts_v2["vector_only"]} solo semantici · '
                f'{counts_v2["both"]} condivisi</span></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.caption(
                f"Le due liste contengono {common_input_v2} chunk in comune. "
                f"`$rankFusion` non concatena 50 + 50 risultati: confronta le posizioni, "
                f"premia il consenso e restituisce soltanto i migliori {len(raw_v2)}."
            )

            st.markdown("##### Movimento dei risultati")
            st.caption(
                "Ogni riga mostra il contributo reale dei due motori e la posizione "
                "assegnata da RRF. La variazione è calcolata rispetto alla posizione "
                "migliore ottenuta dal chunk nei ranking di partenza."
            )
            if raw_v2:
                for i, c in enumerate(raw_v2[:8], start=1):
                    key_v2 = _result_key(c)
                    text_rank_v2 = _position_in(text_input_v2, key_v2)
                    vector_rank_v2 = _position_in(vector_input_v2, key_v2)
                    input_ranks_v2 = [
                        rank for rank in (text_rank_v2, vector_rank_v2)
                        if rank is not None
                    ]
                    best_input_rank_v2 = min(input_ranks_v2) if input_ranks_v2 else None
                    source_type = _chunk_source(c)

                    if best_input_rank_v2 is None:
                        movement_v2 = "Nuovo"
                        movement_color_v2 = "#7e57c2"
                    elif i < best_input_rank_v2:
                        movement_v2 = f"▲ guadagna {best_input_rank_v2 - i}"
                        movement_color_v2 = "#2e7d32"
                    elif i > best_input_rank_v2:
                        movement_v2 = f"▼ perde {i - best_input_rank_v2}"
                        movement_color_v2 = "#c62828"
                    else:
                        movement_v2 = "= stabile"
                        movement_color_v2 = "#616161"

                    if source_type == "both":
                        contribution_v2 = "Consenso keyword + semantica"
                    elif source_type == "text_only":
                        contribution_v2 = "Recuperato dal ramo keyword"
                    elif source_type == "vector_only":
                        contribution_v2 = "Recuperato dal ramo semantico"
                    else:
                        contribution_v2 = "Origine non disponibile"

                    rrf_score = c.get("score", 0.0)
                    fname = c.get("filename", "?")
                    page_lbl = _page_label(c.get("page_number"), missing="auto")
                    snippet = (c.get("content") or "").strip()
                    text_rank_label_v2 = f"#{text_rank_v2}" if text_rank_v2 else "—"
                    vector_rank_label_v2 = f"#{vector_rank_v2}" if vector_rank_v2 else "—"
                    
                    st.markdown(
                        f'<div style="padding:10px 12px;background:rgba(126,87,194,0.035);'
                        f'border-left:3px solid #7e57c2;border-radius:6px;margin-bottom:8px">'
                        f'<div style="display:flex;justify-content:space-between;gap:12px;'
                        f'align-items:center;flex-wrap:wrap">'
                        f'<div><b>{html.escape(fname[:48])}</b> · <small>{page_lbl}</small></div>'
                        f'<div style="font-weight:750;color:{movement_color_v2}">{movement_v2}</div>'
                        f'</div>'
                        f'<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;'
                        f'margin-top:8px;font-size:0.9em">'
                        f'<span style="background:rgba(251,140,0,0.12);padding:3px 9px;'
                        f'border-radius:12px">Keyword <b>{text_rank_label_v2}</b></span>'
                        f'<span style="font-size:1.05em">+</span>'
                        f'<span style="background:rgba(30,136,229,0.12);padding:3px 9px;'
                        f'border-radius:12px">Semantica <b>{vector_rank_label_v2}</b></span>'
                        f'<span style="font-size:1.05em">→</span>'
                        f'<span style="background:#7e57c2;color:white;padding:3px 10px;'
                        f'border-radius:12px">RRF <b>#{i}</b></span>'
                        f'<span style="opacity:0.72">{contribution_v2}</span>'
                        f'<span style="margin-left:auto;opacity:0.65">score {rrf_score:.5f}</span>'
                        f'</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    with st.expander(f"Visualizza il chunk RRF #{i}", expanded=False):
                        highlighted_snippet = _highlight(snippet, query_v2)
                        st.markdown(f'<div style="font-size:0.9em; line-height:1.45; white-space:pre-wrap;">{highlighted_snippet}</div>', unsafe_allow_html=True)
            
            # Vista Pulita rimossa per semplificazione
            pass

        with tab_step4:
            st.markdown("#### 🎯 Rerank di Precisione (Voyage AI)")
            st.markdown(
                "I candidati ibridi vengono analizzati da Voyage Reranker (modello Cross-Encoder) che confronta congiuntamente "
                "la query con l'intero contesto dei chunk per stabilire l'ordinamento definitivo."
            )
            if rerank_mode_v2 == "native":
                st.success(
                    "Eseguito nativamente nell'aggregation framework MongoDB "
                    "tramite `$rerank`."
                )
            
            # Compact markdown table for shifts
            st.markdown("##### ⚖️ Prima (Hybrid) vs Dopo (Reranker)")
            
            comparison_rows = []
            for rank_after, c in enumerate(reranked_v2[:5], start=1):
                k = (c.get("filename"), c.get("page_number"), c.get("chunk_index"))
                rank_before = _position_in(raw_v2, k)
                score_val = c.get("_rerank_score")
                fname = c.get("filename", "?")
                page_lbl = _page_label(c.get("page_number"), missing="auto")
                
                if rank_before is None:
                    change_str = "🆕 Nuovo"
                    change_color = "green"
                elif rank_before > rank_after:
                    diff = rank_before - rank_after
                    change_str = f"▲ +{diff}"
                    change_color = "green"
                elif rank_before < rank_after:
                    diff = rank_after - rank_before
                    change_str = f"▼ -{diff}"
                    change_color = "red"
                else:
                    change_str = "="
                    change_color = "gray"
                    
                score_str = f"{score_val:.3f}" if isinstance(score_val, (int, float)) else "—"
                comparison_rows.append(
                    f"| **#{rank_after}** | {html.escape(fname[:35])}... ({page_lbl}) | #{rank_before if rank_before else '—'} | <span style='color:{change_color};font-weight:bold;'>{change_str}</span> | `{score_str}` |"
                )
                
            headers_md = (
                "| Rerank | Documento / Fonte | RRF Ibrido | Variazione | Score Rerank |\n"
                "| :---: | :--- | :---: | :---: | :---: |\n"
            )
            st.markdown(headers_md + "\n".join(comparison_rows), unsafe_allow_html=True)
            
            st.markdown(
                f"<div style='margin-top:14px; padding:10px 14px; background:rgba(30,136,229,0.06); border-radius:6px; font-size:0.88em; border-left:4px solid #1e88e5;'>"
                f"🎯 **Soglia di Rilevanza**: il sistema considera 'off-topic' qualsiasi chunk con score &lt; `{relevance_threshold_v2:.2f}`."
                f"</div>",
                unsafe_allow_html=True
            )
            
            # Rank Journey (collapsed inside an expander)
            with st.expander("💬 Dettaglio del percorso (Rank Journey) delle fonti citate"):
                _render_rank_journey(
                    cited_chunks_with_cit=cited_ordered,
                    text_deep=text_deep_v2,
                    vector_deep=vector_deep_v2,
                    hybrid_deep=raw_v2,
                    rerank_results=reranked_v2,
                    query=query_v2,
                )

        with tab_step5:
            st.markdown("#### 💬 Generazione Risposta Grounded")
            st.markdown("I chunk riordinati vengono passati al LLM che formula la risposta citando solo le fonti verificate.")
            
            st.markdown(
                f'<div style="display:flex; gap:24px; flex-wrap:wrap; font-size:0.92em; margin-bottom:14px; background:rgba(128,128,128,0.03); padding:10px 14px; border-radius:8px; border:1px solid rgba(128,128,128,0.15);">'
                f'<div><b>Modello LLM:</b> <span>{config.LLM_PROVIDER.upper()}</span></div>'
                f'<div><b>Rerank Score Massimo:</b> <span>{top_disp}</span></div>'
                f'</div>',
                unsafe_allow_html=True
            )

            # Prompt template (collapsed)
            with st.expander("📁 Visualizza Struttura del Prompt", expanded=False):
                if reranked_v2:
                    first_content_v2 = reranked_v2[0].get("content") or ""
                    st.code(
                        f"SYSTEM:\n{config.LLM_SYSTEM_PROMPT}\n\n"
                        f"USER:\nDOMANDA: {query_v2}\n\n"
                        f"[#1] file={reranked_v2[0].get('filename')}\n"
                        f"{first_content_v2[:150]}...",
                        language="text"
                    )
                else:
                    st.caption("Nessun passaggio disponibile per costruire il prompt.")
                
            with st.expander("📊 Visualizza Matrice di Tracciabilità Completa"):
                _render_comparison_matrix(
                    text_deep=text_deep_v2,
                    vector_deep=vector_deep_v2,
                    hybrid_deep=raw_v2,
                    rerank_results=reranked_v2,
                    cited_keys=cited_keys_v2,
                )


# ─── ADMIN ─────────────────────────────────────────────────────────────────────
with tab_admin:
    st.subheader("Modalità embedding")
    st.caption(
        "**auto** — Atlas genera e mantiene gli embedding direttamente dall'indice: "
        "l'applicazione salva il testo e MongoDB mantiene automaticamente allineata "
        "la ricerca semantica, senza pipeline esterne di embedding.  \n"
        "**manual** — l'app genera e salva l'embedding di ogni chunk: offre controllo "
        "esplicito sul modello e sul ciclo di indicizzazione.  \n"
        "**context-4** — Voyage genera ogni embedding usando anche il contesto degli "
        "altri chunk dello stesso documento: migliora la comprensione di riferimenti "
        "e contenuti simili, conservando pagine e tipi di contenuto."
    )
    _mode_options = ["auto", "manual", "context"]
    new_mode = st.radio(
        "Modalità",
        options=_mode_options,
        index=_mode_options.index(st.session_state.embed_mode)
              if st.session_state.embed_mode in _mode_options else 0,
        horizontal=True, key="embed_mode_radio",
        format_func=lambda x: {
            "auto":    "🤖 auto",
            "manual":  "✋ manual",
            "context": "🧬 context-4",
        }[x],
    )
    if new_mode != st.session_state.embed_mode:
        st.session_state.embed_mode = new_mode
        _health_check.clear()  # forza rinfresco del semaforo dopo il cambio
        # Memorizzo il messaggio di successo per mostrarlo dopo il rerun
        # (senza rerun la pagina mostrerebbe ancora la modalità vecchia in
        # cima — title caption, semaforo — perché il render dell'header
        # gira prima di questo blocco in Admin).
        st.session_state.mode_changed_msg = (
            f"Modalità impostata a `{new_mode}` per questa sessione. "
            "**Drop & recreate** dell'indice è necessario se cambi "
            "dimensione vettore o tipo di indice."
        )
        st.rerun()

    # Mostra eventuale messaggio sopravvissuto al rerun (cambio modalità)
    if msg := st.session_state.pop("mode_changed_msg", None):
        st.success(msg)

    st.divider()
    st.subheader("Indici Atlas Search")
    st.caption(
        "**Drop & recreate** elimina l'indice esistente e lo ricrea con la "
        "definizione corrente (necessario dopo cambio modello o modalità)."
    )
    recreate = st.checkbox("Drop & recreate (sovrascrive indici esistenti)", value=False)
    if st.button("Verifica / Crea indici", type="primary"):
        try:
            info = ensure_indexes(_coll(), mode=st.session_state.embed_mode,
                                  recreate=recreate)
            st.success("✅ Operazione completata.")
            st.json(info)
            # Invalida cache health check così il semaforo riflette subito i
            # nuovi indici (status INITIAL_SYNC → READY nei prossimi secondi)
            _health_check.clear()
        except Exception as e:
            err_msg = str(e)
            st.error(f"❌ Errore durante l'operazione sugli indici:\n\n`{err_msg}`")
            # Hint contestuali sulle cause più comuni
            if st.session_state.embed_mode == "auto" and (
                "M30" in err_msg or "Compute Auto-Scale" in err_msg
                or "IllegalOperation" in err_msg
            ):
                st.info(
                    "La modalità `auto` non è disponibile nella configurazione "
                    "Atlas corrente. Seleziona `manual` o `context-4`, oppure "
                    "verifica le funzionalità abilitate sul cluster."
                )
            elif "Duplicate" in err_msg or "already exists" in err_msg:
                st.info(
                    "ℹ L'indice esiste già con lo stesso nome. Spunta "
                    "**Drop & recreate** sopra per sovrascriverlo."
                )
            elif "BadValue" in err_msg or "Unsupported" in err_msg:
                st.info(
                    "ℹ Schema indice rifiutato da Atlas. Verifica nei file "
                    "`schemas/*.json` che i `type` siano corretti per la "
                    "tua versione di MongoDB Atlas (8.1+ richiesto per "
                    "autoEmbed e $rankFusion)."
                )

    st.divider()
    st.subheader("Stato collection")
    if st.button("Refresh metriche"):
        coll = _coll()
        st.metric("Chunk totali (stimati)", coll.estimated_document_count())
        st.metric("Documenti distinti", len(coll.distinct("document_id")))

    st.divider()
    st.subheader("📂 Documenti caricati")
    st.caption(
        "Lista dei documenti distinti in collection (aggregati per `document_id`). "
        "La cancellazione rimuove **tutti i chunk** del documento + il **file GridFS** "
        "associato. L'operazione **non** tocca gli indici Atlas Search."
    )

    docs = _document_list()
    total_chunks = sum(d["chunks"] for d in docs)
    col_count, col_refresh = st.columns([0.7, 0.3])
    col_count.markdown(
        f"**{len(docs)}** documenti distinti · **{total_chunks}** chunk totali"
    )
    if col_refresh.button("🔄 Refresh lista", key="refresh_doc_list",
                          use_container_width=True):
        _document_list.clear()
        st.rerun()

    if not docs:
        st.info("Nessun documento in collection. Vai al tab **📤 Upload** per inserirne uno.")
    else:
        # ── Cancella TUTTI: pattern di conferma a due step ───────────────────
        if st.session_state.get("pending_delete_all"):
            st.error(
                f"⚠ **Confermi la cancellazione di TUTTI i documenti?**  \n"
                f"Verranno rimossi **{len(docs)}** documenti · **{total_chunks}** chunk · "
                f"file GridFS associati. **Operazione irreversibile.**"
            )
            c_yes, c_no = st.columns(2)
            if c_yes.button("✅ Sì, cancella tutto", type="primary",
                            use_container_width=True, key="confirm_delete_all"):
                coll = _coll()
                n_chunks = coll.count_documents({})
                coll.delete_many({})
                # Sweep GridFS bucket
                gridfs_deleted = 0
                try:
                    fs = GridFS(_db(), collection=config.MONGODB_GRIDFS_BUCKET)
                    for f in fs.find():
                        fs.delete(f._id)
                        gridfs_deleted += 1
                except Exception as e:
                    st.warning(f"Errore pulizia GridFS: {e}")
                st.session_state.pop("pending_delete_all", None)
                st.session_state.pop("last_ingestion", None)
                _document_list.clear()
                _available_metadata_filters.clear()
                _distinct_filenames.clear()
                _dominant_doc_langs.clear()
                _gridfs_pdf_bytes.clear()
                st.success(
                    f"Cancellati {n_chunks} chunk · {gridfs_deleted} file GridFS rimossi."
                )
                st.rerun()
            if c_no.button("❌ Annulla", use_container_width=True,
                           key="cancel_delete_all"):
                st.session_state.pop("pending_delete_all", None)
                st.rerun()
        else:
            if st.button("🗑 Cancella TUTTI i documenti", type="secondary",
                         use_container_width=True, key="ask_delete_all"):
                st.session_state.pending_delete_all = True
                st.rerun()

        st.divider()

        # ── Cancellazione singola: lista con conferma a due step ─────────────
        for d in docs:
            doc_id = d["_id"]
            pending = st.session_state.get("pending_delete_doc") == doc_id
            with st.container(border=True):
                c_info, c_btn = st.columns([0.72, 0.28])
                with c_info:
                    name = d.get("title") or d.get("filename") or "(senza nome)"
                    fname = d.get("filename") or "—"
                    st.markdown(f"**📄 {name}**")
                    ingested = d.get("ingested_at")
                    ingested_str = (
                        ingested.strftime("%Y-%m-%d %H:%M")
                        if ingested and hasattr(ingested, "strftime") else "—"
                    )
                    bits = [
                        f"`{d['chunks']}` chunk",
                        f"ingested: `{ingested_str}`",
                    ]
                    if d.get("author"):
                        bits.append(f"author: `{d['author']}`")
                    if d.get("file_type"):
                        bits.append(f"type: `{d['file_type']}`")
                    if d.get("total_pages"):
                        bits.append(f"`{d['total_pages']}` pagine")
                    if d.get("gridfs_file_id"):
                        bits.append("📦 GridFS")
                    bits.append(f"file: `{fname}`")
                    bits.append(f"doc_id: `{doc_id[:12]}…`")
                    st.caption(" · ".join(bits))
                with c_btn:
                    if pending:
                        cy, cn = st.columns(2)
                        if cy.button("✅", help="Conferma cancellazione",
                                     use_container_width=True,
                                     key=f"confirm_del_{doc_id}"):
                            coll = _coll()
                            n_chunks_doc = coll.count_documents({"document_id": doc_id})
                            coll.delete_many({"document_id": doc_id})
                            gridfs_id = d.get("gridfs_file_id")
                            if gridfs_id:
                                try:
                                    fs = GridFS(_db(),
                                                collection=config.MONGODB_GRIDFS_BUCKET)
                                    fs.delete(ObjectId(gridfs_id))
                                except Exception as e:
                                    st.warning(f"GridFS delete: {e}")
                            st.session_state.pop("pending_delete_doc", None)
                            _document_list.clear()
                            _available_metadata_filters.clear()
                            _distinct_filenames.clear()
                            _dominant_doc_langs.clear()
                            _gridfs_pdf_bytes.clear()
                            st.success(
                                f"Cancellati {n_chunks_doc} chunk di `{name}`."
                            )
                            st.rerun()
                        if cn.button("✖", help="Annulla", use_container_width=True,
                                     key=f"cancel_del_{doc_id}"):
                            st.session_state.pop("pending_delete_doc", None)
                            st.rerun()
                    else:
                        if st.button("🗑 Elimina", use_container_width=True,
                                     key=f"del_{doc_id}"):
                            st.session_state.pending_delete_doc = doc_id
                            st.rerun()
