"""Mini frontend Streamlit per la demo RAG.

Layout:
- 📖 How it Works → architettura + valore della pipeline
- 📤 Upload  → ingestion documenti (PDF, DOCX, XLSX, PPTX, TXT, ...)
- 🔎 Query   → ricerca ibrida, riordino dei risultati e risposta con fonti
- 🧭 Ricerca agentica → query iterative con memoria di sessione persistente
- ⚙ Admin   → indici Atlas, metriche, modalità embedding

Run:  streamlit run app.py
"""
from __future__ import annotations

import html
import importlib
import json
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tiktoken
import streamlit as st
import streamlit.components.v1 as components
from bson import ObjectId
from gridfs import GridFS

# Streamlit riesegue la pagina nello stesso processo: un file nuovo in src/ (aggiunto mentre
# l'app gira, anche da una cartella di rete) non verrebbe visto senza svuotare la cache degli import.
importlib.invalidate_caches()
from src import config
from src.agentic_memory import (
    add_event as _agent_add_event,
    create_session as _agent_create_session,
    fail_run as _agent_fail_run,
    finish_run as _agent_finish_run,
    get_agent_collections as _get_agent_collections,
    record_feedback as _agent_record_feedback,
    refinement_history as _agent_refinement_history,
    start_run as _agent_start_run,
    validate_refinement as _agent_validate_refinement,
)
from src.agentic_search import (
    assess_evidence as _agent_assess_evidence,
    compose_search_query as _agent_compose_search_query,
    decide_next_search as _agent_decide_next_search,
    model_label as _agent_model_label,
    plan_search as _agent_plan_search,
    compact_text as _agent_compact_text,
    find_year_ambiguity as _agent_find_year_ambiguity,
    retrieve_candidates as _agent_retrieve_candidates,
    rerank_union as _agent_rerank_union,
)
from src.generation import llm_available
from src.document_inventory import document_inventory
from src.answer_quality import (answer_diagnostics, best_publishable_attempt, explain_answer_check, generate_reviewed_answer,
                                generate_single_answer, inspect_answer)
from src.indexes import ensure_indexes, index_plan, index_command
from src.ingestion import ingest_pdf
from src.init_db import ensure_document_lookup_index, initialize_database, repair_document_page_references
from src.rerank import rerank, format_rerank_query
from src.retrieval_details import passage_key, branch_ranks, semantic_ranks
from src.rank_flow import rank_flow_html, short_file, snippet
from src.citation_check import check_citations, summarize as summarize_citation_check
from src.agent_value import agent_value_html
from src.search_scope import (MODE_DESCRIPTIONS, chunk_counts_by_mode, load_active_mode,
                              preview_scope, require_scope, prepare_vectors, render_scope,
                              pipeline_code, save_active_mode, weights_caption, dedup_caption)
from src.retrieval import (
    build_pipeline,
    build_native_rerank_pipeline,
    search,
)

_TOKENIZER = tiktoken.get_encoding("cl100k_base")


# =================================================================== setup ====

st.set_page_config(page_title="RAG Demo — MongoDB + Voyage", layout="wide")

if not config.MONGODB_URI or "cluster.mongodb.net" in config.MONGODB_URI:
    st.title("RAG Demo — MongoDB + Voyage")
    st.warning("Configurazione da completare: inserisci la connessione MongoDB Atlas nel file .env.")
    st.info("Configura MONGODB_URI, VOYAGE_API_KEY e la chiave del provider LLM selezionato, poi riavvia l’app.")
    st.stop()

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

# Modalità attiva: una sola alla volta, salvata in MongoDB. L'Upload carica in
# questa modalità; Query e Ricerca agentica cercano solo qui. Riletta a ogni
# esecuzione, così un cambio fatto in un'altra scheda vale anche qui.
st.session_state.embed_mode = load_active_mode(st.session_state.get("embed_mode", config.EMBED_MODE))


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
            active_vector_index = config.vector_index_name(embed_mode)
            missing = [
                n for n in (active_vector_index, config.SEARCH_INDEX_NAME)
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
                        f"READY: `{active_vector_index}`, "
                        f"`{config.SEARCH_INDEX_NAME}`",
                    )
                # Coerenza schema indice ↔ embed_mode (anche se status READY)
                vidx = present.get(active_vector_index)
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
                        expected = ("autoEmbed", "content") if embed_mode == "auto" else (
                            "vector", {"manual": "embedding", "context": "embedding_context",
                                       "multimodal": "embedding_multimodal"}[embed_mode]
                        )
                        if (idx_type, idx_path) != expected:
                            out["Schema indice vector"] = (
                                "error",
                                f"incoerente: indice ha `type={idx_type}` "
                                f"`path={idx_path}` ma `EMBED_MODE={embed_mode}` "
                                f"richiede `type={expected[0]}` `path={expected[1]}`. "
                                f"Drop & recreate dal tab Admin + re-ingest.",
                            )
                        else:
                            expected_dims = {
                                "manual": config.VOYAGE_EMBED_DIMENSIONS,
                                "context": config.VOYAGE_CONTEXT_DIMENSIONS,
                                "multimodal": config.VOYAGE_MULTIMODAL_DIMENSIONS,
                            }.get(embed_mode)
                            idx_dims = vec_field.get("numDimensions")
                            if expected_dims is not None and idx_dims != expected_dims:
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
    needs_voyage = embed_mode in ("manual", "context", "multimodal")
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
            "multimodal": "voyage-multimodal-3.5 per pagina/slide",
        }.get(embed_mode, embed_mode)
        used_model = {
            "auto": config.VOYAGE_EMBED_MODEL,
            "manual": config.VOYAGE_EMBED_MODEL,
            "context": config.VOYAGE_CONTEXT_MODEL,
            "multimodal": config.VOYAGE_MULTIMODAL_MODEL,
        }[embed_mode]
        out["Embed mode"] = ("ok", f"{mode_label} · model `{used_model}`")

    # 4. La chiave API è separata dal reranking nativo autenticato via Atlas.
    if config.VOYAGE_API_KEY:
        endpoint_host = config.VOYAGE_API_BASE.replace("https://", "").split("/")[0]
        out["Voyage AI"] = (
            "ok",
            f"chiave configurata · endpoint `{endpoint_host}` · "
            f"rerank `{config.VOYAGE_RERANK_MODEL}` · percorso `{config.RERANK_BACKEND}`",
        )
    else:
        out["Voyage AI"] = (
            "ok" if config.RERANK_BACKEND == "native" and embed_mode == "auto" else "warning",
            "Reranking via Atlas $rerank; chiave API assente per embedding manual/context/multimodal."
            if config.RERANK_BACKEND == "native" else
            "`VOYAGE_API_KEY` mancante — il rerank API verrà saltato in Query",
        )

    # 5. LLM provider — usa l'helper esistente
    provider = config.LLM_PROVIDER
    model_per_provider = {
        "anthropic": config.ANTHROPIC_MODEL,
        "ollama": config.OLLAMA_MODEL,
        "gemini": config.GEMINI_MODEL,
        "gateway": config.AI_GATEWAY_MODEL,
    }
    if llm_available():
        model = model_per_provider.get(provider, "?")
        out[f"LLM ({provider})"] = ("ok", f"raggiungibile · model `{model}`")
    else:
        reason = {
            "anthropic": "`ANTHROPIC_API_KEY` mancante",
            "gemini": "`GEMINI_API_KEY` mancante",
            "gateway": "`AI_GATEWAY_API_KEY` mancante",
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
    "multimodal": "multimodal testo + visuale",
}.get(st.session_state.embed_mode, st.session_state.embed_mode)
active_embed_model = {
    "auto": config.VOYAGE_EMBED_MODEL,
    "manual": config.VOYAGE_EMBED_MODEL,
    "context": config.VOYAGE_CONTEXT_MODEL,
    "multimodal": config.VOYAGE_MULTIMODAL_MODEL,
}.get(st.session_state.embed_mode, config.VOYAGE_EMBED_MODEL)
llm_label = {
    "anthropic": f"Anthropic `{config.ANTHROPIC_MODEL}`",
    "ollama":    f"Ollama `{config.OLLAMA_MODEL}`",
    "gemini":    f"Gemini `{config.GEMINI_MODEL}`",
    "gateway":   f"AI Gateway `{config.AI_GATEWAY_MODEL}`",
}.get(config.LLM_PROVIDER, f"Sconosciuto `{config.LLM_PROVIDER}`")
st.caption(
    f"DB `{config.MONGODB_DB}` · Collection `{config.MONGODB_COLLECTION}` · "
    f"Modalità attiva **{mode_badge}** (Upload, Query e Ricerca agentica) · Modello `{active_embed_model}` · "
    f"Rerank `{config.VOYAGE_RERANK_MODEL}` ({config.RERANK_BACKEND}) · LLM {llm_label}"
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
                       "detection: se il punteggio rerank è sotto la soglia, "
                       "l'app non chiama il modello generativo.",
        "demo_focus": "Mostra un controllo che riduce le risposte non supportate: la soglia non garantisce che ogni risposta sopra soglia sia corretta.",
        "badges": ["soglia rerank", "controllo off-topic", "guardrail"],
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
    ranks = branch_ranks(c)
    has_vec = any(name == "vector" or str(name).startswith("vector_") for name in ranks)
    has_txt = "text" in ranks
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
    return (passage_key(c),)


_TIMING_LABELS = {
    "Embedding domanda": ("Embedding della domanda", "#1D9E75"),
    "Ricerca + rerank Atlas": ("Ricerca + reranking in Atlas", "#0F6E56"),
    "hybrid": ("Ricerca ibrida in Atlas", "#0F6E56"),
    "rerank": ("Reranking Voyage API", "#5DCAA5"),
    "Generazione": ("Generazione", "#AFA9EC"),
}


def _format_ms(ms: float) -> str:
    return f"{ms:.0f} ms" if ms < 1000 else f"{ms / 1000:.1f} s".replace(".", ",")


def _render_timings_bar(timings: dict[str, float], provider: str, model: str, usage: dict) -> None:
    """Barra dei tempi per fase della risposta, con modello e token."""
    phases = [(_TIMING_LABELS[key], ms) for key, ms in timings.items()
              if key in _TIMING_LABELS and ms >= 1]
    total = sum(ms for _, ms in phases)
    tokens = ""
    if usage.get("input_tokens") is not None:
        tokens = (f" · {usage.get('input_tokens', 0):,} token in · {usage.get('output_tokens', 0):,} out"
                  .replace(",", "."))
    model_line = f"{html.escape(provider)} · {html.escape(model)}{tokens}"
    if not phases:
        st.caption(model_line)
        return
    bars = "".join(
        f'<div title="{html.escape(label)} {_format_ms(ms)}" '
        f'style="width:{max(ms / total * 100, 1.5):.2f}%;background:{color}"></div>'
        for (label, color), ms in phases
    )
    legend = "".join(
        f'<span style="white-space:nowrap"><span style="display:inline-block;width:9px;height:9px;'
        f'border-radius:50%;background:{color};margin-right:5px"></span>'
        f'{html.escape(label)} {_format_ms(ms)}</span>'
        for (label, color), ms in phases
    )
    st.markdown(
        f'<div style="font-size:0.85em;opacity:0.8;margin:14px 0 6px">Tempi di questa esecuzione · '
        f'{_format_ms(total)} in totale</div>'
        f'<div style="display:flex;height:20px;border-radius:6px;overflow:hidden;margin-bottom:8px">{bars}</div>'
        f'<div style="display:flex;gap:18px;flex-wrap:wrap;font-size:0.85em;opacity:0.8">{legend}'
        f'<span>{model_line}</span></div>',
        unsafe_allow_html=True,
    )


_CHECK_BADGES = {
    "found": ("✓ trovato", "rgba(29,158,117,0.16)", "#0F6E56"),
    "rounded": ("≈ arrotondato", "rgba(29,158,117,0.10)", "#0F6E56"),
    "not_found": ("⚠ non trovato", "rgba(239,159,39,0.18)", "#854F0B"),
    "uncited": ("senza citazione", "rgba(128,128,128,0.14)", "inherit"),
}


def _render_citation_check(answer_text: str, passages: list[dict], *, rows: list[dict] | None = None,
                           title: bool = True) -> None:
    """Per ogni numero della risposta: compare nel passaggio citato?"""
    rows = check_citations(answer_text, passages) if rows is None else rows
    if title:
        st.markdown("### Verifica delle citazioni")
    if not rows:
        st.caption("La risposta non contiene importi o percentuali da verificare: "
                   "il controllo confronta solo i numeri con i passaggi citati.")
        return
    counts = summarize_citation_check(rows)
    checked = len(rows) - counts["uncited"]
    ok = counts["found"] + counts["rounded"]
    summary = f"{ok} numeri su {checked} trovati nel passaggio citato" if checked else "Nessun numero ha una citazione"
    if counts["uncited"]:
        summary += f" · {counts['uncited']} senza citazione"
    st.caption(summary + ". Controllo meccanico, senza un altro modello: ogni numero della risposta viene "
               "cercato nel testo del passaggio citato nella stessa frase (o nello stesso paragrafo).")
    dark = _theme_is_dark()
    lines = []
    for row in rows:
        label, bg, fg = _CHECK_BADGES[row["status"]]
        if dark and fg != "inherit":
            fg = "#9FE1CB" if row["status"] in {"found", "rounded"} else "#FAC775"
        ref = row["ref"]
        if ref:
            passage = passages[ref - 1]
            where = (f"#{ref} · {html.escape(short_file(passage.get('filename') or 'Documento'))} · "
                     f"{html.escape(_page_label(passage.get('page_number', passage.get('page')), prefix='p. '))}")
            if len(row["refs"]) > 1 and row["status"] == "not_found":
                where = "Passaggi " + ", ".join(f"#{n}" for n in row["refs"])
        else:
            where = "Nessun passaggio citato"
        if row["excerpt"]:
            ex = row["excerpt"]
            detail = (f'“{html.escape(ex["before"])}<mark style="background:rgba(250,199,117,0.45);'
                      f'color:inherit;padding:0 2px;border-radius:3px">{html.escape(ex["match"])}</mark>'
                      f'{html.escape(ex["after"])}”')
            if row["status"] == "rounded":
                detail += " <i>Il valore della risposta è arrotondato rispetto al documento.</i>"
        elif row["status"] == "not_found":
            detail = ("Il numero non compare nel passaggio citato: può essere calcolato dal modello "
                      "o provenire da un altro passaggio. Verifica la fonte prima di usarlo.")
        else:
            detail = "La frase non cita un passaggio: il numero non è verificabile."
        lines.append(
            f'<div style="display:grid;grid-template-columns:minmax(90px,130px) minmax(0,1fr) auto;gap:12px;'
            f'align-items:start;padding:9px 0;border-top:1px solid rgba(128,128,128,0.18)">'
            f'<div style="font-weight:600">{html.escape(row["claim"])}</div>'
            f'<div><div style="opacity:0.7;font-size:0.9em">{where}</div>'
            f'<div style="opacity:0.8;font-size:0.88em;margin-top:2px">{detail}</div></div>'
            f'<div><span style="background:{bg};color:{fg};font-size:0.8em;padding:3px 10px;'
            f'border-radius:6px;white-space:nowrap">{label}</span></div></div>'
        )
    st.markdown(
        '<div style="border:1px solid rgba(128,128,128,0.22);border-radius:10px;padding:4px 14px 6px;'
        f'margin-bottom:6px">{"".join(lines)}</div>',
        unsafe_allow_html=True,
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
        f"Rerank: Voyage {config.VOYAGE_RERANK_MODEL} ({rerank_mode}), "
        f"top_score={top_s}, soglia_off_topic={threshold:.2f}",
        f"LLM: {provider} / {model}",
        f"Timing: {_timing_summary(timings)}",
        f"Fonti citate: {', '.join(cited_sources) if cited_sources else 'nessuna'}",
        "Valore demo: una sola piattaforma per chunk, metadati, full-text search, vector search, ranking e fonti verificabili.",
    ]
    return "\n".join(lines)


def _page_label(page, *, prefix: str = "p.", missing: str = "pagina non disponibile") -> str:
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
                    on_click="ignore",
                    width="stretch",
                )



def _parse_citations(text: str) -> list[int]:
    if not text:
        return []
    return inspect_answer(text, 10**9)["valid_citations"]


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

# Domande di esempio, usate da Query e Ricerca agentica.
demo_questions = [
    "Dove si trovano gli stabilimenti produttivi di 3M?",
    "Quanti stabilimenti produttivi aveva 3M nel 2018?",
    "Quali sono state le spese in conto capitale di 3M nel 2018, in milioni di dollari? Usa il rendiconto finanziario.",
    "Quanto ha speso Adobe per ricerca e sviluppo nell’esercizio 2022? Indica l’importo in milioni di dollari.",
    "Quali sono stati i ricavi e l’utile operativo del segmento AWS di Amazon nel 2022? Riporta entrambi in milioni di dollari.",
    "Quali sono state le spese in conto capitale di 3M nel 2017, in milioni di dollari? Usa il rendiconto finanziario.",
    "Di quanto è aumentato l’utile netto di Adobe tra il 2021 e il 2022? Indica la variazione in milioni di dollari.",
]

# Le schede di Streamlit eseguono il codice di tutte le pagine a ogni interazione,
# comprese le chiamate ad Atlas di Admin e Upload. Con il selettore gira solo la pagina aperta.
_PAGES = {"story": "📖 How it Works", "upload": "📤 Upload", "query": "🔎 Query",
          "agentic": "🧭 Ricerca agentica", "admin": "⚙ Admin"}
if st.session_state.get("page_nav") not in _PAGES:
    st.session_state.page_nav = st.session_state.get("_last_page", "story")
_page = st.segmented_control("Pagina", list(_PAGES), format_func=_PAGES.get,
                             key="page_nav", label_visibility="collapsed") or "story"
st.session_state._last_page = _page


# ─── HOW IT WORKS (home) ───────────────────────────────────────────────────────
if _page == "story":
    st.markdown('<h2 class="gradient-header">Come funziona questa demo RAG</h2>', unsafe_allow_html=True)
    st.markdown(
        "**Dai documenti a una risposta con riferimenti alle fonti.** "
        "RAG significa recuperare contenuti pertinenti dai documenti e fornirli al modello linguistico "
        "per preparare la risposta. Il percorso ha due momenti: prima inserisci e indicizzi i documenti, "
        "poi li consulti facendo domande."
    )
    st.markdown(
        "**Perché MongoDB Atlas:** conserva chunk e metadati nella collection e, in questa demo, "
        "i file originali completi in GridFS; "
        "integra il motore full-text **Atlas Search**, il motore semantico **Atlas Vector Search** "
        "e la fusione delle classifiche. Questo riduce i sistemi da sincronizzare per archiviazione e ricerca. "
        "I parser preparano i documenti, Voyage genera gli embedding e riordina i risultati, "
        "il LLM scrive la risposta."
    )
    st.markdown(
        "**File originali e GridFS:** gli originali sono i file completi caricati, conservati nel loro "
        "formato binario per poterli scaricare e aprire con l'applicazione appropriata. "
        "GridFS è un meccanismo di MongoDB per archiviare e recuperare file binari di qualsiasi formato: "
        "PDF, Word, Excel, PowerPoint, immagini e altri. Suddivide i file in blocchi binari e conserva "
        "i relativi metadati in MongoDB; il driver ricompone il file quando viene richiesto. "
        "Questi blocchi servono all'archiviazione del file; i chunk della ricerca sono invece passaggi "
        "estratti dal documento."
    )
    st.markdown(
        "**Un'alternativa è l'object storage**, un servizio dedicato all'archiviazione dei file: "
        "i binari possono risiedere lì, mentre MongoDB conserva i contenuti ricercabili, i metadati "
        "e il riferimento al file. GridFS è la scelta adottata in questa demo per tenere anche i binari "
        "in MongoDB; il collegamento a un object storage richiederebbe un'integrazione dedicata. "
        "La possibilità di archiviare un formato è distinta dalla capacità dei parser di estrarne il contenuto."
    )

    st.markdown("### 1. Quando inserisci un documento")
    st.caption("Il documento viene preparato al caricamento e resta disponibile per le domande successive.")
    st.graphviz_chart(r"""
        digraph ingestion {
            rankdir=LR
            bgcolor="transparent"
            node [fontname="Helvetica" fontsize=12 style="filled,rounded" shape=box color="#00684a" fillcolor="#F2FAF7" fontcolor="#00684a"]
            edge [color="#888888"]
            file [label="Documento\nPDF, Word, PowerPoint, testo"]
            parsing [label="Preparazione\nTesto, tabelle, OCR e pagine"]
            chunks [label="Chunk e metadati\nPassaggi con riferimenti alla fonte"]
            index [label="Indicizzazione in Atlas\nTesto + embedding Voyage" fillcolor="#00684a" fontcolor="#FFFFFF"]
            file -> parsing -> chunks -> index
        }
    """)
    st.markdown(
        "Un **chunk** è un passaggio del documento che può essere recuperato e citato. "
        "Un **embedding** è una rappresentazione numerica del suo contenuto, usata per cercare per significato. "
        "Nel percorso visivo vengono rappresentate anche le immagini delle pagine e delle slide. "
        "Il file completo caricato resta disponibile in GridFS per poterlo scaricare dalla risposta."
    )

    st.markdown("#### Quattro modalità: cosa cambia e perché")
    st.markdown(
        f"""
        | Modalità | Come viene indicizzato il contenuto | Perché confrontarla |
        |---|---|---|
        | **Testo gestito da Atlas · `auto`** | Atlas genera gli embedding con `{config.VOYAGE_EMBED_MODEL}` e li mantiene nell'indice. | Osservare la gestione degli embedding delegata ad Atlas. |
        | **Testo gestito dall'app · `manual`** | L'app chiama `{config.VOYAGE_EMBED_MODEL}` e salva i vettori nei chunk. | Osservare il controllo applicativo su generazione e memorizzazione. |
        | **Testo con contesto · `context`** | `{config.VOYAGE_CONTEXT_MODEL}` rappresenta ciascun chunk tenendo conto degli altri passaggi del documento. | Confrontare il recupero di passaggi brevi o dipendenti da una spiegazione precedente. |
        | **Testo e contenuti visivi · `multimodal`** | `{config.VOYAGE_MULTIMODAL_MODEL}` rappresenta testo e immagini delle pagine/slide. | Cercare anche contenuti riconoscibili da grafici, diagrammi e disposizione visiva. |
        """
    )
    st.caption(
        "Una sola modalità è attiva alla volta: in Upload scegli la modalità di caricamento, che diventa "
        "quella in cui cercano Query e Ricerca agentica. In Admin puoi cambiare la modalità attiva e trovi "
        "modelli, indici e codice di creazione. Per confrontare due modalità, carica lo stesso documento in "
        "entrambe e cambia la modalità attiva tra una domanda e l'altra."
    )

    st.markdown("### Domande pronte per la demo")
    st.markdown(
        "Queste cinque domande usano il corpus FinanceBench caricato nella demo. "
        "Selezionale in **Query** o in **Ricerca agentica**; lascia invariati filtri e parametri "
        "quando vuoi confrontare i due percorsi. Le spiegazioni indicano che cosa verificare nella fonte, "
        "senza suggerire la risposta al modello."
    )
    st.markdown(
        """
        | Domanda | Cosa mette alla prova | Controllo sulla fonte |
        |---|---|---|
        | **Quali sono state le spese in conto capitale di 3M nel 2018, in milioni di dollari? Usa il rendiconto finanziario.** | Recupero di un dato esplicito in una tabella finanziaria. | Verifica la riga PP&E e la colonna 2018; le parentesi rappresentano un’uscita di cassa. |
        | **Quanto ha speso Adobe per ricerca e sviluppo nell’esercizio 2022? Indica l’importo in milioni di dollari.** | Domanda in italiano con voce contabile nel documento in inglese. | Verifica la riga *Research and development* e l’esercizio 2022; non confonderla con *Sales and marketing*. |
        | **Quali sono stati i ricavi e l’utile operativo del segmento AWS di Amazon nel 2022? Riporta entrambi in milioni di dollari.** | Recupero di due valori correlati dallo stesso segmento. | Controlla entrambe le righe nella sezione AWS; non usare i valori consolidati di Amazon. |
        | **Quali sono state le spese in conto capitale di 3M nel 2017, in milioni di dollari? Usa il rendiconto finanziario.** | Domanda trabocchetto: stessa azienda e metrica della prima, anno diverso. | Verifica la colonna 2017 nel rendiconto; una risposta riferita al 2018 non è corretta. |
        | **Di quanto è aumentato l’utile netto di Adobe tra il 2021 e il 2022? Indica la variazione in milioni di dollari.** | Premessa falsa e confronto tra anni. | Controlla entrambi i valori di *Net income* e se la direzione indicata dalla domanda è supportata. |
        """
    )
    st.caption(
        "Per ogni prova apri il riferimento citato e controlla azienda, voce, anno, unità e valori. "
        "Una risposta plausibile senza una fonte che la sostenga non basta. Le domande e le note di verifica "
        "sono documentate anche in DEMO_QUERIES.md."
    )

    with st.expander("Perché servono parser, OCR e LibreOffice anche usando Voyage"):
        st.markdown(
            """
            - **PDF:** pymupdf4llm estrae testo, titoli e tabelle con le colonne; PyMuPDF renderizza le pagine.
              Tesseract, se disponibile, estrae il testo dalle pagine che richiedono OCR.
            - **Word e PowerPoint:** MarkItDown estrae testo e struttura. LibreOffice converte il documento
              in PDF per ottenere le immagini delle pagine o delle slide. Questa conversione è necessaria
              per il percorso visivo Office; nel percorso testuale è previsto un ripiego su MarkItDown.
            - **Voyage:** riceve il testo e, nel percorso multimodale, le immagini preparate dall'app e
              produce gli embedding. La lettura del file Office e la sua conversione restano compiti dei parser.

            Per un PDF, anche multimodale, LibreOffice non è necessario. Il setup in **Admin** prepara
            collection e indici; Atlas completa l'indicizzazione in modo asincrono. Controlla lo stato
            dell'indice prima di passare alla ricerca.
            """
        )

    st.markdown("### 2. Quando fai una domanda")
    st.graphviz_chart(r"""
        digraph querying {
            rankdir=LR
            bgcolor="transparent"
            node [fontname="Helvetica" fontsize=12 style="filled,rounded" shape=box color="#00684a" fillcolor="#F2FAF7" fontcolor="#00684a"]
            edge [color="#888888"]
            query [label="Domanda\n+ filtri selezionati"]
            text [label="Atlas Search\nParole e codici"]
            vector [label="Atlas Vector Search\nSignificato e contenuti visivi"]
            fusion [label="Fusione\n$rankFusion"]
            rerank [label="Voyage rerank\nPertinenza dei candidati" color="#7e57c2" fillcolor="#F3E5F5" fontcolor="#7e57c2"]
            answer [label="LLM\nRisposta con riferimenti" color="#7e57c2" fillcolor="#7e57c2" fontcolor="#FFFFFF"]
            query -> text -> fusion
            query -> vector -> fusion
            fusion -> rerank -> answer
        }
    """)
    st.markdown(
        """
        | Passaggio | Perché serve |
        |---|---|
        | **Ricerca lessicale e semantica** | Parole e codici aiutano a trovare riferimenti precisi; il significato aiuta quando la domanda usa parole diverse. Il contenuto visivo contribuisce nel percorso `multimodal`. |
        | **Fusione delle classifiche** | `$rankFusion` combina la classifica testuale e le classifiche vettoriali delle modalità incluse in una lista di candidati. |
        | **Reranking** | Voyage confronta la domanda con il testo dei candidati e li riordina per pertinenza. |
        | **Risposta con fonti** | Il LLM riceve i passaggi selezionati ed è istruito a citarli. L'app permette di consultare i riferimenti e scaricare i file originali completi. |
        """
    )
    with st.container(border=True):
        st.markdown("#### Istruzioni al reranker: definire cosa rende utile un risultato")
        st.markdown(
            "**Puoi guidare Voyage con criteri di rilevanza espressi in linguaggio naturale.** "
            "Aggiungendo istruzioni alla domanda inviata al reranker, puoi indicare quali contenuti "
            "privilegiare, quali aspetti del documento considerare o il contesto della richiesta. "
            "Il valore è adattare l'ordinamento dei candidati allo scopo dell'utente, "
            "usando i documenti già indicizzati."
        )
        st.markdown(
            """
            **Domanda:** «Come ripristino la connessione del dispositivo?»

            **Istruzione al reranker:** «Privilegia i passaggi con procedure operative e sequenze
            di azioni. Assegna minore rilevanza alle descrizioni generali del prodotto.»

            **Effetto da osservare:** se, a parità di candidati, le procedure utili risalgono
            nella classifica. L'istruzione influenza i punteggi di pertinenza; le condizioni
            obbligatorie di inclusione o esclusione restano affidate ai filtri sui documenti.
            """
        )
        st.caption(
            "Capacità supportata dai modelli Voyage con istruzioni: "
            "[documentazione](https://www.mongodb.com/docs/voyageai/models/rerankers/?client=python). "
            "In Query trovi il campo facoltativo Istruzioni al reranker sotto la domanda. "
            "Se compilato, aggiunge i criteri alla richiesta di reranking; se vuoto, viene inviata solo la domanda. "
            "Nei dettagli del reranking puoi leggere il testo inviato."
        )
    st.info(
        "**Cosa arriva al modello nel percorso multimodale:** le immagini contribuiscono agli embedding "
        "e al recupero delle pagine. Oggi il reranker e il LLM ricevono però il testo estratto e i riferimenti, "
        "senza le immagini. Una pagina trovata grazie a un grafico può perdere posizioni se contiene poco testo; "
        "la risposta finale non interpreta direttamente quel grafico."
    )

    st.markdown("### Approfondimenti tecnici")
    with st.expander("Indice vettoriale e filtri: quali documenti partecipano alla ricerca"):
        st.markdown(
            """
            Per impostazione predefinita, Atlas Vector Search usa HNSW: una struttura a grafo che accelera
            la ricerca approssimata dei vettori vicini senza esaminare in sequenza tutti i chunk. È possibile
            configurare anche `flat`, che esamina i vettori in modo esaustivo e offre richiamo completo,
            ma può diventare più lento quando il dataset cresce. La demo usa gli indici vettoriali standard
            definiti in Admin; non espone la scelta tra HNSW e `flat`.

            I filtri della demo riguardano campi del documento: nome file, autore, lingua, categoria,
            stato, anno, tag e tipo file. Nel ramo vettoriale i campi devono essere dichiarati nell'indice
            con tipo `filter`. Nel ramo testuale servono mapping compatibili con gli operatori usati;
            le condizioni residue vengono applicate con `$match`.

            Il filtro restringe i documenti ammessi; l'embedding serve a ordinarli per vicinanza semantica.
            In questa demo non sono esposti filtri sulle coordinate dei vettori o soglie sul punteggio
            vettoriale. La soglia di pertinenza riguarda invece il punteggio successivo di Voyage rerank.

            Atlas offre anche pre-filtri su testo analizzato, frasi e altre condizioni attraverso
            l'operatore `vectorSearch` dentro `$search`. È una possibilità distinta dallo stage
            `$vectorSearch` usato qui: vedi la
            [documentazione sulla ricerca ibrida](https://www.mongodb.com/docs/vector-search/hybrid-search/hybrid-search-overview/).
            """
        )

    with st.expander("Quantizzazione: ridurre memoria e costi dell’indice"):
        st.markdown(
            """
            La quantizzazione riduce i bit usati per rappresentare ogni dimensione del vettore. Può diminuire
            memoria e spazio dell’indice e velocizzare la ricerca, con un possibile calo di precisione.
            È una scelta distinta da HNSW o `flat`: riguarda la rappresentazione dei vettori, non la struttura
            usata per cercarli.

            | Modalità | Come rappresenta i vettori | Vantaggi e compromessi |
            |---|---|---|
            | `float` | 32 bit per valore, precisione completa | Massima fedeltà; richiede più memoria e spazio. |
            | `scalar` | 8 bit per valore | In genere circa un quarto della memoria rispetto a `float`, con una perdita di precisione contenuta. È il default di `autoEmbed`. |
            | `binary` | 1 bit per valore nell’indice, con vettori originali conservati per il reranking | Riduce molto la memoria; il reranking recupera precisione sui candidati, aggiungendo lavoro alla query. |
            | `binaryNoRescore` | 1 bit per valore, senza reranking dei vettori originali | Minore memoria e query più rapide; può ridurre maggiormente la precisione. Disponibile per `autoEmbed`. |

            In questa demo la modalità `auto` usa `autoEmbed` e, non specificando la quantizzazione,
            adotta il default `scalar`. Gli indici `manual`, `context` e `multimodal` sono indici vettoriali
            standard e non specificano quantizzazione, quindi usano vettori a precisione completa. La UI
            non espone ancora questa impostazione. Per questi indici standard MongoDB documenta `none`,
            `scalar` e `binary`; `binaryNoRescore` è documentato per `autoEmbed`.

            I risparmi effettivi e l’eventuale perdita di qualità dipendono da modello, corpus e query:
            vanno misurati sul proprio carico. Vedi la
            [documentazione MongoDB sulla quantizzazione](https://www.mongodb.com/docs/vector-search/about/vector-quantization/)
            e [sui tipi di indice e campo](https://www.mongodb.com/docs/vector-search/indexes/vector-search-type/).
            """
        )

    with st.expander("Top-K, Top-N e candidati: regolare ampiezza e costo della ricerca"):
        st.markdown(
            """
            Questi parametri agiscono in punti diversi della pipeline:

            | Parametro | Dove agisce | Se lo aumenti |
            |---|---|---|
            | `numCandidates` | Quanti vicini considera ciascun ramo vettoriale durante la ricerca approssimata | Può aumentare la probabilità di recuperare i vicini migliori, con più lavoro e latenza. |
            | **Top-K** | Quanti risultati conserva ogni ramo e quanti passaggi mantiene la lista dopo la fusione | Allarga il bacino di candidati; aumentano lavoro di ricerca e dati da riordinare. |
            | **Top-N** | Quanti passaggi, dopo il reranking, vengono inviati al LLM | Offre al modello più evidenze, ma aumenta il contesto e può aggiungere passaggi meno pertinenti. |

            In Query puoi regolare Top-K e Top-N. `numCandidates` non ha un controllo nella UI:
            il backend lo calcola per ogni ramo vettoriale come `max(10 × Top-K, 100)`.
            MongoDB raccomanda come punto di partenza almeno 20 volte il `limit` vettoriale; è una
            raccomandazione da verificare sul corpus, non una garanzia di qualità. Per cambiare il valore
            usato dalla demo si configurano `RETRIEVAL_NUM_CANDIDATES_MULTIPLIER` e
            `RETRIEVAL_NUM_CANDIDATES_MIN`.

            Per valutare una modifica, usa le stesse domande e gli stessi filtri e controlla se la fonte
            attesa entra nei risultati e sostiene la risposta. Più candidati non correggono un embedding
            inadatto né garantiscono una risposta corretta.

            [MongoDB: scelta di `numCandidates`](https://www.mongodb.com/docs/vector-search/query/aggregation-stages/vector-search-stage/#numcandidates-selection)
            """
        )

    with st.expander("Similarità vettoriale: cosine, dot product ed euclidean"):
        st.markdown(
            """
            La funzione di similarità confronta il vettore della domanda con quelli indicizzati e aiuta
            a ordinare i risultati. Non cambia il testo dell’embedding: cambia il criterio geometrico usato
            per confrontare i vettori.

            | Funzione | Intuizione | Quando è utile |
            |---|---|---|
            | `cosine` | Confronta l’angolo tra i vettori e ignora la loro grandezza. | Scelta comune per embedding semantici; adatta quando conta la direzione. |
            | `dotProduct` | Considera sia l’angolo sia la grandezza dei vettori. | Ha lo stesso ordinamento di cosine se i vettori sono normalizzati a lunghezza unitaria; altrimenti la grandezza influisce. |
            | `euclidean` | Misura la distanza in linea retta tra i vettori. | Utile quando distanza e scala delle dimensioni riflettono la similarità voluta; la scelta va verificata con il modello. |

            Gli indici `manual`, `context` e `multimodal` della demo specificano `cosine`.
            L’indice `auto` usa `autoEmbed`; lasciando le impostazioni predefinite MongoDB usa
            quantizzazione `scalar` e similarità `cosine`. La demo non espone la scelta della funzione.
            In generale segui le indicazioni del modello di embedding e confronta i risultati sul tuo corpus:
            cambiare metrica richiede una definizione indice coerente.

            [MongoDB: funzioni di similarità](https://www.mongodb.com/docs/vector-search/indexes/vector-search-type/#similarity-functions)
            """
        )

    with st.expander("Fusione: leggere un esempio di pipeline MongoDB"):
        st.markdown(
            "`$rankFusion` combina le posizioni nelle classifiche dei due canali; i punteggi grezzi "
            "di ricerca testuale e vettoriale hanno scale diverse. `scoreDetails` permette di leggere "
            "il contributo di ogni ramo. Questo esempio illustrativo usa `auto`, i nomi di indice predefiniti "
            "e filtri di esempio; la pipeline della domanda effettiva si consulta in Query."
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
            filter: { embedding_mode: "auto", language: "it", category: "Manuale" }
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
                { equals: { path: "embedding_mode", value: "auto" } },
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
        st.caption(
            "La vista didattica esegue anche query diagnostiche separate per mostrare i due canali. "
            "Reranking Voyage e generazione LLM sono chiamate successive alla ricerca in Atlas."
        )

    with st.expander("Embedding testuali: cosa distinguono auto, manual e context"):
        st.markdown(
            """
            **Auto e manual** permettono di confrontare dove avviene la generazione degli embedding:
            Atlas mantiene i vettori nell'indice in `auto`; l'app li genera via Voyage e li salva nei
            documenti in `manual`. La modalità `auto` riduce la logica di embedding da mantenere nell'app.

            **Context** affronta un problema diverso: un passaggio come “questa eccezione vale per la
            regola precedente” è ambiguo se letto da solo. Il modello riceve i chunk del documento
            insieme e rappresenta ciascuno usando il contesto circostante. I confini dei chunk e i
            riferimenti di pagina restano quelli preparati dall'app.

            La configurazione attuale usa input già suddivisi in chunk e controlla un limite stimato
            di 32K token per documento nel percorso `context`. Confronta le stesse domande per capire
            quando il contesto aiuta sul tuo corpus.
            """
        )

    with st.expander("Reranking e soglia: come vengono scelti i passaggi per il LLM"):
        st.markdown(
            f"La demo usa **{'Atlas $rerank nativo' if config.RERANK_BACKEND == 'native' else 'Voyage rerank API'}** "
            f"con il modello `{config.VOYAGE_RERANK_MODEL}`. "
            "Il reranker riordina i candidati già recuperati confrontandone il testo con la domanda. "
            "Nel percorso nativo `$rerank` segue `$rankFusion` nella stessa aggregazione; "
            "Atlas invoca Voyage usando l’abilitazione del progetto. Le istruzioni opzionali "
            "vengono aggiunte a `query.text`. I punteggi della fusione sono conservati separatamente."
        )
        st.markdown(
            "**La soglia controlla il miglior punteggio rerank:** se è inferiore al valore scelto, "
            "la demo si ferma prima di chiamare il LLM. Il controllo non elimina singolarmente tutti "
            "i passaggi sotto soglia. Il punteggio misura pertinenza e non certifica la correttezza "
            "della risposta: verifica sempre che le citazioni la sostengano. "
            "Il percorso nativo non richiede la chiave Voyage nell’app per il reranking. "
            "Se Atlas rifiuta lo stage, la query mostra l’errore senza cambiare percorso in silenzio."
        )


# ─── UPLOAD ────────────────────────────────────────────────────────────────────
if _page == "upload":
    st.subheader("Carica documenti")

    with st.expander("📚 Inventario documenti e embedding", expanded=False):
        st.caption(
            "Per ogni documento e modalità mostra i metadati salvati nei chunk. "
            "Se un campo non era stato registrato, resta non disponibile."
        )
        if st.button("Mostra o aggiorna inventario", key="show_document_inventory"):
            try:
                st.session_state.document_embedding_inventory = document_inventory(_coll())
            except Exception:
                st.session_state.pop("document_embedding_inventory", None)
                st.error("Inventario non disponibile: verifica la connessione e i permessi sul database.")
        if "document_embedding_inventory" in st.session_state:
            _inventory = st.session_state.document_embedding_inventory
            _auto_index = _inventory["auto_index"]
            st.write(f"Indice autoEmbed `{_auto_index['name']}`: ultima definizione disponibile "
                     f"`{_auto_index['model'] or 'modello non disponibile'}` · stato `{_auto_index['status']}`.")
            st.caption("Per autoEmbed il modello registrato al caricamento è un metadato: "
                       "il modello nell'indice può essere cambiato successivamente. "
                       "Lo stato dell'indice non certifica che ogni chunk sia già indicizzato. "
                       "Premi il pulsante per aggiornare l'inventario.")
            if _inventory["rows"]:
                st.dataframe(_inventory["rows"], hide_index=True, use_container_width=True)
            else:
                st.info("Nessun chunk presente nella collection configurata.")

    # Versioning delle key dei widget: incrementare = nuovi widget vuoti.
    # Permette di resettare il form senza perdere embed_mode (che vive
    # su una chiave session_state separata, non versionata).
    if "upload_form_v" not in st.session_state:
        st.session_state.upload_form_v = 0

    # ─── Caso A: ingestion già completata → riepilogo + pulsante reset ────────
    if "last_batch_ingestion" in st.session_state:
        batch = st.session_state.last_batch_ingestion
        results = batch["results"]
        inserted = sum(r["Esito"] == "Inserito" for r in results)
        skipped = sum(r["Esito"] == "Già presente" for r in results)
        errors = sum(r["Esito"] == "Errore" for r in results)
        completed = len(results) == batch["total"]
        if completed and not errors:
            st.success(f"Elaborazione terminata: {len(results)} documenti esaminati.")
        else:
            st.warning(
                f"Elaborati {len(results)} di {batch['total']} documenti · "
                f"{errors} errori. Consulta il dettaglio per file."
            )
        stats = st.columns(4)
        stats[0].metric("Documenti inseriti", inserted)
        stats[1].metric("Già presenti — saltati", skipped)
        stats[2].metric("Errori", errors)
        stats[3].metric("Chunk inseriti", sum(r["Chunk"] for r in results))
        st.caption(
            f"Modalità: {batch['mode']}. I file sono elaborati uno alla volta. "
            "Atlas aggiorna gli indici in modo asincrono: fine elaborazione non "
            "significa che tutti i nuovi passaggi siano già ricercabili."
        )
        st.dataframe(results, hide_index=True, use_container_width=True)
        st.download_button(
            "Scarica riepilogo caricamento", data=json.dumps(batch, ensure_ascii=False, indent=2),
            file_name="riepilogo_caricamento.json", mime="application/json",
        )
        if st.button("Carica altri documenti", type="primary", key="reset_batch_upload"):
            st.session_state.upload_form_v += 1
            st.session_state.pop("last_batch_ingestion", None)
            st.rerun()
    elif "last_ingestion" in st.session_state:
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
        st.success(f"Ingestion completata: **{summary.get('filename', 'documento')}**")
        stats = st.columns(5)
        stats[0].metric("Chunk indicizzati", summary.get("chunks_inserted", 0))
        stats[1].metric("Pagine elaborate", summary.get("pages_processed", summary.get("pdf_native_metadata", {}).get("total_pages", "—")))
        stats[2].metric("Tabelle rilevate", summary.get("tables_detected", "—"))
        stats[3].metric("Dimensione file", f"{summary['file_size_bytes'] / (1024 * 1024):.2f} MB" if summary.get("file_size_bytes") else "—")
        stats[4].metric("Durata", f"{summary['duration_seconds']:.1f} s" if summary.get("duration_seconds") is not None else "—")
        st.caption(
            f"Modalità `{summary.get('mode', '—')}` · embedding "
            f"`{summary.get('embedding', {}).get('model', '—')}` · "
            f"{summary.get('pages_with_text', '—')} pagine con testo · "
            f"{summary.get('pages_ocr', '—')} pagine processate con OCR"
        )
        if summary.get("images_processed"):
            st.info(f"Descrizioni di immagini create: **{summary['images_processed']}**")
        if summary.get("mode") == "multimodal":
            visual_pages = summary.get("chunks_by_type", {}).get("visual_page", 0)
            st.info(f"Pagine/slide rappresentate anche visivamente nell’indice multimodale: **{visual_pages}**")
        with st.expander("Dettagli tecnici dell’ingestion"):
            st.write("**Chunk per tipo di contenuto**")
            st.json(summary.get("chunks_by_type", {}))
            st.write("**Riferimenti e identificativi**")
            st.json({
                "document_id": summary.get("document_id"),
                "storage": summary.get("storage"),
                "duplicate_policy": summary.get("duplicate_policy"),
                "pipeline_version": config.INGESTION_PIPELINE_VERSION,
                "chunker_version": config.CHUNKER_VERSION,
            })
        if summary.get("pages_processed") and not summary.get("pages_with_text"):
            st.warning("Il file è stato elaborato ma non è stato estratto testo. Verifica che il documento non sia vuoto o protetto e che la pipeline OCR sia disponibile.")
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

        _modes_order = ["auto", "manual", "context", "multimodal"]
        try:
            _counts = chunk_counts_by_mode()
        except Exception:
            _counts = {}
        st.markdown("#### In quale modalità vuoi caricare i documenti?")
        st.caption(
            "Esiste una sola modalità attiva alla volta: è quella usata per caricare i documenti "
            "e l'unica in cui cercano Query e Ricerca agentica. I documenti caricati in altre "
            "modalità restano nel database ma non vengono cercati finché non torni a quella modalità."
        )
        selected_upload_mode = st.radio(
            "Modalità di caricamento",
            options=_modes_order,
            index=_modes_order.index(st.session_state.embed_mode) if st.session_state.embed_mode in _modes_order else 0,
            format_func=lambda mode: f"{mode} · {MODE_DESCRIPTIONS[mode][0]}",
            key=f"upload_mode_v{v}",
            label_visibility="collapsed",
        )
        st.caption("Chunk già caricati per modalità: " + " · ".join(
            f"`{mode}` {_counts.get(mode, 0)}" + (" (attiva)" if mode == st.session_state.embed_mode else "")
            for mode in _modes_order))
        with st.container(border=True):
            _title, _description = MODE_DESCRIPTIONS[selected_upload_mode]
            st.markdown(f"**`{selected_upload_mode}` · {_title}**")
            st.write(_description)
            if selected_upload_mode == "multimodal":
                st.caption("PDF e file Office mantengono i riferimenti a pagina o slide. DOCX/PPTX vengono convertiti "
                           "in pagine tramite LibreOffice per la parte visiva.")
            else:
                st.caption("Estraggo testo, struttura e tabelle; sui PDF scannerizzati applico OCR. "
                           "Le immagini incorporate non vengono descritte in questa modalità.")
            if selected_upload_mode != st.session_state.embed_mode:
                st.warning(
                    f"Avviando il caricamento, la modalità attiva passa da `{st.session_state.embed_mode}` "
                    f"a `{selected_upload_mode}`: Query e Ricerca agentica cercheranno solo nei documenti "
                    f"caricati in `{selected_upload_mode}` ({_counts.get(selected_upload_mode, 0)} chunk già presenti, "
                    "più quelli che stai caricando)."
                )
            else:
                st.info(f"Modalità attiva: Query e Ricerca agentica cercano già in `{selected_upload_mode}`.")

        selected_index_name = config.vector_index_name(selected_upload_mode)
        selected_index_health = _health_check(selected_upload_mode).get("Atlas Search Indexes")
        if selected_index_health and selected_index_health[0] == "ok":
            st.success(f"Indice necessario disponibile: `{selected_index_name}` — {selected_index_health[1]}")
        elif selected_index_health:
            st.warning(f"Prima dell’ingestion verifica/crea l’indice `{selected_index_name}` dal pannello Admin. Stato: {selected_index_health[1]}")

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
        upload_source = st.radio(
            "Come vuoi selezionare i documenti?",
            ["Uno o più file", "Una cartella"], horizontal=True,
            key=f"upload_source_v{v}",
        )
        uploaded_files = st.file_uploader(
            "Seleziona la cartella dei documenti" if upload_source == "Una cartella"
            else "Seleziona uno o più documenti",
            type=["pdf", "docx", "doc", "pptx", "ppt", "xlsx", "xls",
                  "txt", "html", "htm", "csv", "json", "xml", "md"],
            accept_multiple_files="directory" if upload_source == "Una cartella" else True,
            help="PDF parsato da pymupdf4llm (testo e tabelle in Markdown) + Tesseract per le scansioni. "
                 "Altri formati via MarkItDown (Microsoft): markdown structured output.",
            key=f"files_{upload_source}_v{v}",
        )
        if upload_source == "Una cartella":
            st.caption(
                "Vengono inclusi anche i documenti nelle sottocartelle. Per FinanceBench "
                "seleziona la cartella pdfs: contiene solo le fonti da indicizzare. "
                "L’inserimento nel database inizia quando premi Avvia ingestion."
            )
        if uploaded_files:
            st.info(
                f"Pronti {len(uploaded_files)} documenti · "
                f"{sum(f.size for f in uploaded_files) / (1024 * 1024):.1f} MB. "
                "Saranno elaborati uno alla volta con il percorso selezionato sopra."
            )
            with st.expander("Elenco dei documenti selezionati"):
                st.dataframe(
                    [{"File": f.name, "MB": round(f.size / (1024 * 1024), 2)}
                     for f in uploaded_files], hide_index=True, use_container_width=True,
                )

        with st.expander("📋 Metadati documento (opzionali — abilitano filtri smart in Query)",
                         expanded=True):
            st.caption(
                "Tutti i campi sono opzionali. Quelli compilati vengono ereditati da "
                "TUTTI i chunk di TUTTI i documenti selezionati e diventano filtrabili nella tab Query "
                "(i filtri appaiono automaticamente solo se ci sono dati popolati)."
            )
            if len(uploaded_files) > 1:
                st.info(
                    "Per bilanci di aziende e anni diversi lascia vuoti Titolo, Autore e Anno: "
                    "compilandoli assegneresti lo stesso valore a tutti i documenti. "
                    "Puoi usare un tag comune, per esempio financebench."
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
                    "replace": "Sostituisci il documento in questo percorso",
                    "append": "Aggiungi una copia in questo percorso",
                }[x],
                horizontal=True,
                key=f"u_duplicate_policy_v{v}",
                help="Il documento è riconosciuto tramite hash SHA-256. La scelta si applica al percorso di ricerca selezionato sopra.",
            )

        if uploaded_files and st.button("Avvia ingestion", type="primary"):
            st.session_state.embed_mode = selected_upload_mode
            st.session_state.last_upload_mode = selected_upload_mode
            try:
                save_active_mode(selected_upload_mode)
            except Exception as exc:
                st.warning(f"Modalità attiva non salvata nel database ({type(exc).__name__}): "
                           "vale solo per questa sessione.")
            preview_scope.clear()
            _health_check.clear()
            user_md: dict = {}
            if u_title.strip():    user_md["title"] = u_title.strip()
            if u_author.strip():   user_md["author"] = u_author.strip()
            if u_category:         user_md["category"] = u_category
            if u_status:           user_md["status"] = u_status
            if u_language:         user_md["language"] = u_language
            if u_year is not None: user_md["year"] = int(u_year)
            if u_tags.strip():
                user_md["tags"] = [t.strip() for t in u_tags.split(",") if t.strip()]
            user_md["duplicate_policy"] = u_duplicate_policy

            batch = {"total": len(uploaded_files), "mode": selected_upload_mode, "results": []}
            st.session_state.last_batch_ingestion = batch
            single_summary = None
            with st.status("Ingestion in corso… lascia aperta questa pagina", expanded=True) as status:
                progress = st.progress(0.0, text=f"0 di {len(uploaded_files)} documenti elaborati")
                current_file = st.empty()
                result_table = st.empty()
                for file_number, uploaded in enumerate(uploaded_files, start=1):
                    current_file.write(f"Documento {file_number}/{len(uploaded_files)}: {uploaded.name}")
                    started = time.perf_counter()
                    row = {"File": uploaded.name, "Esito": "Errore", "Pagine": 0, "Chunk": 0, "Secondi": 0.0, "Dettaglio": ""}
                    try:
                        # Directory uploads include relative paths. Keep only the filename
                        # inside a separate temporary directory for each document.
                        filename = Path(uploaded.name.replace("\\", "/")).name
                        if filename in {"", ".", ".."}:
                            raise ValueError("Nome file non valido")
                        with tempfile.TemporaryDirectory(prefix="rag-upload-") as temp_dir:
                            tmp = Path(temp_dir) / filename
                            tmp.write_bytes(uploaded.getbuffer())
                            summary = ingest_pdf(
                                tmp, db=_db(), collection=_coll(),
                                storage=storage, mode=selected_upload_mode,
                                user_metadata={
                                    **user_md,
                                    "file_type": tmp.suffix.lstrip(".").lower() or "pdf",
                                },
                            )
                        skipped = summary.get("skipped", False)
                        chunks = summary.get("chunks_inserted", 0)
                        row.update({
                            "Esito": "Già presente" if skipped else "Inserito" if chunks else "Errore",
                            "Pagine": summary.get("pages_processed", 0),
                            "Chunk": chunks,
                            "Dettaglio": summary.get("reason", "") if skipped else
                            "" if chunks else "Nessun contenuto indicizzabile estratto dal documento.",
                        })
                        if len(uploaded_files) == 1 and (skipped or chunks):
                            single_summary = summary
                    except Exception as exc:
                        row["Dettaglio"] = str(exc)
                    finally:
                        row["Secondi"] = round(time.perf_counter() - started, 1)
                        batch["results"].append(row)
                        # Persist each result so a rerun keeps the completed portion.
                        st.session_state.last_batch_ingestion = batch
                    progress.progress(
                        file_number / len(uploaded_files),
                        text=f"{file_number} di {len(uploaded_files)} documenti elaborati",
                    )
                    result_table.dataframe(batch["results"], hide_index=True, use_container_width=True)
                errors = sum(r["Esito"] == "Errore" for r in batch["results"])
                status.update(
                    label=f"Elaborazione terminata · {errors} errori",
                    state="error" if errors else "complete",
                )

            # Invalida cache filtri/lista così Query e Admin li ricaricano
            _available_metadata_filters.clear()
            _distinct_filenames.clear()
            _dominant_doc_langs.clear()
            _document_list.clear()
            preview_scope.clear()
            _health_check.clear()

            # Persisto il summary e ricarico: alla prossima render verrà
            # mostrato il blocco "Caso A" con il pulsante "Carica un altro
            # documento" — il form sparisce finché non l'utente non resetta.
            if single_summary is not None:
                st.session_state.pop("last_batch_ingestion", None)
                st.session_state.last_ingestion = single_summary
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
        k = _result_key(c)
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
        f"questa specifica fonte. Con più modalità, la posizione semantica è quella della fusione di confronto dei soli vettori. Da sinistra a destra: lessicale → semantico → "
        f"hybrid → rerank. Le frecce ➔ rappresentano la **promozione** lungo gli "
        f"stadi. Le ricerche per stadio sono profonde fino a top-{_JOURNEY_TOP_K} "
        f"per mostrarti **posizioni reali** anche per fonti inizialmente lontane "
        f"(es. `#27 BM25 → #4 Vector → #2 Hybrid → #1 Rerank` racconta meglio "
        f"di \"non trovata in top-5\")."
    )

    for chunk, cit_nums in cited_chunks_with_cit:
        key = _result_key(chunk)
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


def _semantic_label(mode: str) -> str:
    return {
        "auto": "Testo · embedding generati da Atlas",
        "manual": "Testo · embedding generati dall’app",
        "context": "Testo · embedding con contesto",
        "multimodal": "Testo e contenuti visivi",
    }.get(mode, mode)


def _rank_change(before: int | None, after: int | None) -> str:
    if before is None or after is None:
        return "—"
    if before > after:
        return f"Sale di {before - after}"
    if before < after:
        return f"Scende di {after - before}"
    return "Stabile"


@st.fragment
def _render_passage_inspector(docs: list[dict], query: str, key: str, label: str, *, lexical: bool = True) -> None:
    if not docs:
        return
    with st.expander(label):
        number = st.selectbox(
            "Passaggio da esaminare", range(len(docs)),
            format_func=lambda i: f"#{i + 1} · {docs[i].get('filename', 'Documento')} · {_page_label(docs[i].get('page_number'))} · chunk {docs[i].get('chunk_index', '—')}",
            key=f"inspect_{key}",
        )
        content = docs[number].get("content") or ""
        if not lexical:
            st.caption("La ricerca confronta il significato complessivo della domanda e del passaggio. Non restituisce parole o frasi che spieghino la similarità.")
            st.text(content)
            return
        highlighted = _highlight(content, query)
        st.caption("In giallo: parole della domanda presenti nel testo. L’evidenziazione non spiega il giudizio semantico e non certifica la risposta.")
        if "<mark " not in highlighted:
            st.caption("Nessuna parola significativa della domanda coincide letteralmente con il testo. Controlla anche traduzioni, anno e dati numerici.")
        st.markdown(f'<div style="white-space:pre-wrap;line-height:1.5">{highlighted}</div>', unsafe_allow_html=True)


def _render_search_candidates(docs: list[dict], query: str, score_name: str, key: str, *, rerank_positions: dict | None = None, cited_keys: dict | None = None) -> None:
    st.caption(f"{len(docs)} passaggi recuperati. La tabella mostra l’intera lista, ordinata per punteggio.")
    if not docs:
        st.caption("Nessun risultato.")
        return
    st.dataframe([{
        "Posizione": i, "File": doc.get("filename"),
        "Pagina": _page_label(doc.get("page_number")), "Passaggio": str(doc.get("chunk_index", "—")),
        f"Punteggio {score_name}": doc.get("score"),
        **({
            "Posizione dopo reranking": str((rerank_positions or {}).get(_result_key(doc), "Non disponibile")),
            "Citato nella risposta": "Sì" if _result_key(doc) in (cited_keys or {}) else "No",
        } if score_name == "Vector" else {}),
    } for i, doc in enumerate(docs, 1)], hide_index=True, use_container_width=True)
    if score_name == "Vector":
        st.caption("‘Non disponibile’: il passaggio non compare nei risultati restituiti dal reranker. Le liste di confronto possono differire dai candidati della query principale.")
    _render_passage_inspector(docs, query, f"search_{key}",
        "Leggi il passaggio recuperato per significato" if score_name == "Vector" else "Esamina il testo recuperato",
        lexical=score_name != "Vector")


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
        k = _result_key(c)
        fname = c.get("filename", "?")
        page = c.get("page_number")
        page_lbl = _page_label(page, missing="pagina non disponibile")
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
        return (passage_key(c),)

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
        page_lbl = _page_label(c.get("page_number"), missing="pagina non disponibile")
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
            "Il **cross-encoder Voyage rerank-3** ha letto query + chunk "
            "insieme e ha promosso le seguenti fonti come quelle che davvero "
            "rispondono alla domanda. Sono le **uniche citate dal LLM**."
        )
        for chunk, cit_nums in cited_chunks_with_cit:
            cit_label = ", ".join(f"[#{n}]" for n in sorted(cit_nums))
            fname = chunk.get("filename", "?")
            page_lbl = _page_label(chunk.get("page_number"), missing="pagina non disponibile")
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
                f'Il modello ha citato questo passaggio selezionato dal reranker. '
                f'Verifica che il contenuto sostenga la risposta.'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )



def _render_answer_diagnostics(diagnostics: dict) -> None:
    with st.expander("Dettaglio del controllo della risposta", expanded=False):
        st.caption("I tentativi sono mostrati solo per diagnosi: non costituiscono risposte verificate.")
        for number, attempt in enumerate(diagnostics.get("attempts") or [], 1):
            check = attempt.get("check") or {}
            st.write(f"**Tentativo {number}:** " + explain_answer_check(check.get("status"), check))
            st.caption(f"Provider: {attempt.get('provider') or 'non disponibile'} · modello: {attempt.get('model') or 'non disponibile'}")
            st.code(attempt.get("response") or "(risposta vuota)", language=None)
            if attempt.get("response_truncated"):
                st.caption("Risposta diagnostica limitata ai primi 12.000 caratteri.")
        if diagnostics.get("retry_error_type"):
            st.warning(f"Tentativo successivo non completato ({diagnostics['retry_error_type']}).")
        if not diagnostics.get("attempts"):
            st.caption("I tentativi originali non sono disponibili per questa esecuzione.")


_FLOW_MAX = 10  # passaggi mostrati nel grafico a corsie: oltre diventa illeggibile


def _theme_is_dark() -> bool:
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


def _flow_item(doc: dict, final: int, ranks: dict, cited: bool) -> dict:
    return {"file": doc.get("filename") or "Documento", "file_label": short_file(doc.get("filename")),
            "page_label": _page_label(doc.get("page_number"), missing="pagina n/d"),
            "snippet": snippet(doc.get("content"), 58), "ranks": ranks, "final": final, "cited": cited}


def _render_flow(**kwargs) -> None:
    page, height = rank_flow_html(dark=_theme_is_dark(), **kwargs)
    components.html(page, height=height + 10, scrolling=False)


def _compute_comparisons(result: dict) -> None:
    """Classifiche separate (solo testo, solo vettori per modalità), in parallelo.

    Separate diagnostic lists: never confuse vector-only RRF with similarity.
    Aggiorna il risultato salvato, così restano disponibili ai rerun successivi.
    """
    coll = _coll()
    modes = result["scope_v2"]["modes"]
    vector_modes = [mode for mode in modes if mode != "multimodal"]
    common = {"filters": result["filters_v2"], "top_k": result["top_k_v2"]}
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=1 + len(vector_modes)) as pool:
        text_future = pool.submit(search, coll, result["query_v2"], embed_mode="all",
                                  retrieval_mode="text", fuzzy=result.get("fuzzy_v2", False),
                                  modes=modes, **common)
        vector_futures = {mode: pool.submit(search, coll, result["query_v2"], embed_mode=mode,
                                            retrieval_mode="vector", modes=[mode],
                                            query_vectors=result.get("query_vectors_v2") or {}, **common)
                          for mode in vector_modes}
        result["text_deep_v2"] = text_future.result()
        result["vector_by_mode_v2"] = {mode: future.result() for mode, future in vector_futures.items()}
    result["timings_ms_v2"]["Classifiche di confronto"] = (time.perf_counter() - started) * 1000


def _render_query_result(result: dict) -> None:
    """Render the last completed run without calling retrieval or generation."""
    query_v2 = result["query_v2"]
    rerank_instructions_v2 = result["rerank_instructions_v2"]
    filters_v2 = result["filters_v2"]
    scope_v2 = result["scope_v2"]
    pipeline_v2 = result["pipeline_v2"]
    top_k_v2 = result["top_k_v2"]
    relevance_threshold_v2 = result["relevance_threshold_v2"]
    timings_ms_v2 = result["timings_ms_v2"]
    atlas_results_v2 = result["atlas_results_v2"]
    raw_v2 = result["raw_v2"]
    reranked_v2 = result["reranked_v2"]
    rerank_mode_v2 = result["rerank_mode_v2"]
    top_rerank_v2 = result["top_rerank_v2"]
    reviewed_v2 = result["reviewed_v2"]
    text_deep_v2 = result["text_deep_v2"]
    vector_by_mode_v2 = result["vector_by_mode_v2"]

    st.markdown("#### Domanda eseguita")
    st.write(query_v2)
    with st.expander("Perimetro e query di questa esecuzione", expanded=False):
        render_scope(scope_v2, executed=True)
        st.write("Filtri applicati: " + _filter_summary(filters_v2))
        st.caption(
            "Questa è la pipeline inviata ad Atlas. Il Top-K si applica a ogni ramo e poi alla lista fusa. "
            + ("Il reranking è eseguito da $rerank nella stessa pipeline. Restituiamo tutti i candidati "
               "per mostrare il prima/dopo; solo i primi Top-N vanno all’LLM."
               if rerank_mode_v2 == "native" else "Il reranking API e la generazione avvengono dopo.")
        )
        st.code(pipeline_code(pipeline_v2, scope_v2), language="javascript")

    st.markdown("### Risposta")
    answer_placeholder = st.empty()
    generated_v2 = reviewed_v2["answer"]
    answer_text_v2 = generated_v2.get("answer") or ""
    answer_provider_v2 = generated_v2.get("provider", config.LLM_PROVIDER)
    answer_model_v2 = generated_v2.get("model", "—")
    answer_usage_v2 = generated_v2.get("usage") or {}
    # RAG classica: una sola generazione, nessun secondo tentativo.
    # Una risposta con collegamenti esterni non viene MAI mostrata: contiene
    # contenuto che non proviene dai passaggi recuperati. Gli altri controlli
    # di forma sono avvisi accanto alla risposta.
    quality_v2 = inspect_answer(answer_text_v2, len(reranked_v2),
                                truncated=bool(generated_v2.get("truncated")))
    if quality_v2["status"] == "external_links":
        answer_text_v2 = ""
        answer_placeholder.error(
            "Risposta non pubblicata: " + explain_answer_check("external_links")
            + " Gli indirizzi web vengono tolti dai passaggi prima di inviarli al modello e il prompt li vieta: "
            "se succede di nuovo, il modello lo ha aggiunto di sua iniziativa. Le fonti recuperate restano consultabili sotto."
        )
        quality_v2 = {**quality_v2, "valid_citations": []}
    elif answer_text_v2:
        answer_placeholder.markdown(_normalize_inline_citations(answer_text_v2))
    else:
        answer_placeholder.warning("Il modello ha restituito una risposta vuota. Le fonti recuperate restano consultabili sotto.")
    citations_v2 = quality_v2["valid_citations"]
    _render_cited_sources_downloads(
        reranked_v2, answer_text=answer_text_v2,
    )
    if quality_v2["status"] == "insufficient":
        st.info(
            "Il modello dichiara che i passaggi non bastano per rispondere. "
            "Questo non dimostra che l'argomento sia assente dal database: "
            "controlla i contenuti recuperati nelle sezioni sotto."
        )
    elif quality_v2["status"] not in {"cited", "empty", "external_links"}:
        st.warning("Controllo dei riferimenti: " + explain_answer_check(quality_v2["status"], quality_v2)
                   + " Verifica la risposta nei passaggi recuperati.")
    if quality_v2.get("format_issues") and quality_v2["status"] != "external_links":
        st.caption("Alcune citazioni non sono nel formato previsto [#N]: "
                   + ", ".join(f"`({item})`" for item in quality_v2["format_issues"])
                   + ". Sono riferimenti interni ai passaggi, non collegamenti esterni.")
    if answer_text_v2:
        _render_citation_check(answer_text_v2, reranked_v2)
    _render_timings_bar(timings_ms_v2, answer_provider_v2, answer_model_v2, answer_usage_v2)
    with st.expander("Dettagli della generazione"):
        st.caption("File e pagine forniti al modello per questa risposta. Un file può contribuire con più passaggi.")
        sources = {}
        for number, passage in enumerate(reranked_v2, 1):
            file_id = str(passage.get("gridfs_file_id") or passage.get("document_id") or passage.get("filename"))
            entry = sources.setdefault(file_id, {"File": passage.get("filename", "Documento"), "Pagine": [], "Passaggi": [], "Citato nella risposta": False})
            page = _page_label(passage.get("page_number"))
            if page not in entry["Pagine"]:
                entry["Pagine"].append(page)
            entry["Passaggi"].append(f"#{number}")
            entry["Citato nella risposta"] |= number in citations_v2
        st.dataframe([{**entry, "Pagine": ", ".join(entry["Pagine"]), "Passaggi": ", ".join(entry["Passaggi"])} for entry in sources.values()], hide_index=True, use_container_width=True)

    cited_keys_v2: dict[tuple, list[int]] = {}
    for cit_n in citations_v2:
        if 1 <= cit_n <= len(reranked_v2):
            rk = _result_key(reranked_v2[cit_n - 1])
            cited_keys_v2.setdefault(rk, []).append(cit_n)

    # Riorganizzazione interattiva a Step per la demo (Semplificata & Visiva)
    st.divider()
    st.markdown("<h3 style='margin-top:0;'>🛠️ Come è stata costruita questa risposta?</h3>", unsafe_allow_html=True)
    st.caption("Segui le fonti: quali passaggi vengono trovati, come cambia il loro ordine e quali vengono usati nella risposta.")
    tab_step1, tab_step2, tab_step3, tab_step4, tab_step5 = st.tabs([
        "🎯 1. Analisi Query",
        "🔍 2. Ricerca testuale e semantica",
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
            f'<div><b>Modello Embedding:</b> percorsi {html.escape(", ".join(scope_v2["modes"]))}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        shell_query_v2 = pipeline_code(pipeline_v2, scope_v2)

        with st.expander("👁️ Mostra query", expanded=False):
            if scope_v2["modes"] == ["auto"]:
                st.caption(
                    "In modalità auto MongoDB genera l'embedding direttamente "
                    "nell'indice a partire dal testo della domanda."
                )
            st.markdown("**Pipeline MongoDB eseguita, pronta per `mongosh`**")
            st.code(shell_query_v2, language="javascript")

    with tab_step2:
        st.markdown("#### Due modi diversi di trovare candidati")
        st.write("La ricerca testuale cerca le parole della domanda; quella semantica cerca contenuti vicini per significato. Un candidato può parlare dell’argomento ma sbagliare azienda, anno o dato richiesto: essere recuperato non significa essere una fonte corretta.")
        st.caption("Queste classifiche sono ricerche di confronto eseguite dopo la risposta, con gli stessi filtri e limiti. Le posizioni effettive della fusione sono riportate nel passo 3; possono differire in caso di parità, ricerca approssimata o aggiornamenti dell’indice.")
        if text_deep_v2 is None:
            # Calcolate solo su richiesta: la risposta compare senza aspettarle.
            st.info("Le classifiche separate dei motori non sono ancora calcolate: servono solo a spiegare la fusione.")
            if st.button("Calcola le classifiche di confronto", key="compute_comparisons_v2"):
                with st.spinner("Preparo le classifiche di confronto…"):
                    _compute_comparisons(result)
                st.rerun()
            text_deep_v2, vector_by_mode_v2 = [], {}
        left, right = st.columns(2)
        with left:
            st.markdown("##### Ricerca testuale · BM25")
            st.write("Il punteggio premia le parole corrispondenti considerando frequenza, rarità e lunghezza del testo. Più alto significa meglio classificato per questa domanda; non ha un massimo fisso e non misura la probabilità che la risposta sia corretta.")
            _render_search_candidates(text_deep_v2, query_v2, "BM25", "text")
        with right:
            st.markdown("##### Ricerca semantica · Vector Search")
            st.write("Il punteggio misura la vicinanza tra i vettori della domanda e del passaggio secondo l’indice. Un valore alto non garantisce che coincidano anno, azienda o numero richiesto. Confrontalo solo all’interno dello stesso modello e della stessa ricerca.")
            for mode, candidates in vector_by_mode_v2.items():
                st.markdown(f"**{_semantic_label(mode)}**")
                _render_search_candidates(candidates, query_v2, "Vector", mode,
                    rerank_positions={_result_key(doc): rank for rank, doc in enumerate(atlas_results_v2 if rerank_mode_v2 == "native" else reranked_v2, 1)},
                    cited_keys=cited_keys_v2)
        if "multimodal" in scope_v2["modes"]:
            st.caption("La ricerca include anche i contenuti visivi. Qui mostriamo il confronto sui testi; il contributo dei contenuti visivi resta visibile nella fusione.")
        st.caption("BM25, Vector e RRF hanno scale diverse. Per esempio 0,0164 può essere il contributo RRF di un primo posto: 1 / (60 + 1). Qui sono mostrati i punteggi dei singoli motori; i punti RRF sono nel passo 3.")

    with tab_step3:
        st.markdown("#### Un’unica classifica dai risultati dei diversi motori")
        st.write("Ogni motore prepara una lista ordinata. La fusione RRF assegna punti in base alla posizione: un passaggio ai primi posti riceve più punti. Se lo stesso passaggio compare in più liste, somma i contributi e compare una sola volta nel risultato. " + dedup_caption() + " Passaggi diversi dello stesso file restano distinti.")
        st.write(f"In questa esecuzione partecipano la ricerca testuale e {len(scope_v2['modes'])} ricerche semantiche ({', '.join(_semantic_label(mode) for mode in scope_v2['modes'])}). Ogni lista può proporre fino a {top_k_v2} passaggi; dopo la fusione ne conserviamo {len(raw_v2)} complessivi.")
        st.caption(weights_caption(scope_v2["modes"]) + " Si sommano punti di posizione, non i punteggi BM25 e Vector: questi ultimi usano scale diverse.")
        with st.expander("Come MongoDB calcola la classifica RRF"):
            st.markdown("La formula è quella ufficiale di **MongoDB `$rankFusion`**. Le parole e l’esempio qui sotto servono a illustrarla. [Documentazione MongoDB](https://www.mongodb.com/docs/manual/reference/operator/aggregation/rankFusion/#reciprocal-rank-fusion--rrf--formula)")
            st.code("Punteggio RRF = somma, per ogni lista, di: peso / (60 + posizione)", language="text")
            st.markdown("**Posizione**: 1 per il primo risultato, 2 per il secondo, ecc. **Peso**: importanza della lista; i valori usati in questa esecuzione sono indicati sopra. **60**: costante scelta da MongoDB, non il numero di risultati né una soglia di affinità. Se il passaggio è assente da una lista, quella lista contribuisce zero.")
            st.markdown("**Esempio con due liste e due passaggi** (numeri illustrativi):")
            st.table([
                {"Passaggio": "A", "Testuale": "1°", "Semantica": "Assente", "Calcolo": "1 / (60 + 1)", "Totale": "0,01639", "Ordine finale": "2°"},
                {"Passaggio": "B", "Testuale": "10°", "Semantica": "10°", "Calcolo": "1 / 70 + 1 / 70", "Totale": "0,02857", "Ordine finale": "1°"},
            ])
            st.write("B supera A perché riceve punti da entrambe le liste. MongoDB ordina per somma decrescente. Con ulteriori modalità semantiche si aggiunge un contributo per ciascuna lista. Il punteggio indica priorità nella classifica, non probabilità che il contenuto risponda correttamente.")
        counts = _source_counts(raw_v2)
        st.markdown(f"**Da dove arrivano i {len(raw_v2)} passaggi conservati?**")
        origin_rows = [
            {"Origine": "Solo ricerca testuale", "Passaggi": counts["text_only"]},
            {"Origine": "Solo ricerca semantica (almeno una modalità)", "Passaggi": counts["vector_only"]},
            {"Origine": "Sia testuale sia semantica", "Passaggi": counts["both"]},
            {"Origine": "Dettaglio non disponibile", "Passaggi": counts["unknown"]},
        ]
        st.dataframe(
            [row for row in origin_rows if row["Passaggi"] > 0],
            hide_index=True, use_container_width=True,
        )
        st.caption(f"Totale: {len(raw_v2)} passaggi conservati dopo la fusione. Sono mostrate solo le origini presenti; ogni passaggio è contato una sola volta.")
        st.markdown("##### Posizione in ciascun motore e posizione finale")
        _single_mode = len(scope_v2["modes"]) == 1
        _flow_columns = [{"key": "text", "title": "Testuale", "sub": "BM25"}] + [
            {"key": f"vector_{mode}", "title": "Semantica" if _single_mode else mode, "sub": "vettori" if _single_mode else "semantica"}
            for mode in scope_v2["modes"]
        ]
        _flow_items = []
        for _position, _doc in enumerate(raw_v2[:_FLOW_MAX], 1):
            _branch = branch_ranks(_doc)
            _ranks = {"text": _branch.get("text")}
            for _mode in scope_v2["modes"]:
                _ranks[f"vector_{_mode}"] = _branch.get(f"vector_{_mode}", _branch.get("vector") if _single_mode else None)
            _flow_items.append(_flow_item(_doc, _position, _ranks, _result_key(_doc) in cited_keys_v2))
        st.caption(f"Ogni riga a destra è un passaggio, non un file: lo stesso documento può comparire più volte con pagine o parti diverse. "
                   f"Sono mostrati i primi {min(_FLOW_MAX, len(raw_v2))} passaggi della classifica fusa; a sinistra la loro posizione in ciascun motore.")
        _render_flow(columns=_flow_columns, target={"title": "Classifica fusa", "sub": "RRF"},
                     items=_flow_items, max_rank=_FLOW_MAX)
        fusion_rows = []
        branches = ["text"] + [f"vector_{mode}" for mode in scope_v2["modes"]]
        for position, doc in enumerate(raw_v2, 1):
            ranks = branch_ranks(doc)
            row = {"File": doc.get("filename"), "Pagina": _page_label(doc.get("page_number"))}
            for branch in branches:
                rank = ranks.get(branch)
                if branch.startswith("vector_") and len(scope_v2["modes"]) == 1:
                    rank = ranks.get(branch, ranks.get("vector"))
                label = "Testuale" if branch == "text" else _semantic_label(branch.removeprefix("vector_"))
                row[label] = f"#{rank}" if rank else "Assente"
                row[f"Variazione da {label}"] = _rank_change(rank, position) if rank else "—"
            row["Dopo fusione"] = position
            row["Punti RRF"] = doc.get("score")
            row["Inizio del passaggio"] = snippet(doc.get("content"), 80)
            fusion_rows.append(row)
        with st.expander(f"Tabella con le posizioni esatte di tutti i {len(raw_v2)} passaggi"):
            st.dataframe(fusion_rows, hide_index=True, use_container_width=True)

    with tab_step4:
        st.markdown("#### Il reranker rilegge e riordina i candidati")
        st.write(f"Voyage confronta la domanda con il testo e i metadati di ciascuno dei {len(raw_v2)} candidati. Assegna un nuovo punteggio di pertinenza e seleziona i primi {len(reranked_v2)} da fornire al modello della risposta. Non cerca altri documenti nel database.")
        if isinstance(top_rerank_v2, (int, float)):
            st.caption(f"Punteggio del primo passaggio dopo il reranking: `{top_rerank_v2:.3f}`. "
                       f"Sotto la soglia `{relevance_threshold_v2:.2f}` la domanda è considerata fuori tema "
                       "e il modello non viene chiamato.")
        with st.expander("Domanda e istruzioni inviate al reranker"):
            st.code(format_rerank_query(query_v2, rerank_instructions_v2), language="text")
            st.caption("Le eventuali istruzioni aggiuntive guidano questo riordino; la domanda usata nella ricerca e nella risposta rimane quella originale.")
        st.markdown("##### Prima della rilettura e dopo il reranking")
        all_reranked = atlas_results_v2 if rerank_mode_v2 == "native" else reranked_v2
        after_positions = {_result_key(doc): rank for rank, doc in enumerate(all_reranked, 1)}
        after_docs = {_result_key(doc): doc for doc in all_reranked}
        sent_keys = {_result_key(doc) for doc in reranked_v2}
        rows = []
        promoted = demoted = stable = 0
        for before, doc in enumerate(raw_v2, 1):
            key = _result_key(doc)
            after = after_positions.get(key)
            if after is not None:
                promoted += after < before
                demoted += after > before
                stable += after == before
            rows.append({
                "File": doc.get("filename"), "Pagina": _page_label(doc.get("page_number")),
                "Prima · RRF": before,
                "Dopo · rerank": after, "Variazione": _rank_change(before, after) if after else "Fuori dai risultati restituiti",
                "Punteggio rerank": after_docs.get(key, {}).get("_rerank_score"),
                "Inviato al LLM": "Sì" if key in sent_keys else "No",
                "Citato": ", ".join(f"#{n}" for n in cited_keys_v2.get(key, [])) or "No",
            })
        before_positions = {_result_key(doc): rank for rank, doc in enumerate(raw_v2, 1)}
        _rerank_items = [
            _flow_item(doc, after, {"rrf": before_positions.get(_result_key(doc))}, _result_key(doc) in cited_keys_v2)
            for after, doc in enumerate(all_reranked[:_FLOW_MAX], 1)
        ]
        st.caption(f"Tra i risultati con posizione disponibile: {promoted} salgono · {demoted} scendono · {stable} restano stabili. "
                   f"Il grafico mostra i primi {len(_rerank_items)} dopo il reranking; sopra la linea tratteggiata i passaggi inviati al modello.")
        _render_flow(columns=[{"key": "rrf", "title": "Classifica fusa", "sub": "RRF"}],
                     target={"title": "Dopo il reranking", "sub": "Voyage"}, items=_rerank_items,
                     max_rank=_FLOW_MAX, cut=len(reranked_v2),
                     cut_label=f"sopra la linea tratteggiata i {len(reranked_v2)} passaggi inviati al modello",
                     show_delta_from="rrf")
        for row, doc in zip(rows, raw_v2):
            row["Inizio del passaggio"] = snippet(doc.get("content"), 80)
        with st.expander(f"Tabella con tutti i {len(raw_v2)} candidati, prima e dopo il reranking"):
            st.dataframe(sorted(rows, key=lambda row: row["Dopo · rerank"] or len(raw_v2) + 1), hide_index=True, use_container_width=True)
        st.write("**Perché cambia l’ordine?** RRF usa le posizioni nelle liste; il reranker valuta insieme domanda e contenuto del singolo passaggio. Il servizio restituisce punteggi, non spiegazioni del singolo giudizio: il grafico mostra il cambiamento osservato, senza attribuire motivazioni al modello.")
        st.caption(f"Il punteggio rerank ordina la pertinenza rispetto a questa domanda: non è una percentuale di correttezza e non è confrontabile con BM25, Vector o RRF. Se il migliore è sotto {relevance_threshold_v2:.2f}, l’app non genera una risposta.")

    with tab_step5:
        st.markdown("#### Dalle fonti selezionate alla risposta")
        st.write(f"{answer_model_v2} riceve la domanda e {len(reranked_v2)} passaggi. È istruito a rispondere usando questi contenuti e a citarli. Ricevere un passaggio non significa citarlo; una citazione presente non certifica che il dato sia corretto.")
        st.markdown("##### Quali fonti ha ricevuto e quali ha citato?")
        st.write(f"La ricerca ha selezionato {len(raw_v2)} candidati. Dopo il reranking, solo i primi {len(reranked_v2)} sono stati inviati al LLM. La tabella mostra questi ultimi: ‘Citato’ indica se il modello li ha richiamati nella risposta.")
        st.dataframe([{
            "Riferimento nel prompt": f"#{number}",
            "File": doc.get("filename"), "Pagina": _page_label(doc.get("page_number")),
            "Citato nella risposta": "Sì" if number in citations_v2 else "No",
        } for number, doc in enumerate(reranked_v2, 1)], hide_index=True, use_container_width=True)
        st.caption("Più righe dello stesso file indicano estratti diversi. I candidati esclusi dal LLM e i cambiamenti di posizione sono nel passo 4.")
        with st.expander("Mostra il prompt inviato al LLM"):
            st.caption("Le istruzioni di sistema effettivamente inviate al modello per questa risposta.")
            system_prompts = []
            for attempt in reviewed_v2["attempts"]:
                messages = attempt["response"].get("request_messages") or []
                for message in messages:
                    if message.get("role") == "system" and message.get("content"):
                        if message["content"] not in system_prompts:
                            system_prompts.append(message["content"])
            if not system_prompts:
                st.info("Il prompt di sistema non è stato registrato per questa esecuzione. Sarà disponibile dalla prossima domanda.")
            for prompt in system_prompts:
                st.code(prompt, language="text")


# ─── QUERY — la pagina ─────────────────────────────────────────────────────
if _page == "query":
    selected_question = st.selectbox(
        "Domanda", options=["Domanda libera"] + demo_questions,
        key="demo_question_selection",
    )
    # Preserve the free-text draft when its widget is hidden by a preset.
    if not st.session_state.get("free_question_empty_default_initialized"):
        st.session_state.free_question_draft = ""
        st.session_state.free_question_input = ""
        st.session_state.free_question_empty_default_initialized = True

    # Dentro un form: scrivere o incollare la domanda non riesegue la pagina, solo «Esegui» lo fa.
    with st.form("query_form", border=False):
        if selected_question == "Domanda libera":
            if "free_question_input" not in st.session_state:
                st.session_state.free_question_input = st.session_state.free_question_draft
            query_v2 = st.text_input(
                "Scrivi la tua domanda", placeholder="Inserisci una domanda sui documenti caricati",
                key="free_question_input",
            ).strip()
        else:
            query_v2 = selected_question
        rerank_instructions_v2 = st.text_area(
            "Istruzioni al reranker (opzionale)", value="", key="query_rerank_instructions",
            placeholder="Es. Privilegia le spiegazioni tecniche; dai meno rilevanza alle semplici menzioni dell'argomento.",
            help="Indica quali passaggi privilegiare tra quelli trovati. Lascia vuoto per ordinare in base alla sola domanda.",
        ).strip()
        _run_query_v2 = st.form_submit_button("Esegui", type="primary", key="run_v2")
    if selected_question == "Domanda libera":
        st.session_state.free_question_draft = query_v2
    if _run_query_v2 and not query_v2:
        st.warning("Scrivi una domanda prima di eseguire la ricerca.")

    query_scope_v2 = st.session_state.embed_mode
    st.caption(f"La ricerca usa solo la modalità attiva `{query_scope_v2}`. Per cercare in un'altra modalità, "
               "caricane i documenti da Upload o cambia la modalità attiva in Admin. "
               "Gli eventuali filtri sui documenti si applicano a tutti i percorsi.")
    try:
        render_scope(preview_scope(query_scope_v2))
    except Exception as exc:
        st.warning(f"Inventario non disponibile: {exc}. Verifica la connessione in Admin.")

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
        try:
            available_v2 = _available_metadata_filters()
            doc_langs_v2 = _dominant_doc_langs()
        except Exception as metadata_error:
            available_v2 = {}
            doc_langs_v2 = {}
            st.warning(f"Filtri metadata non disponibili: {type(metadata_error).__name__}. Verifica la connessione MongoDB.")
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

    if _run_query_v2 and query_v2:
        filters_v2 = _build_adaptive_filters(filter_values_v2)
        timings_ms_v2: dict[str, float] = {}
        try:
            scope_v2 = require_scope(query_scope_v2)
            t0 = time.perf_counter()
            with st.spinner("Preparo la domanda con il modello di ciascun percorso…"):
                query_vectors_v2 = prepare_vectors(query_v2, scope_v2["modes"])
            if query_vectors_v2:
                timings_ms_v2["Embedding domanda"] = (time.perf_counter() - t0) * 1000
            pipeline_builder_v2 = build_native_rerank_pipeline if config.RERANK_BACKEND == "native" else build_pipeline
            rerank_options_v2 = {
                "top_n": top_n_v2, "instructions": rerank_instructions_v2,
                "include_candidates": True,
            } if config.RERANK_BACKEND == "native" else {"retrieval_mode": "hybrid"}
            pipeline_v2 = pipeline_builder_v2(
                query_v2, filters=filters_v2, top_k=top_k_v2,
                embed_mode=query_scope_v2, fuzzy=fuzzy_v2,
                modes=scope_v2["modes"], query_vectors=query_vectors_v2,
                **rerank_options_v2,
            )
        except Exception as exc:
            st.error(f"Preparazione della ricerca non completata: {exc}")
            st.stop()
        # 1. Retrieval ibrido principale
        t0 = time.perf_counter()
        try:
            with st.spinner("Ricerca ibrida e reranking Atlas…" if config.RERANK_BACKEND == "native" else "Ricerca ibrida…"):
                atlas_results_v2 = list(_coll().aggregate(pipeline_v2))
                raw_v2 = sorted(atlas_results_v2, key=lambda d: d["_hybrid_rank"]) if config.RERANK_BACKEND == "native" else atlas_results_v2
        except Exception as e:
            err_msg = str(e)
            if "INITIAL_SYNC" in err_msg or "while in state" in err_msg:
                st.warning("🛠 Indice in costruzione, aspetta 30-60s e riprova.")
            elif "HostUnreachable" in err_msg or ":28000" in err_msg:
                st.error("🚨 Atlas Search non raggiungibile dal cluster.")
            elif "is not indexed as vector" in err_msg or "queryVector must be present" in err_msg:
                st.error("🚨 Schema indice incoerente con la modalità embedding. Vai in Admin.")
            else:
                st.error("Atlas non ha completato la pipeline. Controlla il dettaglio sotto; per $rerank verifica abilitazione del progetto e disponibilità del modello.")
            with st.expander("Errore tecnico"):
                st.code(err_msg)
            st.stop()
        timings_ms_v2["Ricerca + rerank Atlas" if config.RERANK_BACKEND == "native" else "hybrid"] = (time.perf_counter() - t0) * 1000

        if not raw_v2:
            st.warning(
                "Nessun risultato trovato con la query e i filtri selezionati."
            )
            st.caption(
                f"Filtri applicati: `{_filter_summary(filters_v2)}`. "
                "Modifica o rimuovi uno dei filtri e riprova."
            )
            st.stop()

        # Il percorso nativo ha già riordinato gli stessi candidati in Atlas.
        t0 = time.perf_counter()
        rerank_mode_v2 = config.RERANK_BACKEND
        reranked_v2: list[dict] = []
        if config.RERANK_BACKEND == "native":
            reranked_v2 = atlas_results_v2[:top_n_v2]
            for rank, doc in enumerate(reranked_v2, start=1):
                doc["_rerank_rank"] = rank
        elif config.VOYAGE_API_KEY:
            with st.spinner("Reranking Voyage API…"):
                reranked_v2 = rerank(query_v2, raw_v2, top_n=top_n_v2,
                                     instructions=rerank_instructions_v2)
        else:
            rerank_mode_v2 = "disabled"
            st.warning("VOYAGE_API_KEY mancante: uso direttamente i primi risultati RRF.")
            reranked_v2 = raw_v2[:top_n_v2]
            for c in reranked_v2:
                c["_rerank_score"] = None

        if config.RERANK_BACKEND != "native":
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

        # Run providers only on an explicit click, never while inspecting results.
        if not llm_available():
            st.info("LLM disattivato. La generazione non è disponibile.")
            st.stop()
        generation_started_v2 = time.perf_counter()
        try:
            with st.spinner("Generazione della risposta…"):
                reviewed_v2 = generate_single_answer(query_v2, reranked_v2)
        except Exception as exc:
            st.error("La generazione della risposta non è riuscita.")
            with st.expander("Dettaglio tecnico"):
                st.code(str(exc))
            st.stop()
        timings_ms_v2["Generazione"] = (time.perf_counter() - generation_started_v2) * 1000

        # Le classifiche di confronto (passo 2) non vengono calcolate qui:
        # la risposta compare subito e le classifiche si calcolano su richiesta.

        st.session_state.last_query_result = {
            "query_v2": query_v2,
            "rerank_instructions_v2": rerank_instructions_v2,
            "filters_v2": filters_v2,
            "scope_v2": scope_v2,
            "pipeline_v2": pipeline_v2,
            "top_k_v2": top_k_v2,
            "relevance_threshold_v2": relevance_threshold_v2,
            "timings_ms_v2": timings_ms_v2,
            "atlas_results_v2": atlas_results_v2,
            "raw_v2": raw_v2,
            "reranked_v2": reranked_v2,
            "rerank_mode_v2": rerank_mode_v2,
            "top_rerank_v2": top_rerank_v2,
            "reviewed_v2": reviewed_v2,
            "text_deep_v2": None,
            "vector_by_mode_v2": None,
            "fuzzy_v2": fuzzy_v2,
            "query_vectors_v2": query_vectors_v2,
        }

    if result := st.session_state.get("last_query_result"):
        _render_query_result(result)


_AGENT_STOP_SHORT = {
    "evidence_sufficient": "si è fermato perché le prove bastano",
    "limit_reached": "si è fermato al limite di ricerche, con prove forse incomplete",
    "awaiting_feedback": "si è fermato dopo la prima ricerca e attende il tuo feedback",
    "no_followup_query": "si è fermato: manca qualcosa ma il valutatore non ha proposto una nuova ricerca",
    "duplicate_query": "si è fermato: la nuova ricerca ripeteva una già fatta",
    "rephrased_duplicate": "si è fermato: la nuova ricerca ripeteva una già fatta con altre parole",
    "evaluation_unavailable": "si è fermato: la valutazione non è riuscita",
    "off_topic": "domanda fuori tema: il passaggio migliore è sotto la soglia, il modello non è stato chiamato",
    "no_results": "la ricerca non ha restituito passaggi",
    "needs_clarification": "si è fermato per chiederti l’anno: le fonti riportano valori diversi",
}
_AGENT_GRADE_FAILURES = {
    "timeout": "tempo scaduto",
    "connection": "servizio non raggiungibile",
    "service_error": "errore del servizio",
    "request_error": "richiesta non riuscita",
    "invalid_response": "risposta del modello non leggibile",
    "unverifiable_evidence": "estratti non verificabili nei passaggi",
}


def _agent_ref_label(refs: list[dict], ref: Any) -> str:
    if isinstance(ref, int) and 1 <= ref <= len(refs):
        item = refs[ref - 1]
        return f"#{ref}, {short_file(item.get('filename') or 'Documento')} {_page_label(item.get('page'), prefix='p. ')}"
    return f"#{ref}" if ref else ""


def _agent_journey_summary(events: list[dict]) -> str:
    searches = sum(e.get("type") == "retrieval" for e in events)
    last_grade = next((e for e in reversed(events) if e.get("type") == "grading"), None)
    total = sum(e.get("duration_ms") or 0 for e in events)
    parts = [f"{searches} {'ricerca' if searches == 1 else 'ricerche'}"]
    if any(e.get("type") == "clarification" for e in events):
        parts.append("valori diversi per anno: ti chiede quale")
    elif last_grade:
        parts.append("prove sufficienti" if last_grade.get("sufficient") else "prove incomplete")
    if total:
        parts.append(_format_ms(total))
    return " · ".join(parts)


def _render_agent_journey(events: list[dict], status: str, step: str = "", max_searches: int | None = None,
                          run: dict | None = None) -> None:
    """Percorso compatto: una riga per passo, con il tempo; i dettagli dietro un interruttore."""
    plan = next((event for event in events if event.get("type") == "plan"), {})
    max_searches = max_searches if max_searches is not None else plan.get("max_searches")
    searches = [event for event in events if event.get("type") == "retrieval"]
    gradings = [event for event in events if event.get("type") == "grading"]
    answer_event = next((event for event in reversed(events) if event.get("type") == "answer"), None)
    stop_decision = next((event for event in reversed(events)
                          if event.get("type") == "decision" and event.get("action") == "stop"), None)

    rows: list[dict[str, Any]] = []
    for event in events:
        kind = event.get("type")
        duration = event.get("duration_ms")
        if kind == "plan":
            if event.get("planner_skipped"):
                rows.append({"icon": "🧭", "title": "Domanda usata così com’è",
                             "detail": "Prima domanda della sessione: nessun riferimento da risolvere.", "ms": None})
            else:
                rows.append({"icon": "🧭", "title": "Domanda interpretata",
                             "detail": event.get("planned_query") or event.get("search_query", ""), "ms": duration})
            if event.get("planner_error"):
                rows.append({"icon": "⚠️", "title": "Il pianificatore non ha risposto",
                             "detail": f"Uso la domanda originale · {event['planner_error']}", "ms": None})
        elif kind == "retrieval":
            iteration = event.get("iteration", 1)
            top = " · ".join(f"{short_file(ref.get('filename') or 'Documento')} {_page_label(ref.get('page'), prefix='p. ')}"
                             for ref in (event.get("ranked_sources") or [])[:3])
            if iteration > 1:
                title = f"Ricerca {iteration}, mirata · «{event.get('search_query', '')}»"
            else:
                title = f"Ricerca {iteration} · {event.get('candidate_count', 0)} candidati"
            rows.append({"icon": "🔍", "title": title, "detail": ("In cima: " + top) if top else "Nessun passaggio trovato.",
                         "ms": duration, "kind": "atlas"})
        elif kind == "agent_rank_fusion" and rows:
            reranked = event.get("reranked")
            rows[-1]["detail"] += (f" — unite alle precedenti: {event.get('unique_candidates', 0)} passaggi distinti"
                                   + (f", i primi {reranked} riordinati sulla domanda" if reranked else ", riordinati sulla domanda"))
            if event.get("duration_ms") is not None:
                rows[-1]["ms"] = (rows[-1]["ms"] or 0) + event["duration_ms"]
        elif kind == "grading":
            refs = event.get("evaluated_source_refs") or []
            elements = event.get("elements") or []
            if not event.get("evaluated", True):
                rows.append({"icon": "⚠️", "title": "Valutazione non riuscita",
                             "detail": _AGENT_GRADE_FAILURES.get(event.get("failure_reason"), "motivo non registrato")
                             + (f" · {event['failure_detail']}" if event.get("failure_detail") else ""),
                             "ms": duration, "kind": "agent"})
                continue
            found = [e for e in elements if e.get("found")]
            missing = [e.get("element") for e in elements if not e.get("found")] or event.get("missing_elements") or []
            if event.get("sufficient"):
                quote = (event.get("supporting_quotes") or [{}])[0]
                detail = (f"«{quote.get('quote', '')}» · {_agent_ref_label(refs, quote.get('ref'))}"
                          if quote.get("quote") else event.get("comment", ""))
                if found:
                    detail = " · ".join(
                        f"{e['element']}" + (f" = {e['value']}" if e.get("value") else "") + f" ({_agent_ref_label(refs, e.get('ref'))})"
                        for e in found)
                rows.append({"icon": "✅", "title": "Valutazione: prove sufficienti", "detail": detail,
                             "ms": duration, "kind": "agent"})
            else:
                detail = ("Trovato: " + " · ".join(
                    f"{e['element']}" + (f" = {e['value']}" if e.get("value") else "") + f" ({_agent_ref_label(refs, e.get('ref'))})"
                    for e in found)
                          if found else event.get("comment", ""))
                unverified = event.get("unverified_elements") or []
                if unverified:
                    detail += (" · Indicato come trovato ma senza un estratto verificabile: " + "; ".join(unverified))
                rows.append({"icon": "⚠️", "title": "Valutazione: manca " + ("; ".join(missing) if missing else "qualcosa"),
                             "detail": detail, "ms": duration, "kind": "agent"})
        elif kind == "clarification":
            detail = " · ".join(f"{o['year']}: " + ", ".join(o.get("values", {}).values()) for o in event.get("options", []))
            rows.append({"icon": "❓", "title": "Valori diversi in anni diversi: chiede quale anno",
                         "detail": detail, "ms": None})
        elif kind == "answer" and event.get("status") == "needs_input":
            continue
        elif kind == "answer":
            reason = (stop_decision or {}).get("reason")
            stop_text = _AGENT_STOP_SHORT.get(reason, event.get("stop_reason") or "")
            if event.get("status") == "blocked":
                block = (event.get("diagnostics") or {}).get("block_reason")
                title = "Risposta non generata" if block in {"off_topic", "no_results"} else "Risposta bloccata dal controllo"
                rows.append({"icon": "🚫", "title": title, "detail": stop_text[:1].upper() + stop_text[1:] if stop_text else "",
                             "ms": event.get("duration_ms"), "kind": "answer"})
            else:
                rows.append({"icon": "💬", "title": f"Risposta con {len(event.get('sources') or [])} fonti",
                             "detail": stop_text[:1].upper() + stop_text[1:] if stop_text else "",
                             "ms": event.get("duration_ms"), "kind": "answer"})

    model_calls = (0 if plan.get("planner_skipped") or not plan else 1) + len(gradings) + (
        (answer_event or {}).get("attempts") or 0)
    timed = [row["ms"] for row in rows if row.get("ms")]
    summary = f"{len(searches)} {'ricerca' if len(searches) == 1 else 'ricerche'}" + (f" su {max_searches}" if max_searches else "")
    summary += f" · {model_calls} chiamate al modello"
    if timed:
        summary += f" · {_format_ms(sum(timed))}"
    if status == "running":
        summary += " · in corso…"

    body = "".join(
        f'<div style="display:grid;grid-template-columns:26px minmax(0,1fr) 64px;gap:0 10px;padding:7px 0;'
        f'border-bottom:1px solid rgba(128,128,128,0.16)">'
        f'<div style="text-align:center">{row["icon"]}</div>'
        f'<div><div style="font-weight:600">{html.escape(row["title"])}</div>'
        + (f'<div style="opacity:0.72;font-size:0.88em">{html.escape(row["detail"])}</div>' if row.get("detail") else "")
        + f'</div><div style="text-align:right;opacity:0.65;font-size:0.88em">'
        f'{_format_ms(row["ms"]) if row.get("ms") else ""}</div></div>'
        for row in rows
    )
    bar = ""
    groups = {"atlas": ("Atlas: ricerca e reranking", "#1D9E75"), "agent": ("Decisioni dell’agente", "#AFA9EC"),
              "answer": ("Risposta", "#7F77DD")}
    totals = {key: sum(row["ms"] for row in rows if row.get("kind") == key and row.get("ms")) for key in groups}
    if plan and not plan.get("planner_skipped") and plan.get("duration_ms"):
        totals["agent"] += plan["duration_ms"]
    grand = sum(totals.values())
    if grand:
        segments = "".join(
            f'<div style="width:{max(row["ms"] / grand * 100, 1):.2f}%;background:{groups[row["kind"]][1]}"></div>'
            for row in rows if row.get("kind") in groups and row.get("ms"))
        legend = "".join(
            f'<span style="white-space:nowrap"><span style="display:inline-block;width:8px;height:8px;border-radius:50%;'
            f'background:{color};margin-right:5px"></span>{html.escape(label)} {_format_ms(totals[key])}</span>'
            for key, (label, color) in groups.items() if totals[key])
        bar = (f'<div style="display:flex;height:9px;border-radius:5px;overflow:hidden;margin:12px 0 6px">{segments}</div>'
               f'<div style="display:flex;gap:16px;flex-wrap:wrap;font-size:0.82em;opacity:0.75">{legend}</div>')
    # Nella ricerca salvata il titolo e il riepilogo sono già nell'intestazione del pannello.
    header = "" if run else (
        '<div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px;margin-bottom:4px">'
        '<div style="font-weight:600;font-size:1.05em">Il percorso dell’agente</div>'
        f'<div style="opacity:0.72;font-size:0.88em">{html.escape(summary)}</div></div>')
    st.markdown(
        '<div style="border:1px solid rgba(128,128,128,0.22);border-radius:10px;padding:10px 14px;margin:8px 0">'
        f'{header}{body}{bar}</div>',
        unsafe_allow_html=True,
    )
    if status == "failed":
        st.warning("Ricerca interrotta: il percorso mostra i passi completati prima dell’errore.")
    elif status == "running_saved":
        st.warning("Esecuzione salvata come in corso. Dopo un riavvio non riparte automaticamente: avvia una nuova ricerca.")
    if status == "running":
        return

    toggle_key = f"agent_tech_{(run or {}).get('_id', 'live')}"
    if not st.toggle("Mostra il dettaglio tecnico", key=toggle_key):
        return
    with st.container(border=True):
        if plan.get("database_search"):
            db = plan["database_search"]
            st.caption(f"Collection `{db['database']}.{db['collection']}` · indice testuale `{db['text_index']}` · "
                       + " · ".join(f"vettoriale `{row['index']}` ({row['mode']})" for row in db.get("vector_indexes", []))
                       + f" · filtri: {_filter_summary(plan.get('filters') or {})}")
        for event in searches:
            st.markdown(f"**Ricerca {event.get('iteration', 1)}** — query completa")
            st.code(event.get("search_query", ""), language=None)
        for event in gradings:
            st.markdown(f"**Valutazione {event.get('iteration', 1)}**")
            if event.get("comment"):
                st.write(event["comment"])
            for element in event.get("elements") or []:
                mark = "✓" if element.get("found") else ("≈ indicato ma non ritrovato nel passaggio" if element.get("claimed_found") else "✗")
                value = element.get("value") or element.get("quote")
                st.caption(f"{mark} {element.get('element')}" + (f" = {value}" if value else "")
                           + (f" · passaggio #{element['ref']}" if element.get("ref") else ""))
            for i, ref in enumerate(event.get("evaluated_source_refs") or [], 1):
                st.caption(f"#{i} · {ref.get('filename') or 'Documento'} · {_page_label(ref.get('page'))}")
                if ref.get("excerpt"):
                    st.text(ref["excerpt"])
        if run:
            st.markdown("**Documento salvato in MongoDB (`agentic_runs`)**")
            st.caption("Stato e decisioni osservabili dell’agente, non il ragionamento interno del modello.")
            st.json({"run_id": run.get("_id"), "status": run.get("status"), "state": run.get("state"),
                     "events": run.get("events", [])}, expanded=False)


def _agent_run_sources(run: dict, memory: list[dict]) -> tuple[str, list[dict]]:
    """Use snapshots for new runs and recover older references when possible."""
    result = run.get("result") or {}
    message = next((m for m in memory if m.get("turn_id") == run.get("_id") and m.get("role") == "assistant"), {})
    answer_event = next((event for event in reversed(run.get("events", [])) if event.get("type") == "answer"), {})
    answer = result.get("answer") or message.get("text") or ""
    refs = result.get("sources") or message.get("sources") or answer_event.get("sources") or []
    sources = []
    for ref in refs:
        source = dict(ref)
        if not source.get("content") and ObjectId.is_valid(str(source.get("chunk_id", ""))):
            try:
                doc = _coll().find_one({"_id": ObjectId(str(source["chunk_id"]))},
                                       {"content": 1, "gridfs_file_id": 1, "filename": 1, "page_number": 1})
                if doc:
                    source = {**doc, **{key: value for key, value in source.items() if value not in (None, "")}}
            except Exception:
                pass
        sources.append(source)
    return answer, sources


def _agent_verdict(run: dict, answer: str, sources: list[dict]) -> dict[str, Any]:
    """Unisce i controlli in un solo verdetto.

    Distingue due cose diverse: un RISCHIO (numeri non ritrovati nel passaggio citato,
    citazioni mancanti o sbagliate) da una LACUNA (un dato richiesto che i documenti
    semplicemente non contengono, e che la risposta dichiara). Solo il rischio è "Da verificare".
    ok: True verificata, False da verificare, None informativo (parziale o assente).
    """
    events = run.get("events", [])
    grade = next((e for e in reversed(events) if e.get("type") == "grading"), None)
    answer_event = next((e for e in reversed(events) if e.get("type") == "answer"), {})
    rows = answer_event.get("citation_check")
    if rows is None:  # esecuzioni precedenti: verifica sugli estratti salvati
        rows = check_citations(answer, sources)
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
        risks.append(f"{len(not_found)} numeri non compaiono nel passaggio citato ({', '.join(not_found[:4])})")
    if uncited:
        risks.append(f"{len(uncited)} numeri senza citazione ({', '.join(uncited[:4])})")
    elements = (grade or {}).get("elements") or []
    found_elements = [e for e in elements if e.get("found")]
    gaps = [e.get("element") for e in elements if not e.get("found")]
    searches = sum(e.get("type") == "retrieval" for e in events)
    checked = len(rows) - len(uncited)
    found_numbers = sum(r["status"] in {"found", "rounded"} for r in rows)

    if risks:
        return {"ok": False, "label": "Da verificare", "problems": risks, "rows": rows}
    if grade and elements and not found_elements and found_numbers:
        # Il valutatore non ha riconosciuto i dati, ma la risposta riporta numeri ritrovati nelle fonti.
        return {"ok": None, "label": "Parziale: dati verificati, risposta forse incompleta", "rows": rows,
                "problems": [f"i {found_numbers} numeri riportati sono nelle fonti citate; il valutatore non li "
                             "considera una risposta completa a: " + "; ".join(gaps)]}
    if grade and elements and not found_elements:
        return {"ok": None, "label": "Non presente nei documenti", "rows": rows,
                "problems": [f"l’agente ha cercato {searches} {'volta' if searches == 1 else 'volte'} e i documenti "
                             "non contengono questi dati: non risponde a memoria"]}
    if gaps:
        return {"ok": None, "label": f"Parziale: {len(found_elements)} dati su {len(elements)} nei documenti",
                "rows": rows,
                "problems": ["i numeri riportati sono verificati; i documenti non contengono: " + "; ".join(gaps)
                             + (f" (cercato con {searches} ricerche)" if searches > 1 else "")]}
    if checked:
        label = f"Verificata: {found_numbers} numeri su {checked} trovati nelle fonti"
    else:
        label = "Verificata: prove trovate dal valutatore"
    return {"ok": True, "label": label, "problems": [], "rows": rows}


def _render_agent_clarification(run: dict) -> None:
    """L'agente ha trovato valori diversi in anni diversi: chiede quale anno usare."""
    event = next((e for e in reversed(run.get("events", [])) if e.get("type") == "clarification"), {})
    question = run.get("question") or ""
    st.markdown("#### L’agente ha una domanda per te")
    st.write("Ho trovato valori diversi in anni diversi per: **" + "; ".join(event.get("elements", [])) + "**. "
             "Quale anno ti interessa?")

    def _choose(payload: dict) -> None:
        st.session_state.agentic_clarification = payload

    for option in event.get("options", []):
        values = " · ".join(f"{name}: {value}" for name, value in option.get("values", {}).items())
        label = f"{option['year']} — {values}" + (f"  ({short_file(option['filename'])})" if option.get("filename") else "")
        st.button(label, key=f"clar_{run['_id']}_{option['year']}", use_container_width=True,
                  on_click=_choose, args=({
                      "session_id": run.get("session_id"), "parent_run_id": run["_id"],
                      "question": f"{question} (anno {option['year']})",
                      "context": (f"L’utente ha scelto l’anno {option['year']}. Usa solo i dati di quell’anno"
                                  + (f", dal documento {option['filename']}." if option.get("filename") else ".")),
                      "filters": {"filename": option["filename"]} if option.get("filename") else None,
                  },))
    st.button("⇄ Confronta gli anni", key=f"clar_{run['_id']}_compare", use_container_width=True, type="primary",
              on_click=_choose, args=({
                  "session_id": run.get("session_id"), "parent_run_id": run["_id"],
                  "question": f"{question} Confronta i valori per ciascun anno disponibile.",
                  "context": "L’utente vuole il confronto tra gli anni: indica sempre l’anno di ogni valore.",
                  "skip_ambiguity": True,
              },))


def _render_agent_result(run: dict, memory: list[dict]) -> None:
    """Risposta, verdetto di verifica e fonti citate. Il resto è a un clic."""
    answer, sources = _agent_run_sources(run, memory)
    status = run.get("status")
    with st.container(border=True):
        st.caption("Domanda: " + (run.get("question") or "non disponibile"))
        if status == "failed":
            st.warning("La ricerca si è interrotta prima di completare la risposta.")
            return
        if status == "running":
            st.info("Ricerca salvata come in corso: dopo un riavvio non riparte. Avvia una nuova ricerca.")
            return
        if status == "needs_input":
            _render_agent_clarification(run)
            return
        if status == "blocked":
            st.warning(answer or "Risposta non disponibile.")
            answer_event = next((e for e in reversed(run.get("events", [])) if e.get("type") == "answer"), {})
            diagnostics = answer_event.get("diagnostics")
            if diagnostics and diagnostics.get("block_reason") not in {"off_topic", "no_results"}:
                _render_answer_diagnostics(diagnostics)
            return
        if not answer:
            st.info("La risposta non è disponibile per questa esecuzione.")
            return

        verdict = _agent_verdict(run, answer, sources)
        title_col, badge_col = st.columns([0.45, 0.55], vertical_alignment="center")
        title_col.markdown("#### Risposta")
        if verdict["ok"] is None:
            bg, fg, mark = "rgba(128,128,128,0.16)", "inherit", "○"
        elif verdict["ok"]:
            bg, fg, mark = "rgba(29,158,117,0.16)", ("#9FE1CB" if _theme_is_dark() else "#0F6E56"), "✓"
        else:
            bg, fg, mark = "rgba(239,159,39,0.2)", ("#FAC775" if _theme_is_dark() else "#854F0B"), "⚠"
        badge_col.markdown(
            f'<div style="text-align:right"><span style="background:{bg};color:{fg};font-size:0.85em;'
            f'padding:4px 12px;border-radius:6px">{mark} {html.escape(verdict["label"])}</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(_normalize_inline_citations(answer))
        if verdict["problems"]:
            prefix = "" if verdict["ok"] is None else "Da controllare: "
            text = "; ".join(verdict["problems"]) + "."
            st.caption(prefix + (text[:1].upper() + text[1:] if not prefix else text))

        cited = [n for n in inspect_answer(answer, len(sources))["valid_citations"] if 1 <= n <= len(sources)]
        excerpts: dict[int, list[str]] = {}
        for r in verdict["rows"]:
            if r.get("excerpt") and r.get("ref") and r["excerpt"]["match"] not in excerpts.get(r["ref"], []):
                excerpts.setdefault(r["ref"], []).append(r["excerpt"]["match"])
        if cited:
            st.markdown("**Fonti citate**")
            shown_files: set[str] = set()
            for number in cited:
                source = sources[number - 1]
                name = source.get("filename") or "Documento"
                page = _page_label(source.get("page", source.get("page_number")), prefix="p. ")
                text_col, button_col = st.columns([0.84, 0.16], vertical_alignment="center")
                line = f"[#{number}] {short_file(name)} · {page}"
                if number in excerpts:
                    line += " · " + " ".join(f"«{match}»" for match in excerpts[number])
                text_col.markdown(html.escape(line))
                file_id = source.get("gridfs_file_id")
                if file_id and str(file_id) not in shown_files:
                    shown_files.add(str(file_id))
                    data = _gridfs_pdf_bytes(str(file_id))
                    if data:
                        button_col.download_button("PDF", data, file_name=name, mime="application/octet-stream",
                                                   on_click="ignore", key=f"agent_source_{run['_id']}_{number}",
                                                   use_container_width=True)
        others = [n for n in range(1, len(sources) + 1) if n not in cited]
        with st.expander("Come è stata verificata" + (f" · altre {len(others)} fonti esaminate" if others else ""),
                         expanded=False):
            _render_citation_check(answer, sources, rows=verdict["rows"], title=False)
            st.caption("Controllo meccanico: dice se i numeri vengono dai documenti, non se il modello li ha interpretati bene.")
            if others:
                st.markdown("**Fonti date al modello e non citate**")
                for number in others:
                    source = sources[number - 1]
                    st.caption(f"#{number} · {short_file(source.get('filename') or 'Documento')} · "
                               f"{_page_label(source.get('page', source.get('page_number')), prefix='p. ')}")


def _render_agent_feedback_controls(run: dict, sessions, runs, session_id: str,
                                    available: dict, current_settings: dict) -> None:
    """Persist the decision first; a refinement always requires a new explicit question."""
    if run.get("status") not in {"complete", "blocked"}:
        return
    run_id = run["_id"]
    history = run.get("feedback") or []
    latest = history[-1] if history else {}
    # Una sola azione, discreta: riformulare. L'agente chiede da solo quando non è sicuro.
    if latest.get("action") != "needs_refinement":
        if st.button("↻ Non ti convince? Riformula la domanda", key=f"agent_refine_{run_id}", type="tertiary"):
            _agent_record_feedback(runs, sessions, session_id=session_id, run_id=run_id, action="needs_refinement")
            st.rerun()

    if latest.get("action") == "refinement_submitted":
        child = runs.find_one({"_id": latest["id"], "session_id": session_id}, {"_id": 1, "status": 1})
        if child:
            st.caption(f"Approfondimento collegato: {child.get('status', 'stato non disponibile')}. Selezionalo in «Rivedi una ricerca precedente».")
        elif st.button("▶️ Avvia l’approfondimento salvato", key=f"agent_resume_{latest['id']}"):
            st.session_state.agentic_pending_refinement = {"session_id": session_id, "run_id": run_id, "id": latest["id"]}
            st.rerun()

    if latest.get("action") != "needs_refinement":
        return

    previous_settings = run.get("settings") or next(
        (event.get("settings") for event in run.get("events", []) if event.get("type") == "plan" and event.get("settings")),
        {},
    )
    old_plan = next((event for event in run.get("events", []) if event.get("type") == "plan"), {})
    prior_filters = previous_settings.get("filters", old_plan.get("filters", {})) or {}
    top_k = min(100, max(5, int(previous_settings.get("top_k", current_settings["top_k"])) ))
    top_n = min(20, max(1, int(previous_settings.get("top_n", current_settings["top_n"])) ))
    prior_instructions = previous_settings.get("rerank_instructions", current_settings.get("rerank_instructions", ""))

    st.markdown("#### Prepara una nuova ricerca")
    st.caption("Riscrivi una domanda completa e diversa: indica soggetto, periodo, dato e fonte desiderata. Le indicazioni aggiuntive non vengono trattate come fatti verificati.")
    with st.form(f"agent_refinement_form_{run_id}", clear_on_submit=False):
        reason = st.selectbox("Che cosa va migliorato?", [
            "Risposta incompleta", "Risposta errata", "Fonte errata o non verificabile", "Altro",
        ], key=f"agent_refinement_reason_{run_id}")
        question = st.text_area("Nuova domanda completa *", key=f"agent_refinement_question_{run_id}",
                                placeholder="Es. Quali sono i ricavi AWS nel 2022 e quale unità indica la tabella?",
                                max_chars=4000, height=100)
        details = st.text_area("Cosa manca o quali informazioni aggiungere? (facoltativo)",
                               key=f"agent_refinement_details_{run_id}", max_chars=2000,
                               placeholder="Es. Cerca l’intestazione della tabella e il documento originale.")
        st.markdown("**Filtri sui documenti**")
        filter_values = {}
        fields = [("filename", "Documento", "(tutti)"), ("category", "Categoria", "(tutte)"),
                  ("author", "Autore", "(tutti)"), ("year", "Anno", "(tutti gli anni)")]
        cols = st.columns(2)
        for index, (field, label, default) in enumerate(fields):
            values = list(available.get(field, []))
            prior = prior_filters.get(field, default)
            if prior != default and prior not in values:
                values.append(prior)
            choices = [default] + values
            if values:
                filter_values[field] = cols[index % 2].selectbox(
                    label, choices, index=choices.index(prior), key=f"agent_refinement_{run_id}_{field}")
        k = st.slider("Passaggi candidati per la nuova ricerca", 5, 100, top_k, key=f"agent_refinement_k_{run_id}")
        n = st.slider("Passaggi da usare nella nuova risposta", 1, 20, top_n, key=f"agent_refinement_n_{run_id}")
        instructions = st.text_input("Indicazioni al reranker (facoltative)", value=prior_instructions,
                                     key=f"agent_refinement_rerank_{run_id}", max_chars=2000)
        st.caption("La nuova ricerca si ferma dopo un tentativo e attende nuovamente il tuo feedback.")
        submitted = st.form_submit_button("Salva feedback e avvia la nuova ricerca", type="primary")
        if submitted:
            try:
                question = _agent_validate_refinement(run.get("question", ""), question, top_k=k, top_n=n)
            except ValueError as exc:
                st.warning(str(exc))
            else:
                context = f"{reason}." + (f" {details.strip()}" if details.strip() else "")
                settings = {"interaction_mode": "Con feedback", "top_k": k, "top_n": n, "retries": 0,
                            "filters": _build_adaptive_filters(filter_values) or {},
                            "rerank_instructions": instructions.strip()}
                entry = _agent_record_feedback(
                    runs, sessions, session_id=session_id, run_id=run_id, action="refinement_submitted",
                    details={"question": question, "context": context, "reason": reason, "settings": settings},
                )
                st.session_state.agentic_pending_refinement = {"session_id": session_id, "run_id": run_id, "id": entry["id"]}
                st.rerun()


# ─── AGENTIC SEARCH — memoria + retrieval iterativo ──────────────────────────
if _page == "agentic":
    st.title("🧭 Ricerca agentica")
    st.caption("L’agente cerca, controlla se le prove bastano e, se serve, cerca ancora. "
               "Stessi indici della pagina Query; la conversazione è salvata in MongoDB.")

    try:
        _agent_sessions, _agent_runs = _get_agent_collections()
        _recent_agent_sessions = list(
            _agent_sessions.find({}, {"title": 1, "updated_at": 1, "memory.turn_count": 1, "active_run": 1})
            .sort("updated_at", -1).limit(20)
        )
        _session_ids = {row["_id"] for row in _recent_agent_sessions}
        def _empty_or_new_session() -> dict:
            """Riusa una sessione mai usata invece di crearne una nuova a ogni apertura della pagina."""
            empty = _agent_sessions.find_one(
                {"$or": [{"memory.turn_count": 0}, {"memory.turn_count": {"$exists": False}}], "active_run": None},
                sort=[("updated_at", -1)])
            return empty or _agent_create_session(_agent_sessions)

        if st.session_state.get("agentic_session_id") not in _session_ids:
            _new_agent_session = _empty_or_new_session()
            st.session_state.agentic_session_id = _new_agent_session["_id"]
            _recent_agent_sessions = list(
                _agent_sessions.find({}, {"title": 1, "updated_at": 1, "memory.turn_count": 1, "active_run": 1})
                .sort("updated_at", -1).limit(20)
            )

        _session_col, _new_session_col = st.columns([0.82, 0.18], vertical_alignment="bottom")
        _labels_to_ids = {}
        for _session_row in _recent_agent_sessions:
            _updated = _session_row.get("updated_at")
            _when = _updated.strftime("%d/%m %H:%M") if _updated else ""
            _title = _session_row.get("title") or "Sessione agentica"
            _label = f"{_title[:70]} · {_when} · {_session_row['_id'][:6]}"
            _labels_to_ids[_label] = _session_row["_id"]
        _current_label = next(
            (label for label, sid in _labels_to_ids.items() if sid == st.session_state.agentic_session_id),
            next(iter(_labels_to_ids), None),
        )
        if _current_label:
            if st.session_state.get("agentic_session_picker") not in _labels_to_ids:
                st.session_state.pop("agentic_session_picker", None)
            _selected_label = _session_col.selectbox(
                "Sessione persistente", list(_labels_to_ids),
                index=list(_labels_to_ids).index(_current_label), key="agentic_session_picker",
            )
            st.session_state.agentic_session_id = _labels_to_ids[_selected_label]
        if _new_session_col.button("＋ Nuova", key="agentic_new_session", use_container_width=True):
            _created_session = _empty_or_new_session()
            st.session_state.agentic_session_id = _created_session["_id"]
            st.session_state.pop("agentic_session_picker", None)
            st.rerun()

        _agent_session = _agent_sessions.find_one({"_id": st.session_state.agentic_session_id})
        if not _agent_session:
            _agent_session = _agent_create_session(_agent_sessions)
            st.session_state.agentic_session_id = _agent_session["_id"]
        _agent_memory = (_agent_session.get("memory") or {}).get("messages", [])
        _agent_feedback = (_agent_session.get("memory") or {}).get("feedback", [])
        _recent_memory = _agent_memory[-8:]
        _agent_turn_summaries = (_agent_session.get("memory") or {}).get("turn_summaries", [])
        _planning_memory, _planning_summaries = _recent_memory, _agent_turn_summaries
        _pending_preview = st.session_state.get("agentic_pending_refinement")
        if _pending_preview and _pending_preview.get("session_id") == st.session_state.agentic_session_id:
            _preview_parent = _agent_runs.find_one({"_id": _pending_preview.get("run_id"),
                                                     "session_id": st.session_state.agentic_session_id}) or {}
            _preview_entry = next((item for item in _preview_parent.get("feedback", [])
                                   if item.get("id") == _pending_preview.get("id")
                                   and item.get("action") == "refinement_submitted"), None)
            if _preview_entry:
                st.session_state.agentic_question_selection = "Domanda libera"
                st.session_state.agentic_question_draft = _preview_entry.get("question", "")
                st.session_state.agentic_question = _preview_entry.get("question", "")

        # Una sola modalità: l'agente cerca da solo e ti chiede qualcosa solo quando non può decidere.
        _agent_interaction = "Automatica"
        with st.container():
            _agent_selected_question = st.selectbox(
                "Domanda", options=["Domanda libera"] + demo_questions,
                key="agentic_question_selection",
            )
            # Keep the free-text draft even when the widget is hidden by a preset.
            if "agentic_question_draft" not in st.session_state:
                st.session_state.agentic_question_draft = st.session_state.get("agentic_question", "")

            # Dentro un form: scrivere o incollare non riesegue la pagina, solo il pulsante lo fa.
            with st.form("agentic_form", border=False):
                if _agent_selected_question == "Domanda libera":
                    if "agentic_question" not in st.session_state:
                        st.session_state.agentic_question = st.session_state.agentic_question_draft
                    _agent_question = st.text_area(
                        "Fai una domanda sui documenti",
                        placeholder="Esempio: confronta due anni o cerca il passaggio che spiega un’eccezione…",
                        key="agentic_question", height=110,
                    ).strip()
                else:
                    _agent_question = _agent_selected_question
                _agent_submit = st.form_submit_button("Avvia ricerca agentica", type="primary", key="agentic_submit")
            if _agent_selected_question == "Domanda libera":
                st.session_state.agentic_question_draft = _agent_question

        with st.expander("⚙️ Filtri e impostazioni della ricerca", expanded=False):
            _agent_filter_values: dict[str, Any] = {}
            with st.expander("Filtri sui documenti", expanded=False):
                _agent_available = _available_metadata_filters()
                _agent_filter_fields = [
                    ("filename", "Documento", "(tutti)"),
                    ("category", "Categoria", "(tutte)"),
                    ("author", "Autore", "(tutti)"),
                    ("year", "Anno", "(tutti gli anni)"),
                ]
                _filter_cols = st.columns(2)
                for _idx, (_field, _label, _default) in enumerate(_agent_filter_fields):
                    _values = _agent_available.get(_field, [])
                    if _values:
                        _chosen = _filter_cols[_idx % 2].selectbox(
                            _label, [_default] + _values, key=f"agentic_filter_{_field}",
                        )
                        _agent_filter_values[_field] = _chosen
            _agent_top_k = st.slider(
                "Passaggi candidati per ogni ricerca", 5, 100, config.AGENT_TOP_K,
                key="agentic_top_k", help="Quanti passaggi conservare da ciascun tentativo.",
            )
            _agent_top_n = st.slider(
                "Passaggi da usare nella risposta", 1, 20, min(config.RERANK_TOP_N, 20),
                key="agentic_top_n", help="Quante fonti inviare al modello che formula la risposta.",
            )
            _agent_retries = st.slider(
                "Ricerche aggiuntive se le fonti non bastano", 0, 2, 1,
                key="agentic_retries", help="Oltre alla ricerca iniziale, quante query mirate può provare l’agente.",
            )
            _agent_instructions = st.text_input(
                "Indicazioni facoltative per ordinare le fonti", key="agentic_rerank_instructions",
                placeholder="Es. privilegia tabelle con il rendiconto finanziario",
            ).strip()

        _feedback_context = ""
        _parent_run_id = None
        _feedback_id = None
        _pending = st.session_state.pop("agentic_pending_refinement", None)
        if _pending:
            if _pending.get("session_id") != st.session_state.agentic_session_id:
                st.warning("L’approfondimento è salvato nella sessione originale: selezionala per riprenderlo.")
            else:
                _parent = _agent_runs.find_one({"_id": _pending.get("run_id"),
                                                "session_id": st.session_state.agentic_session_id}) or {}
                _entry = next((item for item in _parent.get("feedback", [])
                               if item.get("id") == _pending.get("id") and item.get("action") == "refinement_submitted"), None)
                if not _entry:
                    st.warning("L’approfondimento salvato non è disponibile: prepara nuovamente la richiesta.")
                elif _agent_runs.find_one({"_id": _entry["id"]}, {"_id": 1}):
                    st.info("Questa richiesta di approfondimento è già stata avviata. Seleziona la ricerca salvata per rivederla.")
                else:
                    _saved_settings = _entry["settings"]
                    _agent_question = _agent_validate_refinement(
                        _parent.get("question", ""), _entry.get("question", ""),
                        top_k=_saved_settings["top_k"], top_n=_saved_settings["top_n"],
                    )
                    _agent_top_k, _agent_top_n = _saved_settings["top_k"], _saved_settings["top_n"]
                    _agent_retries = 0
                    _agent_instructions = _saved_settings.get("rerank_instructions", "")
                    _feedback_context = _entry.get("context", "")
                    _parent_run_id, _feedback_id = _parent["_id"], _entry["id"]
                    _parent_answer, _ = _agent_run_sources(_parent, _agent_memory)
                    _planning_memory, _planning_summaries = _agent_refinement_history(
                        _parent, _parent_answer, _agent_turn_summaries,
                    )
                    _agent_submit = True
                    st.info(f"Approfondimento richiesto: {_agent_question}")

        # Risposta alla domanda dell'agente (anno scelto o confronto tra anni).
        _clarification = st.session_state.pop("agentic_clarification", None)
        _clarification_filters = None
        _skip_ambiguity = False
        _is_clarification = False
        if _clarification and _clarification.get("session_id") == st.session_state.agentic_session_id:
            _agent_question = _clarification["question"]
            _feedback_context = _clarification["context"]
            _parent_run_id = _clarification["parent_run_id"]
            _clarification_filters = _clarification.get("filters")
            _skip_ambiguity = bool(_clarification.get("skip_ambiguity"))
            _is_clarification = True
            _agent_submit = True

        if _agent_submit and not _agent_question:
            st.warning("Scrivi una domanda prima di avviare la ricerca agentica.")

        if _agent_submit and _agent_question:
            _filters = (_saved_settings.get("filters") or None) if _feedback_id else _build_adaptive_filters(_agent_filter_values)
            if _clarification_filters:
                _filters = {**(_filters or {}), **_clarification_filters}
            _run_settings = {"interaction_mode": _agent_interaction, "top_k": _agent_top_k, "top_n": _agent_top_n,
                             "retries": _agent_retries, "filters": _filters or {}, "rerank_instructions": _agent_instructions,
                             "skip_ambiguity": _skip_ambiguity}
            _run_id = _agent_start_run(
                _agent_runs, _agent_sessions, st.session_state.agentic_session_id, _agent_question,
                parent_run_id=_parent_run_id, feedback_id=_feedback_id, settings=_run_settings,
                feedback_context=_feedback_context,
            )
            st.session_state.agentic_working_state = {"run_id": _run_id, "step": "planning", "iteration": 0}
            _trace: list[dict[str, Any]] = []
            _live_journey = st.empty()
            with _live_journey.container():
                _render_agent_journey(_trace, "running", "planning", _agent_retries + 1)

            def _record_agent_event(event: dict[str, Any], *, step: str, iteration: int = 0, candidate_count: int = 0) -> None:
                _event = {**event, "model": _agent_model_label()}
                _trace.append(_event)
                st.session_state.agentic_working_state = {
                    "run_id": _run_id, "step": step, "iteration": iteration,
                    "candidate_count": candidate_count,
                }
                _agent_add_event(
                    _agent_runs, _agent_sessions, st.session_state.agentic_session_id, _run_id,
                    _event,
                    state={"step": step, "iteration": iteration, "candidate_count": candidate_count},
                )
                with _live_journey.container():
                    _render_agent_journey(_trace, "running", step, _agent_retries + 1)

            _agent_status = None
            try:
                _scope = require_scope(st.session_state.embed_mode)
                _modes = _scope["modes"]
                with st.status("1/4 · Capisco la domanda e preparo la ricerca", expanded=True) as _planning_status:
                    _plan_started = time.perf_counter()
                    if (_planning_memory or _planning_summaries or _feedback_context) and not _is_clarification:
                        # Il pianificatore serve a risolvere riferimenti alla conversazione ("E nel 2017?").
                        _plan = _agent_plan_search(_agent_question, _planning_memory, _planning_summaries)
                        _plan["skipped"] = False
                    else:
                        # Prima domanda della sessione: nulla da risolvere, si risparmia una chiamata al modello.
                        _plan = {"search_query": _agent_question, "comment": "", "error": None, "skipped": True}
                    _plan_ms = (time.perf_counter() - _plan_started) * 1000
                    # Una sola query: l'interpretazione del pianificatore contiene già tutti i vincoli.
                    # La scelta dell'anno è già nella domanda e nel filtro: le istruzioni non vanno nella ricerca.
                    _initial_query = _agent_compose_search_query(
                        _plan["search_query"], "" if _is_clarification else _feedback_context)
                    _planning_status.write(f"Prima ricerca: {_initial_query}")
                    if _plan.get("comment"):
                        _planning_status.write(_plan["comment"])
                    _record_agent_event({
                        "type": "plan", "search_query": _initial_query,
                        "planned_query": _plan["search_query"], "planner_skipped": _plan["skipped"],
                        "duration_ms": round(_plan_ms),
                        "comment": _plan["comment"], "memory_messages_used": len(_planning_memory),
                        "memory_context": [{"role": m.get("role"), "text": str(m.get("text", ""))[:1200],
                                            "feedback_context": m.get("feedback_context", "")}
                                           for m in _planning_memory[-2:]],
                        "filters": _filters or {}, "modes": _modes,
                        "settings": _run_settings, "feedback_context": _feedback_context,
                        "parent_run_id": _parent_run_id,
                        "database_search": {
                            "database": _scope["database"], "collection": _scope["collection"],
                            "text_index": config.SEARCH_INDEX_NAME, "text_path": "content",
                            "vector_indexes": [{"mode": mode, "index": config.vector_index_name(mode),
                                                "path": {"auto": "content", "manual": "embedding",
                                                         "context": "embedding_context", "multimodal": "embedding_multimodal"}[mode]}
                                               for mode in _modes],
                        },
                        "max_searches": _agent_retries + 1,
                        "planner_error": _plan.get("error"),
                    }, step="retrieval", iteration=0)
                    if _plan.get("error"):
                        _planning_status.write(f"⚠️ Il pianificatore non ha risposto ({_plan['error']}): uso la domanda originale.")
                    _planning_status.update(label="1/4 · Domanda interpretata", state="complete", expanded=False)

                _ranked_rounds: list[list[dict[str, Any]]] = []
                _executed_queries: list[str] = []
                _candidate_union: dict[str, dict[str, Any]] = {}
                _search_query = _initial_query
                _seen_queries = {_agent_question.casefold(), _plan["search_query"].casefold(), _initial_query.casefold()}
                _stop_reason = "Ho usato tutte le ricerche configurate."
                _previous_grade: dict[str, Any] | None = None
                # Domanda completa: riordino dei passaggi raccolti e valutazione usano sempre questa.
                _final_query = _initial_query
                _pool: list[dict[str, Any]] = []
                _fused: dict[str, dict[str, Any]] = {}
                _not_answerable: str | None = None
                _threshold = config.RERANK_RELEVANCE_THRESHOLD
                _stop_reasons = {
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
                with st.status("2/4 · Cerco nei documenti e controllo le fonti", expanded=True) as _agent_status:
                    for _iteration in range(_agent_retries + 1):
                        _agent_status.write(f"Ricerca {_iteration + 1}: {_search_query}")
                        _state = {"step": "retrieval", "iteration": _iteration + 1, "candidate_count": 0}
                        st.session_state.agentic_working_state = {"run_id": _run_id, **_state}
                        _step_started = time.perf_counter()
                        _candidates, _ranked = _agent_retrieve_candidates(
                            _coll(), _search_query, filters=_filters, modes=_modes,
                            top_k=_agent_top_k, instructions=_agent_instructions,
                        )
                        _retrieval_ms = (time.perf_counter() - _step_started) * 1000
                        _ranked_rounds.append(_ranked)
                        _executed_queries.append(_search_query)
                        _top_source_labels = [
                            f"{doc.get('filename', 'Documento')} · {_page_label(doc.get('page_number'))}"
                            for doc in _ranked[:3]
                        ]
                        _agent_status.write(
                            f"Ho trovato {len(_candidates)} passaggi candidati. "
                            + ("Tra le fonti in cima: " + "; ".join(_top_source_labels) if _top_source_labels else "Non sono emersi passaggi utili.")
                        )
                        for _candidate in _candidates:
                            _key = str(_candidate.get("_id", passage_key(_candidate)))
                            _candidate_union.setdefault(_key, _candidate)
                        _refs = [{
                            "rank": _rank,
                            "filename": _doc.get("filename"),
                            "page": _doc.get("page_number"),
                            "title": _doc.get("title"),
                            "chunk_id": str(_doc.get("_id", "")),
                        } for _rank, _doc in enumerate(_ranked[:10], start=1)]
                        _record_agent_event({
                            "type": "retrieval", "iteration": _iteration + 1,
                            "search_query": _search_query, "candidate_count": len(_candidates),
                            "ranked_sources": _refs, "search_modes": _modes,
                            "duration_ms": round(_retrieval_ms), "focused": _iteration > 0,
                        }, step="grading", iteration=_iteration + 1, candidate_count=len(_candidates))

                        # Prima ricerca: stessa soglia fuori tema della pagina Query.
                        # Il punteggio è riferito alla domanda completa, quindi è confrontabile con la soglia.
                        if _iteration == 0:
                            _top_rerank = _ranked[0].get("_rerank_score") if _ranked else None
                            if not _ranked or (isinstance(_top_rerank, (int, float)) and _top_rerank < _threshold):
                                _not_answerable = "no_results" if not _ranked else "off_topic"
                                _stop_reason = _stop_reasons[_not_answerable]
                                if _not_answerable == "off_topic":
                                    _stop_reason += f" (miglior passaggio {_top_rerank:.3f} < soglia {_threshold:.2f})"
                                _record_agent_event({"type": "decision", "iteration": 1, "reason": _not_answerable,
                                                     "action": "stop", "top_rerank": _top_rerank, "threshold": _threshold},
                                                    step="decision", iteration=1, candidate_count=len(_candidates))
                                break

                        # Insieme cumulativo: tutti i passaggi trovati finora, riordinati
                        # rispetto alla domanda completa. Il valutatore giudica questo insieme,
                        # che è lo stesso da cui verranno scelte le fonti della risposta.
                        _fused = {}
                        for _round_idx, _round in enumerate(_ranked_rounds, start=1):
                            for _rank, _doc in enumerate(_round, start=1):
                                _key = str(_doc.get("_id", passage_key(_doc)))
                                _entry = _fused.setdefault(_key, {"doc": _doc, "score": 0.0, "ranks": []})
                                _entry["score"] += 1 / (60 + _rank)
                                _entry["ranks"].append({"iteration": _round_idx, "rank": _rank})
                        if len(_ranked_rounds) == 1:
                            _pool = list(_ranked)
                        else:
                            _ordered_union = sorted(_fused.values(), key=lambda item: item["score"], reverse=True)
                            # Al riordino vanno solo i primi passaggi per RRF: riordinarli tutti costa tempo
                            # e quelli in fondo a ogni ricerca non arrivano comunque tra le fonti.
                            _step_started = time.perf_counter()
                            _pool = _agent_rerank_union(
                                _coll(), _final_query,
                                [item["doc"] for item in _ordered_union[:config.AGENT_UNION_RERANK]],
                                top_n=max(_agent_top_n, 8), instructions=_agent_instructions,
                            )
                            _union_ms = (time.perf_counter() - _step_started) * 1000
                            _record_agent_event({
                                "type": "agent_rank_fusion", "method": "reciprocal_rank_fusion",
                                "constant": 60, "unique_candidates": len(_fused),
                                "leading_sources": [{
                                    "filename": item["doc"].get("filename"),
                                    "page": item["doc"].get("page_number"),
                                    "rrf_score": item["score"], "iteration_ranks": item["ranks"],
                                } for item in _ordered_union[:_agent_top_n]],
                                "comment": ("Ho riunito i passaggi di tutte le ricerche, rimosso i duplicati e li ho "
                                            "riordinati rispetto alla domanda completa prima della valutazione."),
                                "reranked": min(len(_ordered_union), config.AGENT_UNION_RERANK),
                                "duration_ms": round(_union_ms),
                            }, step="grading", iteration=_iteration + 1, candidate_count=len(_fused))
                            _agent_status.write(f"Riunisco {len(_fused)} passaggi delle {len(_ranked_rounds)} ricerche e li riordino rispetto alla domanda.")

                        _pool_refs = [{
                            "rank": _rank, "filename": _doc.get("filename"), "page": _doc.get("page_number"),
                            "title": _doc.get("title"), "chunk_id": str(_doc.get("_id", "")),
                        } for _rank, _doc in enumerate(_pool[:8], start=1)]
                        _step_started = time.perf_counter()
                        # Per una domanda di seguito ("E nel 2017?") il valutatore deve sapere a cosa si riferisce.
                        _grade_question = (_agent_question if _plan.get("skipped") or _is_clarification
                                           else f"{_agent_question} (riferita a: {_plan['search_query']})")
                        _grade = _agent_assess_evidence(_grade_question, _search_query, _pool, _executed_queries)
                        _grade_ms = (time.perf_counter() - _step_started) * 1000
                        _record_agent_event({
                            "type": "grading", "iteration": _iteration + 1,
                            "cumulative": len(_ranked_rounds) > 1, "duration_ms": round(_grade_ms),
                            "elements": _grade.get("elements", []),
                            "unverified_elements": _grade.get("unverified_elements", []),
                            "evaluated": _grade.get("evaluated", True),
                            "sufficient": _grade["sufficient"], "comment": _grade["comment"],
                            "followup_query": _grade.get("followup_query"),
                            "evaluated_sources": len(_pool[:8]),
                            "evaluated_source_refs": [{**ref, "excerpt": _agent_compact_text(doc.get("content", ""))[:3000]}
                                                      for ref, doc in zip(_pool_refs, _pool[:8])],
                            "missing_elements": _grade.get("missing_elements", []),
                            "supporting_refs": _grade.get("supporting_refs", []),
                            "supporting_quotes": _grade.get("supporting_quotes", []),
                            "failure_reason": _grade.get("failure_reason"),
                            "failure_detail": _grade.get("failure_detail"),
                        }, step="decision", iteration=_iteration + 1, candidate_count=len(_candidate_union))
                        _previous_grade = _grade
                        # Valori diversi in anni diversi e domanda senza anno: si ferma e chiede.
                        _ambiguity = None if _skip_ambiguity else _agent_find_year_ambiguity(
                            _grade, _agent_question, _plan["search_query"], _feedback_context)
                        if _ambiguity:
                            _not_answerable = "needs_clarification"
                            _stop_reason = "Valori diversi in anni diversi: serve sapere quale anno."
                            _record_agent_event({"type": "clarification", "iteration": _iteration + 1,
                                                 "reason": "year", **_ambiguity},
                                                step="decision", iteration=_iteration + 1, candidate_count=len(_candidate_union))
                            _record_agent_event({"type": "decision", "iteration": _iteration + 1,
                                                 "reason": "needs_clarification", "action": "stop"},
                                                step="decision", iteration=_iteration + 1, candidate_count=len(_candidate_union))
                            break
                        _decision = _agent_decide_next_search(
                            _grade, iteration=_iteration, retries=_agent_retries, seen_queries=_seen_queries,
                        )
                        _record_agent_event({"type": "decision", "iteration": _iteration + 1,
                                             "reason": _decision["reason"], "action": _decision["action"],
                                             "search_query": _decision.get("query")},
                                            step="rewrite" if _decision["action"] == "continue" else "decision",
                                            iteration=_iteration + 1, candidate_count=len(_candidate_union))
                        if _decision["action"] == "stop":
                            _stop_reason = _stop_reasons[_decision["reason"]]
                            break
                        # Ricerca mirata: solo l'elemento mancante. La domanda completa non
                        # viene ripetuta, altrimenti la ricerca restituisce gli stessi passaggi;
                        # la pertinenza rispetto alla domanda la garantisce il riordino cumulativo.
                        _next_search = _decision["query"]
                        _seen_queries.add(_next_search.casefold())
                        _record_agent_event({
                            "type": "rewrite", "iteration": _iteration + 1,
                            "search_query": _next_search, "focus_query": _next_search,
                            "reason": _grade["comment"],
                        }, step="rewrite", iteration=_iteration + 1, candidate_count=len(_candidate_union))
                        _agent_status.write(f"Provo una ricerca mirata all’elemento mancante: {_next_search}")
                        _search_query = _next_search

                    _final_sources = _pool[:_agent_top_n]
                    if _final_sources:
                        _record_agent_event({
                            "type": "final_rerank", "search_query": _final_query,
                            "comment": ("Fonti scelte dall’insieme di tutte le ricerche, riordinato rispetto alla domanda completa: "
                                        "è lo stesso insieme esaminato dal valutatore."
                                        if len(_ranked_rounds) > 1 else
                                        "Una sola ricerca: le fonti mantengono l’ordine del reranker rispetto alla domanda completa."),
                            "selected_sources": [{"filename": doc.get("filename"), "page": doc.get("page_number")}
                                                 for doc in _final_sources],
                        }, step="generation", iteration=len(_ranked_rounds), candidate_count=len(_fused))
                        _agent_status.write(
                            f"Ho raccolto {len(_fused)} passaggi senza duplicati e selezionato "
                            f"{len(_final_sources)} fonti da usare per la risposta."
                        )
                    _agent_status.update(label="3/4 · Ho selezionato le fonti", state="running", expanded=True)

                    if _not_answerable == "needs_clarification":
                        _years = ", ".join(str(o["year"]) for o in _ambiguity["options"])
                        _answer = (f"Ho trovato valori diversi in anni diversi ({_years}) per: "
                                   + "; ".join(_ambiguity["elements"]) + ". Quale anno ti interessa?")
                        _record_agent_event({
                            "type": "answer", "status": "needs_input", "answer_preview": _answer, "sources": [],
                            "attempts": 0, "stop_reason": _stop_reason,
                        }, step="complete", iteration=len(_ranked_rounds), candidate_count=len(_fused))
                        _agent_finish_run(
                            _agent_runs, _agent_sessions,
                            session_id=st.session_state.agentic_session_id,
                            run_id=_run_id, question=_agent_question, answer=_answer,
                            sources=[], feedback_context=_feedback_context, parent_run_id=_parent_run_id,
                            status="needs_input",
                        )
                        _agent_status.update(label="L’agente ha una domanda per te", state="complete", expanded=False)
                    elif _not_answerable or not _final_sources:
                        # Nessuna chiamata al modello della risposta: come nella pagina Query.
                        _block_reason = _not_answerable or "no_results"
                        _answer = "Risposta non generata. " + _stop_reason
                        _record_agent_event({
                            "type": "answer", "status": "blocked", "answer_preview": _answer[:700], "sources": [],
                            "attempts": 0, "stop_reason": _stop_reason,
                            "diagnostics": {"block_reason": _block_reason, "attempts": [], "retry_error_type": None},
                        }, step="complete", iteration=len(_ranked_rounds), candidate_count=len(_fused))
                        _agent_finish_run(
                            _agent_runs, _agent_sessions,
                            session_id=st.session_state.agentic_session_id,
                            run_id=_run_id, question=_agent_question, answer=_answer,
                            sources=[], feedback_context=_feedback_context, parent_run_id=_parent_run_id,
                            status="blocked",
                        )
                        _agent_status.update(label="Domanda fuori tema: nessuna risposta generata", state="complete", expanded=False)
                    else:
                        if not llm_available():
                            raise RuntimeError("Il provider LLM configurato non è disponibile.")
                        _generation_question = _agent_question
                        if _plan["search_query"].casefold() != _agent_question.casefold():
                            _generation_question = (
                                f"Domanda dell'utente: {_agent_question}\n\n"
                                f"Interpretazione autonoma della domanda, ricavata anche dalla memoria recente: "
                                f"{_plan['search_query']}\n\n"
                                "Usa questa interpretazione solo per risolvere i riferimenti conversazionali. "
                                "Basa i fatti della risposta esclusivamente sui passaggi recuperati."
                            )
                        if _feedback_context:
                            _generation_question += (
                                "\n\nIndicazioni dell'utente per l'approfondimento (non sono evidenze): "
                                f"{_feedback_context}\nUsa queste indicazioni per capire la richiesta; "
                                "verifica i fatti esclusivamente nei passaggi recuperati."
                            )
                        _step_started = time.perf_counter()
                        with st.spinner("4/4 · Preparo la risposta usando le fonti selezionate…"):
                            _reviewed = generate_reviewed_answer(_generation_question, _final_sources)
                        _answer_ms = (time.perf_counter() - _step_started) * 1000
                        _answer = _reviewed["answer"].get("answer", "")
                        _diagnostics = answer_diagnostics(_reviewed)
                        _blocked = bool(_reviewed.get("blocked"))
                        _answer_check = (((_reviewed.get("attempts") or [{}])[-1].get("check") or {}).get("status", "cited")
                                         if not _blocked else _diagnostics.get("block_reason"))
                        if _blocked:
                            # Come nella Query: si blocca solo ciò che non va mai mostrato (indirizzi web,
                            # risposta troncata). Il resto si mostra con il verdetto «Da verificare».
                            _fallback = best_publishable_attempt(_reviewed)
                            if _fallback:
                                _answer, _answer_check = _fallback
                                _blocked = False
                            else:
                                _last_check = (_diagnostics["attempts"][-1].get("check") or {}) if _diagnostics["attempts"] else {}
                                _answer = "Risposta bloccata. " + explain_answer_check(_diagnostics.get("block_reason"), _last_check) + " Consulta i passaggi recuperati."
                        _source_refs = [{
                            "filename": doc.get("filename"), "page": doc.get("page_number"),
                            "chunk_id": str(doc.get("_id", "")),
                            "content": (doc.get("content") or "")[:8000],
                            "gridfs_file_id": str(doc.get("gridfs_file_id")) if doc.get("gridfs_file_id") else None,
                        } for doc in _final_sources]
                        _record_agent_event({
                            "type": "answer", "status": "blocked" if _blocked else "generated", "answer_check": _answer_check,
                            "answer_preview": _answer[:700], "sources": _source_refs,
                            "attempts": len(_reviewed.get("attempts", [])), "stop_reason": _stop_reason,
                            "diagnostics": _diagnostics, "duration_ms": round(_answer_ms),
                            # Verifica fatta sui passaggi completi, non sull'estratto salvato.
                            "citation_check": [] if _blocked else check_citations(_answer, _final_sources),
                        }, step="complete", iteration=len(_ranked_rounds), candidate_count=len(_fused))
                        _agent_finish_run(
                            _agent_runs, _agent_sessions,
                            session_id=st.session_state.agentic_session_id,
                            run_id=_run_id, question=_agent_question, answer=_answer,
                            sources=_source_refs, feedback_context=_feedback_context, parent_run_id=_parent_run_id,
                            status="blocked" if _blocked else "complete",
                        )
                        _agent_status.update(label="4/4 · Risposta pronta con le fonti", state="complete", expanded=False)

                _live_journey.empty()
                st.session_state.pop("agentic_run_picker", None)
                st.rerun()
            except Exception as _agent_error:
                if _agent_status is not None:
                    _agent_status.update(label="Ricerca agentica interrotta", state="error", expanded=True)
                _agent_fail_run(
                    _agent_runs, _agent_sessions,
                    session_id=st.session_state.agentic_session_id,
                    run_id=_run_id,
                    message=f"{type(_agent_error).__name__}: {_agent_error}",
                )
                _live_journey.empty()
                st.session_state.pop("agentic_run_picker", None)
                st.error("L’esecuzione agentica si è interrotta. Lo stato e l’errore sono stati salvati in MongoDB.")
                with st.expander("Dettaglio tecnico"):
                    st.code(f"{type(_agent_error).__name__}: {_agent_error}")

        _saved_runs = list(_agent_runs.find({"session_id": st.session_state.agentic_session_id})
                           .sort("created_at", -1).limit(20))
        if _saved_runs:
            _runs_by_id = {run["_id"]: run for run in _saved_runs}
            if st.session_state.get("agentic_run_picker") not in _runs_by_id:
                st.session_state.pop("agentic_run_picker", None)
            # La ricerca mostrata è l'ultima, oppure quella scelta in fondo alla pagina.
            _selected_run = _runs_by_id.get(st.session_state.get("agentic_run_picker"), _saved_runs[0])
            _render_agent_result(_selected_run, _agent_memory)
            if _selected_run.get("status") in {"complete", "blocked", "needs_input"}:
                _value_card = agent_value_html(_selected_run, dark=_theme_is_dark())
                if _value_card:
                    components.html(_value_card[0], height=_value_card[1])
            if _selected_run.get("parent_run_id"):
                _previous_run = _agent_runs.find_one({"_id": _selected_run["parent_run_id"],
                                                      "session_id": st.session_state.agentic_session_id})
                if _previous_run:
                    _previous_answer, _previous_sources = _agent_run_sources(_previous_run, _agent_memory)
                    with st.expander("Confronta con la risposta precedente", expanded=False):
                        st.write("**Domanda precedente:** " + (_previous_run.get("question") or "non disponibile"))
                        st.write("**Risposta precedente:** " + (_previous_answer or "non disponibile"))
                        st.write("**Nuova domanda:** " + (_selected_run.get("question") or "non disponibile"))
                        if _selected_run.get("feedback_context"):
                            st.write("**Indicazioni dell’utente:** " + _selected_run["feedback_context"])
            if _selected_run.get("status") in {"complete", "blocked"}:
                _render_agent_feedback_controls(
                    _selected_run, _agent_sessions, _agent_runs, st.session_state.agentic_session_id,
                    _agent_available, {"top_k": _agent_top_k, "top_n": _agent_top_n,
                                       "rerank_instructions": _agent_instructions},
                )
            _run_status = _selected_run.get("status", "")
            _journey_status = "running_saved" if _run_status == "running" else _run_status
            with st.expander("🧭 Come ci è arrivato · " + _agent_journey_summary(_selected_run.get("events", [])),
                             expanded=False):
                _render_agent_journey(_selected_run.get("events", []), _journey_status, run=_selected_run)

            st.divider()
            with st.expander(f"Ricerche precedenti di questa sessione ({len(_saved_runs)})", expanded=False):
                st.selectbox(
                    "Mostra la ricerca", list(_runs_by_id), key="agentic_run_picker",
                    format_func=lambda run_id: (
                        f"{(_runs_by_id[run_id].get('created_at').strftime('%d/%m %H:%M') if _runs_by_id[run_id].get('created_at') else '')} · "
                        f"{(_runs_by_id[run_id].get('question') or 'Domanda')[:100]}"
                    ),
                )
        else:
            st.info("Scrivi una domanda e avvia la ricerca: qui vedrai la risposta, le fonti e come l’agente ci è arrivato.")
        with st.expander("💾 Sessione e memoria in MongoDB", expanded=False):
            st.write(
                "La conversazione è conservata in `agentic_sessions`; "
                "domanda, stato e passaggi di ogni esecuzione sono conservati in `agentic_runs`. "
                "La memoria è limitata alla sessione selezionata."
            )
            st.caption(
                f"Messaggi conservati: {len(_agent_memory)} · "
                f"Riepiloghi: {len(_agent_turn_summaries)} · "
                f"Feedback salvati: {len(_agent_feedback)} · "
                f"Esecuzioni completate: {(_agent_session.get('memory') or {}).get('turn_count', 0)}"
            )
            st.caption("La demo non ha autenticazione per singolo utente: le sessioni nel database sono visibili nel selettore.")
            if _recent_memory:
                st.markdown("**Conversazione recente**")
                for _message in _recent_memory:
                    with st.chat_message("user" if _message.get("role") == "user" else "assistant"):
                        st.write(_message.get("text", ""))
            if _agent_feedback:
                st.markdown("**Feedback della sessione**")
                for _feedback in _agent_feedback[-10:]:
                    _feedback_label = {"accepted": "Risultato accettato", "needs_refinement": "Richiesto approfondimento",
                                       "refinement_submitted": "Nuova ricerca preparata"}.get(_feedback.get("action"), "Feedback")
                    st.write(f"{_feedback_label} · ricerca {_feedback.get('run_id', '')[:8]}")
                    if _feedback.get("question"):
                        st.write(f"Nuova domanda: {_feedback['question']}")
                    if _feedback.get("context"):
                        st.write(f"Indicazioni aggiuntive: {_feedback['context']}")
                    if _feedback.get("settings"):
                        st.caption("Parametri salvati: " + _filter_summary(_feedback["settings"]))

        with st.expander("🎬 Due percorsi per la demo", expanded=False):
            st.markdown(
                "**1 · Feedback e approfondimento.** Scegli la modalità «Con feedback», fai una domanda e, "
                "se manca qualcosa, scegli **No, approfondisci** e scrivi una nuova domanda con dettagli e filtri.\n\n"
                "**2 · Memoria della conversazione.** In una nuova sessione chiedi le spese 3M del 2018; "
                "poi chiedi «E nel 2017?». Nel percorso vedi come l’agente ha interpretato la domanda."
            )
    except Exception as _agent_storage_error:
        st.error("Memoria agentica MongoDB non disponibile. Verifica la connessione e i permessi sul database.")
        with st.expander("Dettaglio tecnico"):
            st.code(f"{type(_agent_storage_error).__name__}: {_agent_storage_error}")


# ─── ADMIN ─────────────────────────────────────────────────────────────────────
if _page == "admin":
    st.title("Setup ambiente MongoDB")
    st.markdown(
        "Questa procedura prepara il database per l’applicazione e rende visibili "
        "i risultati di ogni passaggio. Crea solo elementi mancanti; non cancella "
        "documenti o indici esistenti, salvo quando si sceglie esplicitamente "
        "**Drop & recreate** più sotto."
    )
    st.caption(
        f"Database `{config.MONGODB_DB}` · collection chunk `{config.MONGODB_COLLECTION}` · "
        f"bucket GridFS `{config.MONGODB_GRIDFS_BUCKET}` · modalità attiva "
        f"`{st.session_state.embed_mode}`"
    )

    run_full_setup = st.button(
        "Prepara database e indici",
        type="primary",
        help="Verifica MongoDB, prepara collection, indice documenti e indici GridFS, quindi crea o aggiorna gli indici Atlas per tutte le modalità.",
    )
    st.info(
        "Il setup prepara le collection, l’indice document_id/embedding_mode, gli indici GridFS e i cinque indici Atlas: quattro vettoriali e uno testuale. "
        "Confronta gli schemi: crea quelli mancanti, aggiorna quelli diversi e mantiene quelli già allineati. "
        "Corregge anche gli eventuali riferimenti di pagina sintetici degli estratti Office. "
        "Le creazioni e gli aggiornamenti Atlas sono asincroni: il rendiconto distingue la richiesta inviata dall’indice pronto."
    )

    with st.expander("Verifica definizioni e mostra il codice MongoDB di creazione / aggiornamento"):
        st.caption("Confronto tra le definizioni presenti su Atlas e quelle richieste dalla demo. Il setup rilegge lo stato prima di applicare le modifiche; l'indice testuale condiviso viene gestito una sola volta.")
        try:
            planned_indexes = index_plan(_coll())
            action_labels = {"create": "Da creare", "update": "Da aggiornare", "unchanged": "Schema allineato", "blocked": "Intervento richiesto"}
            st.dataframe([{
                "Indice": item["spec"]["name"], "Tipo": item["spec"]["type"],
                "Confronto schema": action_labels[item["action"]], "Stato Atlas": item["status"],
                "Interrogabile": item["queryable"],
                "Differenze": "; ".join(item["differences"]) or "Nessuna",
            } for item in planned_indexes], hide_index=True, use_container_width=True)
            for item in planned_indexes:
                spec = item["spec"]
                st.markdown(f"**{spec['name']}** · {action_labels[item['action']]}")
                for difference in item["differences"]:
                    st.text(difference)
                if item["action"] in ("create", "update"):
                    st.code(index_command(config.MONGODB_DB, config.MONGODB_COLLECTION, spec, item["action"]), language="javascript")
                elif item["action"] == "unchanged":
                    st.caption("Nessuna scrittura necessaria. Codice di creazione mostrato come riferimento:")
                    st.code(index_command(config.MONGODB_DB, config.MONGODB_COLLECTION, spec, "create"), language="javascript")
                else:
                    st.warning("Il setup non applicherà una sostituzione automatica a questo indice.")
        except Exception as exc:
            st.warning(f"Impossibile confrontare gli indici: {exc}")

    if run_full_setup:
        st.session_state.pop("database_index_report", None)
        setup_status = st.status("Preparazione database e indici in corso…", expanded=True)
        setup_status.write(f"Verifico connessione al database {config.MONGODB_DB}.")
        setup_steps: list[dict] = []
        try:
            _db().command("ping")
            setup_steps.append({"Passaggio": "Connessione MongoDB Atlas", "Stato": "Completato", "Dettaglio": f"Database `{config.MONGODB_DB}` raggiungibile"})
        except Exception as exc:
            setup_steps.append({"Passaggio": "Connessione MongoDB Atlas", "Stato": "Errore", "Dettaglio": f"{type(exc).__name__}: {str(exc).splitlines()[0][:240]}"})

        if setup_steps[-1]["Stato"] == "Completato":
            try:
                setup_status.write("Preparo le collection, l’indice dei documenti e gli indici GridFS, poi leggo i conteggi.")
                db_setup = initialize_database()
                created_label = ", ".join(db_setup["created"]) or "nessuna (già presenti)"
                setup_steps.append({"Passaggio": "Collection applicative e GridFS", "Stato": "Completato", "Dettaglio": f"Create: {created_label}; già presenti: {', '.join(db_setup['already_present']) or 'nessuna'}"})
                setup_steps.append({"Passaggio": "Indice lookup documenti", "Stato": "Completato", "Dettaglio": f"{config.MONGODB_COLLECTION}: {db_setup['document_lookup_index']['name']} ({'creato' if db_setup['document_lookup_index']['created'] else 'già presente'})"})
                setup_steps.append({"Passaggio": "Indici locali GridFS", "Stato": "Completato", "Dettaglio": f"{config.MONGODB_GRIDFS_BUCKET}.files: {', '.join(db_setup['gridfs_indexes']['files'])}; {config.MONGODB_GRIDFS_BUCKET}.chunks: {', '.join(db_setup['gridfs_indexes']['chunks'])}"})
                setup_steps.append({"Passaggio": "Conteggi iniziali", "Stato": "Completato", "Dettaglio": "; ".join(f"{name}: {count}" for name, count in db_setup["document_counts_estimated"].items())})
            except Exception as exc:
                setup_steps.append({"Passaggio": "Collection / indici documenti e GridFS", "Stato": "Errore", "Dettaglio": f"{type(exc).__name__}: {str(exc).splitlines()[0][:240]}"})

            if setup_steps[-1]["Stato"] == "Completato":
                try:
                    setup_status.write("Correggo eventuali numeri di pagina sintetici negli estratti Office a livello documento.")
                    repaired = repair_document_page_references(_coll())
                    setup_steps.append({"Passaggio": "Riferimenti alle pagine Office", "Stato": "Completato", "Dettaglio": f"Estratti documento corretti: {repaired['office_structure_corrected']}; fallback MarkItDown corretti: {repaired['markitdown_fallback_corrected']}. Testo e vettori conservati."})
                except Exception as exc:
                    setup_steps.append({"Passaggio": "Riferimenti alle pagine Office", "Stato": "Errore", "Dettaglio": str(exc)})
                try:
                    result = ensure_indexes(_coll(), modes=("auto", "manual", "context", "multimodal"), on_progress=setup_status.write)
                    st.session_state.database_index_report = result
                    action_labels = {"create": "Creazione richiesta", "update": "Aggiornamento richiesto", "unchanged": "Schema allineato", "blocked": "Bloccato"}
                    for operation in result["operations"]:
                        name = operation["name"]
                        state = result["atlas_status"][name]
                        setup_steps.append({
                            "Passaggio": f"Indice Atlas · {name}",
                            "Stato": "Errore" if operation["error"] else action_labels[operation["action"]],
                            "Dettaglio": operation["error"] or f"Atlas: {state['status']}; interrogabile: {state['queryable']}. " + ("; ".join(operation["differences"]) or "Definizione già conforme."),
                        })
                except Exception as exc:
                    setup_steps.append({"Passaggio": "Indici Atlas", "Stato": "Errore", "Dettaglio": str(exc)})

        st.session_state.database_setup_report = setup_steps
        setup_has_errors = any(step["Stato"] == "Errore" for step in setup_steps)
        setup_status.update(label="Setup terminato con errori: consulta il rendiconto" if setup_has_errors else "Setup completato; lo stato di costruzione degli indici è indicato nel rendiconto", state="error" if setup_has_errors else "complete", expanded=setup_has_errors)
        _health_check.clear()
        preview_scope.clear()
        _document_list.clear()

    if report := st.session_state.get("database_setup_report"):
        st.markdown("#### Rendiconto operazione")
        st.dataframe(report, hide_index=True, use_container_width=True)
        if index_report := st.session_state.get("database_index_report"):
            st.caption(f"Indici creati: {len(index_report['created'])} · aggiornati: {len(index_report['updated'])} · già allineati: {len(index_report['already_present'])} · errori: {sum(bool(op['error']) for op in index_report['operations'])}")
            with st.expander("Operazioni richieste: definizioni precedenti, nuove e comandi MongoDB"):
                st.json(index_report)
        if st.button("Aggiorna stato attuale da Atlas", key="refresh_setup_report"):
            _health_check.clear()
        with st.expander("Dettaglio collection e indici presenti su Atlas"):
            try:
                all_indexes = list(_coll().list_search_indexes())
                if all_indexes:
                    mode_models = {
                        "auto": config.VOYAGE_EMBED_MODEL,
                        "manual": config.VOYAGE_EMBED_MODEL,
                        "context": config.VOYAGE_CONTEXT_MODEL,
                        "multimodal": config.VOYAGE_MULTIMODAL_MODEL,
                    }
                    mode_fields = {
                        "auto": "content (autoEmbed)",
                        "manual": f"embedding ({config.VOYAGE_EMBED_DIMENSIONS} dimensioni)",
                        "context": f"embedding_context ({config.VOYAGE_CONTEXT_DIMENSIONS} dimensioni)",
                        "multimodal": f"embedding_multimodal ({config.VOYAGE_MULTIMODAL_DIMENSIONS} dimensioni)",
                    }
                    rows = []
                    for idx in all_indexes:
                        name = idx.get("name", "?")
                        mode_for_index = next((m for m in mode_models if name == config.vector_index_name(m)), None)
                        if name == config.SEARCH_INDEX_NAME:
                            role, embedding, model = "Ricerca testuale condivisa", "content", "—"
                        elif mode_for_index:
                            role = f"Ricerca vettoriale · {mode_for_index}"
                            embedding = mode_fields[mode_for_index]
                            model = mode_models[mode_for_index]
                        else:
                            role, embedding, model = "Altro indice Atlas", "—", "—"
                        rows.append({
                            "Collection": config.MONGODB_COLLECTION,
                            "Uso / modalità": role,
                            "Nome indice Atlas": name,
                            "Stato Atlas": idx.get("status", "?"),
                            "Campo indicizzato": embedding,
                            "Modello embedding": model,
                            "Definizione Atlas": json.dumps(idx.get("latestDefinition", idx.get("definition", {})), ensure_ascii=False),
                        })
                    st.dataframe(rows, hide_index=True, use_container_width=True)
                else:
                    st.caption("Atlas non restituisce ancora indici per questa collection.")
            except Exception as exc:
                st.warning(f"Impossibile leggere ora gli indici Atlas: {type(exc).__name__}: {str(exc).splitlines()[0][:200]}")
        st.caption("‘Richiesto ad Atlas’ significa che la creazione o l'aggiornamento è stato accettato. Durante un aggiornamento l'indice può restare interrogabile con la versione precedente: attendi READY per usare la nuova definizione. Il rendiconto dell'operazione resta storico; la tabella dello stato attuale viene riletta da Atlas.")

    st.subheader("Modalità di indicizzazione: guida alla demo")
    st.markdown(
        "Questa demo mette a confronto **quattro modi di trasformare i documenti in contenuti ricercabili**. "
        "Le modalità `auto`, `manual` e `context` lavorano sul testo; `multimodal` aggiunge anche ciò che si vede "
        "nelle pagine e nelle slide. Non sono quattro pulsanti equivalenti: servono a mostrare dove viene creato "
        "l’embedding e quanto contesto o contenuto visivo entra nella ricerca."
    )

    mode_col1, mode_col2 = st.columns(2)
    with mode_col1:
        with st.container(border=True):
            st.markdown("#### 1. Atlas genera gli embedding — `auto`")
            st.write(
                f"L’app estrae testo e lo divide in chunk. Atlas genera gli embedding con "
                f"`{config.VOYAGE_EMBED_MODEL}` quando indicizza i chunk; l’app non salva un vettore "
                "nel singolo documento. È il percorso gestito da Atlas, con meno logica di embedding nell’app."
            )
            st.caption("Cosa dimostra: la ricerca semantica testuale con embedding gestiti direttamente dall’indice Atlas.")
        with st.container(border=True):
            st.markdown("#### 2. L’app genera gli embedding — `manual`")
            st.write(
                f"L’app invia ogni chunk testuale a Voyage (`{config.VOYAGE_EMBED_MODEL}`), riceve il vettore "
                "e lo salva nel campo `embedding`. Atlas cerca poi in quel campo con un indice vector standard."
            )
            st.caption("Cosa dimostra: il flusso esplicito lato applicazione, con vettori e modello controllabili e ispezionabili.")
    with mode_col2:
        with st.container(border=True):
            st.markdown("#### 3. Voyage considera il contesto — `context`")
            st.write(
                f"Voyage (`{config.VOYAGE_CONTEXT_MODEL}`) calcola gli embedding dei chunk tenendo conto anche "
                "degli altri passaggi dello stesso documento. Il vettore è salvato in `embedding_context`; "
                "pagina e riferimenti dei chunk restano disponibili per le citazioni."
            )
            st.caption("Cosa dimostra: se il contesto del documento aiuta su riferimenti, acronimi e passaggi che da soli sono ambigui.")
        with st.container(border=True):
            st.markdown("#### 4. Voyage legge testo e immagini — `multimodal`")
            st.write(
                f"L’app prepara ogni pagina o slide come testo più immagine e Voyage (`{config.VOYAGE_MULTIMODAL_MODEL}`) "
                "li trasforma in vettori multimodali, salvati in `embedding_multimodal`. Così la ricerca può recuperare "
                "anche grafici, diagrammi, screenshot e immagini, mantenendo il riferimento alla pagina o slide."
            )
            st.caption(
                "PDF: le pagine vengono renderizzate direttamente. DOCX/PPTX: MarkItDown estrae la struttura e "
                "LibreOffice (`soffice`) converte il file in PDF per renderizzare fedelmente pagine/slide; poi Voyage "
                "riceve testo e immagini. Voyage non riceve il file Office grezzo. LibreOffice serve solo per il percorso "
                "visivo dei formati Office, non per i PDF."
            )

    with st.container(border=True):
        st.markdown("**Come fare un confronto corretto**")
        st.write(
            "Usa gli stessi documenti e le stesse domande. Per ogni strategia crea/verifica il relativo indice "
            "in Admin e reingestisci il documento con quella strategia: le modalità usano vettori e indici distinti. "
            "Confronta se recuperano la pagina giusta, soprattutto per domande su riferimenti tra sezioni e su "
            "informazioni presenti solo in figure o slide. Il selettore qui sotto cambia la modalità attiva, la stessa "
            "che scegli in Upload: Query e Ricerca agentica cercano solo in quella."
        )
    _mode_options = ["auto", "manual", "context", "multimodal"]
    new_mode = st.radio(
        "Modalità attiva",
        options=_mode_options,
        index=_mode_options.index(st.session_state.embed_mode)
              if st.session_state.embed_mode in _mode_options else 0,
        horizontal=True, key="embed_mode_radio",
        format_func=lambda x: {
            "auto":    "Atlas genera embedding · auto",
            "manual":  "App genera embedding · manual",
            "context": "Voyage usa il contesto · context",
            "multimodal": "Voyage usa testo e immagini · multimodal",
        }[x],
    )
    if new_mode != st.session_state.embed_mode:
        st.session_state.embed_mode = new_mode
        try:
            save_active_mode(new_mode)
        except Exception as exc:
            st.warning(f"Modalità attiva non salvata nel database ({type(exc).__name__}): vale solo per questa sessione.")
        preview_scope.clear()
        _health_check.clear()  # forza rinfresco del semaforo dopo il cambio
        # Memorizzo il messaggio di successo per mostrarlo dopo il rerun
        # (senza rerun la pagina mostrerebbe ancora la modalità vecchia in
        # cima — title caption, semaforo — perché il render dell'header
        # gira prima di questo blocco in Admin).
        st.session_state.mode_changed_msg = (
            f"Modalità attiva impostata a `{new_mode}`: Upload carica in questa modalità e Query e "
            "Ricerca agentica cercano solo qui. Ogni modalità usa il proprio indice: se non ci sono "
            "ancora documenti in questa modalità, caricali da Upload."
        )
        st.rerun()

    # Mostra eventuale messaggio sopravvissuto al rerun (cambio modalità)
    if msg := st.session_state.pop("mode_changed_msg", None):
        st.success(msg)

    st.divider()
    st.subheader("Indici Atlas Search")
    st.caption(
        "**Drop & recreate** elimina l'indice esistente e lo ricrea con la "
        "definizione corrente (necessario dopo cambio modello o modalità). "
        "La verifica include anche l’indice ordinario dei documenti, senza eliminarlo."
    )
    recreate = st.checkbox("Drop & recreate (sovrascrive indici esistenti)", value=False)
    if st.button("Verifica / Crea indici", type="primary"):
        try:
            document_index = ensure_document_lookup_index(_coll())
            info = ensure_indexes(_coll(), mode=st.session_state.embed_mode,
                                  recreate=recreate)
            info["document_lookup_index"] = document_index
            if any(operation["error"] for operation in info["operations"]):
                st.warning("Operazione completata con errori: consulta il rendiconto per indice.")
            else:
                st.success("Verifica completata; eventuali creazioni o aggiornamenti sono stati richiesti ad Atlas.")
            st.json(info)
            preview_scope.clear()
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
