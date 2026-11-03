"""Perimetro di ricerca condiviso: inventario, indici e spiegazione nella demo."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

from . import config
from .indexes import _canonical, index_specs
from .retrieval import EMBED_MODES, embed_query, fusion_weights


MODE_LABELS = {
    "all": "Tutti i documenti · testo e contenuti visivi",
    "auto": "Solo testo con embedding gestiti da Atlas (auto)",
    "manual": "Solo testo con embedding generati dall'app (manual)",
    "context": "Solo testo con embedding contestualizzati (context)",
    "multimodal": "Solo documenti indicizzati con testo e immagini (multimodal)",
}


def inspect_scope(scope: str) -> dict:
    """Legge soltanto conteggi e definizioni degli indici, mai i contenuti."""
    if scope not in MODE_LABELS:
        raise ValueError(f"Perimetro sconosciuto: {scope}")
    coll = config.get_collection()
    counts = {row["_id"]: row["count"] for row in coll.aggregate([
        {"$group": {"_id": "$embedding_mode", "count": {"$sum": 1}}},
    ])}
    indexes = {idx["name"]: idx for idx in coll.list_search_indexes()}
    modes = [mode for mode in EMBED_MODES if counts.get(mode, 0)] if scope == "all" else [scope]
    models = {"auto": config.VOYAGE_EMBED_MODEL, "manual": config.VOYAGE_EMBED_MODEL,
              "context": config.VOYAGE_CONTEXT_MODEL, "multimodal": config.VOYAGE_MULTIMODAL_MODEL}
    rows = []
    for mode in EMBED_MODES:
        name = config.vector_index_name(mode)
        idx = indexes.get(name, {})
        # Per autoEmbed il modello effettivo è quello nell'indice Atlas.
        if mode == "auto":
            for field in idx.get("latestDefinition", {}).get("fields", []):
                if field.get("type") == "autoEmbed":
                    models[mode] = field.get("model", models[mode])
        rows.append({"Modalità": mode, "Chunk presenti": counts.get(mode, 0),
                     "Inclusa": "Sì" if mode in modes else "No",
                     "Indice vettoriale": name, "Modello": models[mode],
                     "Stato indice": idx.get("status", "MANCANTE")})
    required = [config.SEARCH_INDEX_NAME] + [config.vector_index_name(mode) for mode in modes]
    unavailable = [name for name in required if not indexes.get(name, {}).get("queryable", False)]
    specs = {spec["name"]: spec for mode in (modes or ["auto"]) for spec in index_specs(mode)}
    mismatched = []
    for name in required:
        actual = indexes.get(name)
        if not actual:
            continue
        expected = specs[name]
        definition = actual.get("latestDefinition", actual.get("definition"))
        if (actual.get("type", "search") != expected["type"] or definition is None
                or _canonical(definition) != _canonical(expected["definition"])):
            mismatched.append(name)
    return {"database": coll.database.name, "collection": coll.name, "modes": modes,
            "rows": rows, "text_index": config.SEARCH_INDEX_NAME,
            "text_index_status": indexes.get(config.SEARCH_INDEX_NAME, {}).get("status", "MANCANTE"),
            "unavailable": unavailable, "mismatched": mismatched,
            "unclassified": sum(n for mode, n in counts.items() if mode not in EMBED_MODES)}


# ─── Modalità di ricerca ──────────────────────────────────────────────────
# Dove cercano Query e Ricerca agentica: una modalità o tutte ("all"). Si sceglie
# nelle due pagine (e in Admin) e ogni richiesta la porta con sé; l'Upload non la
# cambia. L'ultima scelta è salvata in MongoDB come predefinita per la prossima apertura.
SEARCH_MODES = [*EMBED_MODES, "all"]

_SETTINGS_COLLECTION = "app_settings"
_ACTIVE_MODE_ID = "active_embed_mode"

MODE_DESCRIPTIONS = {
    "auto": ("Testo · embedding generati da Atlas",
             f"Atlas genera i vettori con {config.VOYAGE_EMBED_MODEL} direttamente nell'indice: "
             "l'app salva solo il testo dei chunk."),
    "manual": ("Testo · embedding generati dall'app",
               f"L'app chiama Voyage ({config.VOYAGE_EMBED_MODEL}) e salva il vettore in ogni chunk."),
    "context": ("Testo · embedding con il contesto del documento",
                f"Voyage ({config.VOYAGE_CONTEXT_MODEL}) calcola ogni vettore tenendo conto degli altri "
                "passaggi del documento."),
    "multimodal": ("Testo e immagini delle pagine",
                   f"Ogni pagina o slide è un solo passaggio: immagine e testo nello stesso vettore "
                   f"({config.VOYAGE_MULTIMODAL_MODEL}). Si trovano anche grafici, diagrammi e schermate."),
}


def load_active_mode(default: str | None = None) -> str:
    """Modalità attiva salvata; se assente o non leggibile, il default (EMBED_MODE)."""
    fallback = default if default in EMBED_MODES else (config.EMBED_MODE if config.EMBED_MODE in EMBED_MODES else "auto")
    try:
        doc = config.get_db()[_SETTINGS_COLLECTION].find_one({"_id": _ACTIVE_MODE_ID})
    except Exception:
        return fallback
    mode = (doc or {}).get("mode")
    return mode if mode in SEARCH_MODES else fallback


def save_active_mode(mode: str) -> None:
    if mode not in SEARCH_MODES:
        raise ValueError(f"Modalità sconosciuta: {mode}")
    from datetime import datetime, timezone
    config.get_db()[_SETTINGS_COLLECTION].update_one(
        {"_id": _ACTIVE_MODE_ID},
        {"$set": {"mode": mode, "updated_at": datetime.now(timezone.utc)}},
        upsert=True,
    )


def require_scope(scope: str) -> dict:
    """Rilegge lo stato al clic: non esclude silenziosamente indici non pronti."""
    info = inspect_scope(scope)
    if not info["modes"] or not any(row["Chunk presenti"] for row in info["rows"] if row["Inclusa"] == "Sì"):
        label = scope if scope != "all" else "tutte le modalità"
        raise ValueError(f"Non ci sono documenti caricati in modalità `{label}`. Caricali dalla pagina Upload "
                         "con questa modalità, oppure scegli un'altra modalità di ricerca.")
    if info["unavailable"]:
        raise ValueError("Indici non interrogabili: " + ", ".join(info["unavailable"])
                         + ". Apri Admin → Prepara database e indici e attendi che siano pronti.")
    if info["mismatched"]:
        raise ValueError("Definizione degli indici diversa da quella configurata: "
                         + ", ".join(info["mismatched"])
                         + ". Controlla il piano degli indici in Admin prima di cercare.")
    return info


def prepare_vectors(query: str, modes: list[str]) -> dict[str, list[float]]:
    """Calcola in parallelo i vettori della domanda per le modalità non-auto."""
    targets = [mode for mode in modes if mode != "auto"]
    if not targets:
        return {}
    if len(targets) == 1:
        return {targets[0]: embed_query(query, targets[0])}
    with ThreadPoolExecutor(max_workers=len(targets)) as pool:
        futures = {mode: pool.submit(embed_query, query, mode) for mode in targets}
        return {mode: future.result() for mode, future in futures.items()}


def weights_caption(modes: list[str], custom: dict[str, float] | None = None) -> str:
    """Descrive i pesi effettivi della fusione per la UI."""
    weights = fusion_weights([f"vector_{mode}" for mode in modes] + ["text"], custom)
    if custom:
        it = lambda x: f"{x:g}".replace(".", ",")
        return (f"Pesi scelti per questa domanda: ricerca testuale {it(weights['text'])}, "
                f"ricerca semantica {it(custom['vector'])}" + (" per ogni modalità." if len(modes) > 1 else "."))
    if weights["text"] == 1 or len(modes) <= 1:
        return "Ogni ramo della fusione ha peso 1."
    return (f"Ogni ramo semantico ha peso 1; il ramo testuale ha peso {weights['text']:g}, "
            "pari alla somma dei rami semantici, così aggiungere modalità di embedding "
            "non riduce il contributo della ricerca per parole.")


def dedup_caption() -> str:
    return ("Lo stesso passaggio salvato in più modalità viene unito in un solo risultato prima del reranking."
            if config.HYBRID_DEDUP else
            "Le copie dello stesso passaggio salvate in modalità diverse restano risultati separati.")


def pipeline_code(pipeline: list, info: dict) -> str:
    return (f"db.getSiblingDB({json.dumps(info['database'])})"
            f".getCollection({json.dumps(info['collection'])}).aggregate(\n"
            + json.dumps(pipeline, ensure_ascii=False, indent=2, default=str) + "\n)")
