"""Creazione / verifica degli indici Atlas Search a partire dagli schemi JSON.

Supporta tre modalità di indice vector:
- "auto"    → autoEmbed con Voyage (richiede cluster M30+, vettori NON nei documenti)
- "manual"  → vector classico (campo `embedding` calcolato dall'app, dim da .env)
- "context" → vector classico con voyage-context-4 (campo `embedding`, 1024 dim default)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pymongo.collection import Collection
from pymongo.operations import SearchIndexModel

from . import config

_SCHEMAS_DIR = Path(__file__).resolve().parent.parent / "schemas"

Mode = Literal["auto", "manual", "context"]


def _load(name: str) -> dict[str, Any]:
    with (_SCHEMAS_DIR / name).open() as f:
        return json.load(f)


def _existing_indexes(coll: Collection) -> set[str]:
    try:
        return {idx["name"] for idx in coll.list_search_indexes()}
    except Exception:
        return set()


def _vector_schema(mode: Mode) -> dict[str, Any]:
    if mode == "auto":
        schema = _load("vector_index.json")
        # Allinea il modello a quello in env
        for f in schema["definition"]["fields"]:
            if f.get("type") == "autoEmbed":
                f["model"] = config.VOYAGE_EMBED_MODEL
        return schema
    elif mode == "manual":
        schema = _load("vector_index_manual.json")
        # Allinea numDimensions all'env (deve combaciare con il modello scelto)
        for f in schema["definition"]["fields"]:
            if f.get("type") == "vector":
                f["numDimensions"] = config.VOYAGE_EMBED_DIMENSIONS
        return schema
    elif mode == "context":
        schema = _load("vector_index_context.json")
        # voyage-context-4: 1024 default, allineato all'env
        for f in schema["definition"]["fields"]:
            if f.get("type") == "vector":
                f["numDimensions"] = config.VOYAGE_CONTEXT_DIMENSIONS
        return schema
    raise ValueError(f"mode non valido: {mode}")


def ensure_indexes(
    coll: Collection,
    *,
    mode: Mode = "auto",
    recreate: bool = False,
) -> dict[str, Any]:
    """Crea (o ricrea) i due indici. Idempotente se recreate=False.

    recreate=True → droppa l'indice esistente prima di crearlo (utile dopo cambio
    modello o cambio modalità, dato che il modello/dimensioni sono cablate).
    """
    vec = _vector_schema(mode)
    txt = _load("search_index.json")

    vec["name"] = config.VECTOR_INDEX_NAME
    txt["name"] = config.SEARCH_INDEX_NAME

    existing = _existing_indexes(coll)
    dropped: list[str] = []
    created: list[str] = []

    for spec, is_vector in [(vec, True), (txt, False)]:
        name = spec["name"]
        if name in existing:
            if recreate:
                coll.drop_search_index(name)
                dropped.append(name)
            else:
                continue

        coll.create_search_index(
            SearchIndexModel(
                definition=spec["definition"],
                name=name,
                type="vectorSearch" if is_vector else None,
            )
        )
        created.append(name)

    return {
        "mode": mode,
        "recreate": recreate,
        "dropped": dropped,
        "created": created,
        "already_present": sorted(existing - set(dropped)),
    }


if __name__ == "__main__":
    print(ensure_indexes(config.get_collection(), mode=config.EMBED_MODE))
