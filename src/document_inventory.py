"""Read-only inventory of the embedding metadata stored with document chunks."""
from __future__ import annotations

from typing import Any

from . import config


def document_inventory(collection) -> dict[str, Any]:
    grouped = collection.aggregate([
        {"$group": {
            "_id": {"document_id": "$document_id", "filename": "$filename",
                    "mode": "$embedding_mode", "model": "$embedding_model",
                    "strategy": "$context_strategy"},
            "chunks": {"$sum": 1}, "first_ingested": {"$min": "$ingested_at"},
            "last_ingested": {"$max": "$ingested_at"},
        }},
    ])
    rows = []
    for item in grouped:
        metadata = item["_id"]
        rows.append({
            "Documento": metadata.get("filename") or "Nome non disponibile",
            "ID documento": str(metadata.get("document_id") or "Non disponibile"),
            "Modalità registrata": metadata.get("mode") or "Non disponibile",
            "Modello registrato": metadata.get("model") or "Non disponibile",
            "Strategia registrata": metadata.get("strategy") or "Non disponibile",
            "Chunk": item["chunks"],
            "Primo caricamento": item.get("first_ingested"),
            "Ultimo caricamento": item.get("last_ingested"),
        })
    rows.sort(key=lambda row: (str(row["Documento"]).casefold(), str(row["ID documento"]),
                               str(row["Modalità registrata"]), str(row["Modello registrato"])))
    index_info: dict[str, Any] = {"name": config.vector_index_name("auto"), "status": "Non disponibile", "model": None}
    try:
        for index in collection.list_search_indexes():
            if index.get("name") != index_info["name"]:
                continue
            index_info["status"] = index.get("status", "Non disponibile")
            definition = index.get("latestDefinition") or index.get("definition") or {}
            index_info["model"] = next((field.get("model") for field in definition.get("fields", [])
                                        if field.get("type") == "autoEmbed" and field.get("path") == "content"), None)
            break
    except Exception:
        # Metadata is still useful when the account cannot inspect index definitions.
        pass
    return {"rows": rows, "auto_index": index_info}
