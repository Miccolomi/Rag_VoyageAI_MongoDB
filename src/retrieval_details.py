"""Identità dei passaggi e posizioni dei rami della ricerca Atlas."""
from __future__ import annotations

import hashlib
import json


def passage_key(doc: dict) -> str:
    if doc.get("_id") is not None:
        return str(doc["_id"])
    identity = [str(doc.get(field, "")) for field in (
        "document_id", "gridfs_file_id", "filename", "page_number", "chunk_index",
        "content_type", "embedding_mode", "content",
    )]
    return hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode()).hexdigest()


def branch_ranks(doc: dict) -> dict:
    details = (doc.get("scoreDetails") or {}).get("details") or []
    ranks = {}
    for item in details:
        if not isinstance(item, dict):
            continue
        rank = item.get("rank")
        if isinstance(rank, (int, float)) and rank > 0:
            ranks[item.get("inputPipelineName")] = int(rank)
    return ranks

