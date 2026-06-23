"""Reranker Voyage AI (cross-encoder) — chiamata HTTP diretta, no SDK."""
from __future__ import annotations

from typing import Any

from . import config
from .voyage_http import post_json


def _format_rerank_document(doc: dict[str, Any], text_field: str) -> str:
    parts = []
    if doc.get("title"):
        parts.append(f"Titolo: {doc['title']}")
    if doc.get("filename"):
        page = doc.get("page_number")
        page_label = f", pagina {page}" if page not in (None, "") else ""
        parts.append(f"Fonte: {doc['filename']}{page_label}")
    if doc.get("content_type"):
        parts.append(f"Tipo contenuto: {doc['content_type']}")
    if doc.get("category"):
        parts.append(f"Categoria: {doc['category']}")
    parts.append(f"Contenuto:\n{doc.get(text_field, '')}")
    return "\n".join(parts)


def rerank(
    query: str,
    documents: list[dict[str, Any]],
    *,
    top_n: int | None = None,
    model: str | None = None,
    text_field: str = "content",
    timeout: float = 30.0,
) -> list[dict[str, Any]]:
    """Riordina `documents` rispetto a `query`. Ritorna i top-N con score.

    Ogni elemento del risultato è il documento originale + `_rerank_score` + `_rerank_rank`.
    """
    if not documents:
        return []
    if not config.VOYAGE_API_KEY:
        raise RuntimeError("VOYAGE_API_KEY non impostata nel .env")

    top_n = top_n or config.RERANK_TOP_N
    model = model or config.VOYAGE_RERANK_MODEL

    texts = [_format_rerank_document(d, text_field) for d in documents]
    payload = {
        "query": query,
        "documents": texts,
        "model": model,
        "top_k": min(top_n, len(texts)),
        "return_documents": False,
        "truncation": False,
    }
    headers = {
        "Authorization": f"Bearer {config.VOYAGE_API_KEY}",
        "Content-Type": "application/json",
    }
    url = f"{config.VOYAGE_API_BASE}/rerank"
    data = post_json(
        url,
        payload=payload,
        headers=headers,
        timeout=timeout,
        operation="reranking",
    )

    out: list[dict[str, Any]] = []
    for rank, item in enumerate(data.get("data", []), start=1):
        idx = item["index"]
        doc = dict(documents[idx])
        doc["_rerank_score"] = item.get("relevance_score")
        doc["_rerank_rank"] = rank
        out.append(doc)
    return out
