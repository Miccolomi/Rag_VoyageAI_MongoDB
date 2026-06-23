"""Pipeline di retrieval. Supporta 3 modalità:

- "text"   → solo $search (BM25 / keyword full-text)
- "vector" → solo $vectorSearch (semantica via embedding Voyage)
- "hybrid" → $rankFusion(vector, text) con Reciprocal Rank Fusion nativa

In tutte e tre, filtri sui metadati come pre-filter dove possibile.
In modalità autoEmbed la query semantica è una stringa; in modalità manual
viene calcolato `queryVector` dall'app via Voyage embed API.
"""
from __future__ import annotations

from typing import Any, Literal

from pymongo.collection import Collection

from . import config
from .embeddings import embed_voyage, embed_voyage_contextualized

RetrievalMode = Literal["text", "vector", "hybrid"]
_SEARCH_FILTER_FIELDS = {
    "document_id",
    "filename",
    "language",
    "content_type",
    "page_number",
    "category",
    "status",
    "year",
    "tags",
    "file_type",
}


# -------------------------------------------------------- stage builders ----

def _query_vector(query: str, embed_mode: str) -> list[float]:
    """Embed la query con il modello coerente alla modalità di indicizzazione."""
    if embed_mode == "manual":
        return embed_voyage([query], input_type="query")[0]
    if embed_mode == "context":
        # context-4: input nested anche per la query → estraggo il vettore unico
        result = embed_voyage_contextualized([[query]], input_type="query")
        return result[0][0]
    raise ValueError(f"_query_vector non gestisce embed_mode={embed_mode!r}")


def _vector_stage(
    query: str,
    *,
    embed_mode: str,
    k: int,
    num_candidates: int,
    filters: dict[str, Any] | None,
    query_vector: list[float] | None = None,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "index": config.VECTOR_INDEX_NAME,
        "path": "content" if embed_mode == "auto" else "embedding",
        "numCandidates": num_candidates,
        "limit": k,
    }
    if embed_mode == "auto":
        base["query"] = query
    else:
        # Se disponibile, riusa lo stesso vettore tra retrieval e visualizzazione
        # della pipeline; altrimenti lo calcola qui.
        base["queryVector"] = query_vector or _query_vector(query, embed_mode)
    if filters:
        base["filter"] = filters
    return {"$vectorSearch": base}


def _text_pipeline(
    query: str,
    *,
    k: int,
    filters: dict[str, Any] | None,
    fuzzy: bool = False,
) -> list[dict[str, Any]]:
    text_block: dict[str, Any] = {"query": query, "path": "content"}
    if fuzzy:
        # Tollera typo: maxEdits=1 = una sostituzione/inserzione/cancellazione di carattere.
        text_block["fuzzy"] = {"maxEdits": 1}
    search_body: dict[str, Any]
    native_filters = _search_filter_clauses(filters)
    if native_filters:
        search_body = {
            "compound": {
                "must": [{"text": text_block}],
                "filter": native_filters,
            }
        }
    else:
        search_body = {"text": text_block}
    pipeline: list[dict[str, Any]] = [
        {"$search": {"index": config.SEARCH_INDEX_NAME, **search_body}}
    ]
    residual_filters = _residual_match_filters(filters)
    if residual_filters:
        pipeline.append({"$match": residual_filters})
    pipeline.append({"$limit": k})
    return pipeline


def _search_filter_clauses(filters: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not filters:
        return []
    clauses: list[dict[str, Any]] = []
    for path, value in filters.items():
        if path not in _SEARCH_FILTER_FIELDS:
            continue
        if isinstance(value, dict) and "$in" in value:
            clauses.append({"in": {"path": path, "value": value["$in"]}})
        elif isinstance(value, dict):
            continue
        else:
            clauses.append({"equals": {"path": path, "value": value}})
    return clauses


def _residual_match_filters(filters: dict[str, Any] | None) -> dict[str, Any] | None:
    if not filters:
        return None
    residual = {
        path: value
        for path, value in filters.items()
        if path not in _SEARCH_FILTER_FIELDS
        or (isinstance(value, dict) and "$in" not in value)
    }
    return residual or None


# --------------------------------------------------------- projection ----

_PROJECT_COMMON: dict[str, Any] = {
    "document_id": 1,
    "filename": 1,
    "page_number": 1,
    "chunk_index": 1,
    "content_type": 1,
    "content": 1,
    "title": 1,
    "category": 1,
    "gridfs_file_id": 1,
}


# ------------------------------------------------------------ pipelines ----

def build_pipeline(
    query: str,
    *,
    filters: dict[str, Any] | None = None,
    top_k: int | None = None,
    weights: dict[str, float] | None = None,
    embed_mode: str | None = None,
    retrieval_mode: RetrievalMode = "hybrid",
    fuzzy: bool = False,
    query_vector: list[float] | None = None,
) -> list[dict[str, Any]]:
    k = top_k or config.RETRIEVAL_TOP_K
    num_candidates = max(
        k * config.RETRIEVAL_NUM_CANDIDATES_MULTIPLIER,
        config.RETRIEVAL_NUM_CANDIDATES_MIN,
    )
    embed_mode = (embed_mode or config.EMBED_MODE).lower()

    if retrieval_mode == "text":
        return _text_pipeline(query, k=k, filters=filters, fuzzy=fuzzy) + [
            {"$project": {**_PROJECT_COMMON, "score": {"$meta": "searchScore"}}}
        ]

    if retrieval_mode == "vector":
        vec = _vector_stage(
            query, embed_mode=embed_mode, k=k, num_candidates=num_candidates,
            filters=filters, query_vector=query_vector,
        )
        return [
            vec,
            {"$project": {**_PROJECT_COMMON, "score": {"$meta": "vectorSearchScore"}}},
        ]

    if retrieval_mode == "hybrid":
        w = weights or {"vector": 1.0, "text": 1.0}
        vec = _vector_stage(
            query, embed_mode=embed_mode, k=k, num_candidates=num_candidates,
            filters=filters, query_vector=query_vector,
        )
        pipeline = [
            {
                "$rankFusion": {
                    "input": {
                        "pipelines": {
                            "vector": [vec],
                            "text": _text_pipeline(query, k=k, filters=filters, fuzzy=fuzzy),
                        }
                    },
                    "combination": {"weights": w},
                    "scoreDetails": True,
                }
            },
        ]
        # Difesa aggiuntiva: i filtri sono già applicati nei due rami prima
        # del ranking, ma questo match impedisce che un risultato fuori scope
        # possa raggiungere reranker e LLM dopo future modifiche alla pipeline.
        if filters:
            pipeline.append({"$match": filters})
        pipeline.extend([
            {"$limit": k},
            {
                "$project": {
                    **_PROJECT_COMMON,
                    "score": {"$meta": "score"},
                    "scoreDetails": {"$meta": "scoreDetails"},
                }
            },
        ])
        return pipeline

    raise ValueError(f"retrieval_mode non valido: {retrieval_mode}")


def search(
    coll: Collection,
    query: str,
    *,
    filters: dict[str, Any] | None = None,
    top_k: int | None = None,
    embed_mode: str | None = None,
    retrieval_mode: RetrievalMode = "hybrid",
    fuzzy: bool = False,
    query_vector: list[float] | None = None,
) -> list[dict[str, Any]]:
    return list(coll.aggregate(
        build_pipeline(
            query,
            filters=filters,
            top_k=top_k,
            embed_mode=embed_mode,
            retrieval_mode=retrieval_mode,
            fuzzy=fuzzy,
            query_vector=query_vector,
        )
    ))


def build_native_rerank_pipeline(
    query: str,
    *,
    filters: dict[str, Any] | None = None,
    top_k: int | None = None,
    top_n: int | None = None,
    embed_mode: str | None = None,
    fuzzy: bool = False,
    query_vector: list[float] | None = None,
    model: str | None = None,
) -> list[dict[str, Any]]:
    """Hybrid retrieval seguito dal reranking Voyage dentro MongoDB."""
    k = top_k or config.RETRIEVAL_TOP_K
    n = min(top_n or config.RERANK_TOP_N, k)
    pipeline = build_pipeline(
        query,
        filters=filters,
        top_k=k,
        embed_mode=embed_mode,
        retrieval_mode="hybrid",
        fuzzy=fuzzy,
        query_vector=query_vector,
    )
    pipeline.extend([
        {
            "$rerank": {
                "query": {"text": query},
                "model": model or config.VOYAGE_RERANK_MODEL,
                "path": "content",
                "numDocsToRerank": k,
            }
        },
        {"$limit": n},
        {"$set": {"_rerank_score": {"$meta": "score"}}},
    ])
    return pipeline


def search_with_native_rerank(
    coll: Collection,
    query: str,
    *,
    filters: dict[str, Any] | None = None,
    top_k: int | None = None,
    top_n: int | None = None,
    embed_mode: str | None = None,
    fuzzy: bool = False,
    query_vector: list[float] | None = None,
) -> list[dict[str, Any]]:
    results = list(coll.aggregate(build_native_rerank_pipeline(
        query,
        filters=filters,
        top_k=top_k,
        top_n=top_n,
        embed_mode=embed_mode,
        fuzzy=fuzzy,
        query_vector=query_vector,
    )))
    for rank, doc in enumerate(results, start=1):
        doc["_rerank_rank"] = rank
    return results
