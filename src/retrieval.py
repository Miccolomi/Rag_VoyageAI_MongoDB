"""Pipeline di retrieval. Supporta 3 modalità:

- "text"   → solo $search (BM25 / keyword full-text)
- "vector" → solo $vectorSearch (semantica via embedding Voyage)
- "hybrid" → $rankFusion(vector, text) con Reciprocal Rank Fusion nativa

In tutte e tre, filtri sui metadati come pre-filter dove possibile.
Con embed_mode="all", modes elenca le modalità presenti nel database:
un ramo testuale condiviso e un ramo vettoriale per modello, fusi per rango.
In modalità autoEmbed la query è una stringa; negli altri percorsi ogni
queryVector viene calcolato con il modello corrispondente.
"""
from __future__ import annotations

from typing import Any, Literal

from pymongo.collection import Collection

from . import config
from .embeddings import embed_voyage, embed_voyage_contextualized
from .rerank import format_rerank_query

RetrievalMode = Literal["text", "vector", "hybrid"]
EMBED_MODES = ("auto", "manual", "context", "multimodal")
_SEARCH_FILTER_FIELDS = {
    "document_id",
    "filename",
    "author",
    "language",
    "content_type",
    "page_number",
    "category",
    "status",
    "year",
    "tags",
    "file_type",
    "embedding_mode",
}


# -------------------------------------------------------- stage builders ----

def embed_query(query: str, embed_mode: str) -> list[float]:
    """Embed la query con il modello coerente alla modalità di indicizzazione."""
    if embed_mode == "manual":
        return embed_voyage([query], input_type="query")[0]
    if embed_mode == "context":
        # context-4: input nested anche per la query → estraggo il vettore unico
        result = embed_voyage_contextualized([[query]], input_type="query")
        return result[0][0]
    if embed_mode == "multimodal":
        from .embeddings import embed_voyage_multimodal
        return embed_voyage_multimodal(
            [[{"type": "text", "text": query}]], input_type="query"
        )[0]
    raise ValueError(f"embed_query non gestisce embed_mode={embed_mode!r}")


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
        "index": config.vector_index_name(embed_mode),
        "path": {
            "auto": "content", "manual": "embedding",
            "context": "embedding_context", "multimodal": "embedding_multimodal",
        }[embed_mode],
        "numCandidates": num_candidates,
        "limit": k,
    }
    if embed_mode == "auto":
        base["query"] = query
    else:
        # Se disponibile, riusa lo stesso vettore tra retrieval e visualizzazione
        # della pipeline; altrimenti lo calcola qui.
        base["queryVector"] = query_vector or embed_query(query, embed_mode)
    # Ogni ramo cerca solo nella sua modalità: il filtro sulla modalità sostituisce quello
    # generale (che può elencarne più d'una), senza ripeterlo due volte nello stesso $and.
    mode_filter = {"embedding_mode": embed_mode}
    other = {key: value for key, value in (filters or {}).items() if key != "embedding_mode"}
    base["filter"] = {"$and": [other, mode_filter]} if other else mode_filter
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

# Testo passato a $rerank: le stesse etichette del percorso via API (rerank.py).
# I campi facoltativi possono mancare: $rerank deve sempre ricevere una stringa.
RERANK_TEXT: dict[str, Any] = {"$concat": [
    "Titolo: ", {"$ifNull": ["$title", ""]},
    "\nFonte: ", {"$ifNull": ["$filename", ""]},
    ", pagina ", {"$convert": {"input": "$page_number", "to": "string", "onNull": "n/d", "onError": "n/d"}},
    "\nTipo contenuto: ", {"$ifNull": ["$content_type", ""]},
    "\nCategoria: ", {"$ifNull": ["$category", ""]},
    "\nContenuto:\n", {"$ifNull": ["$content", ""]},
]}

PROJECT_COMMON: dict[str, Any] = {
    "document_id": 1,
    "filename": 1,
    "page_number": 1,
    "chunk_index": 1,
    "content_type": 1,
    "content": 1,
    "title": 1,
    "category": 1,
    "gridfs_file_id": 1,
    "embedding_mode": 1,
}


def fusion_weights(branch_names: list[str], weights: dict[str, float] | None = None) -> dict[str, float]:
    """Pesi di $rankFusion per ramo.

    Con HYBRID_TEXT_WEIGHT_MODE=balanced il ramo testuale pesa quanto la somma
    dei rami vettoriali: aggiungere modalità di embedding non riduce il peso
    della ricerca per parole. Pesi espliciti passati dal chiamante prevalgono.
    """
    weights = weights or {}
    vector_branches = [name for name in branch_names if name.startswith("vector")]
    default_text = (float(len(vector_branches))
                    if config.HYBRID_TEXT_WEIGHT_MODE == "balanced" and vector_branches else 1.0)
    result = {}
    for name in branch_names:
        if name in weights:
            result[name] = float(weights[name])
        elif name.startswith("vector"):
            result[name] = float(weights.get("vector", 1.0))
        else:
            result[name] = float(weights.get("text", default_text))
    return result


def dedup_stages(k: int) -> list[dict[str, Any]]:
    """Unisce le copie dello stesso passaggio salvate in modalità diverse.

    Lo stesso testo ingerito in auto, manual, context e multimodal produce
    documenti con _id diversi: $rankFusion li tratta come passaggi distinti.
    Qui si tiene, per ogni (document_id, chunk_hash), la copia con il punteggio
    di fusione più alto e si registra in quali modalità compariva.
    """
    return [
        {"$project": {**PROJECT_COMMON, "chunk_hash": 1,
                      "_fusion_score": 1, "_fusion_details": 1}},
        {"$sort": {"_fusion_score": -1, "_id": 1}},
        {"$group": {
            "_id": {"document_id": "$document_id",
                    "chunk": {"$ifNull": ["$chunk_hash", "$_id"]}},
            "doc": {"$first": "$$ROOT"},
            "modes": {"$addToSet": "$embedding_mode"},
            "copies": {"$sum": 1},
        }},
        {"$set": {"doc._found_in_modes": "$modes", "doc._copies": "$copies"}},
        {"$replaceRoot": {"newRoot": "$doc"}},
        {"$sort": {"_fusion_score": -1, "_id": 1}},
        {"$limit": k},
    ]


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
    modes: list[str] | None = None,
    query_vectors: dict[str, list[float]] | None = None,
) -> list[dict[str, Any]]:
    k = top_k or config.RETRIEVAL_TOP_K
    num_candidates = max(
        k * config.RETRIEVAL_NUM_CANDIDATES_MULTIPLIER,
        config.RETRIEVAL_NUM_CANDIDATES_MIN,
    )
    embed_mode = (embed_mode or config.EMBED_MODE).lower()
    if embed_mode not in (*EMBED_MODES, "all"):
        raise ValueError(f"embed_mode non valida: {embed_mode!r}")
    active_modes = list(dict.fromkeys(modes or [])) if embed_mode == "all" else [embed_mode]
    if not active_modes or any(mode not in EMBED_MODES for mode in active_modes):
        raise ValueError("Specificare le modalità presenti nel database per la ricerca completa.")
    if embed_mode == "all" and query_vector is not None:
        raise ValueError("Usare query_vectors per modalità: un unico vettore non può essere condiviso tra modelli.")
    # Stesso perimetro per tutti i canali; ogni ramo semantico usa il proprio
    # spazio vettoriale. La fusione combina ranghi, mai coordinate o similarità.
    mode_filter = active_modes[0] if len(active_modes) == 1 else {"$in": active_modes}
    filters = {**(filters or {}), "embedding_mode": mode_filter}

    if retrieval_mode == "text":
        return _text_pipeline(query, k=k, filters=filters, fuzzy=fuzzy) + [
            {"$project": {**PROJECT_COMMON, "score": {"$meta": "searchScore"}}}
        ]

    if retrieval_mode == "vector" and len(active_modes) == 1:
        vec = _vector_stage(
            query, embed_mode=active_modes[0], k=k, num_candidates=num_candidates,
            filters=filters, query_vector=(query_vectors or {}).get(active_modes[0], query_vector),
        )
        return [
            vec,
            {"$project": {**PROJECT_COMMON, "score": {"$meta": "vectorSearchScore"}}},
        ]

    if retrieval_mode in ("hybrid", "vector"):
        branches = {}
        for mode in active_modes:
            name = f"vector_{mode}" if embed_mode == "all" else "vector"
            branches[name] = [_vector_stage(
                query, embed_mode=mode, k=k, num_candidates=num_candidates,
                filters=filters, query_vector=(query_vectors or {}).get(mode, query_vector),
            )]
        if retrieval_mode == "hybrid":
            branches["text"] = _text_pipeline(query, k=k, filters=filters, fuzzy=fuzzy)
        w = fusion_weights(list(branches), weights)
        pipeline = [
            {
                "$rankFusion": {
                    "input": {
                        "pipelines": branches
                    },
                    "combination": {"weights": w},
                    "scoreDetails": True,
                }
            },
        ]
        # I metadati di fusione vanno copiati in campi prima di $group:
        # dopo il raggruppamento $meta non è più disponibile.
        pipeline.append({"$set": {"_fusion_score": {"$meta": "score"},
                                  "_fusion_details": {"$meta": "scoreDetails"}}})
        # Difesa aggiuntiva: i filtri sono già applicati in ogni ramo prima
        # del ranking, ma questo match impedisce che un risultato fuori scope
        # possa raggiungere reranker e LLM dopo future modifiche alla pipeline.
        if filters:
            pipeline.append({"$match": filters})
        if config.HYBRID_DEDUP:
            pipeline.extend(dedup_stages(k))
        else:
            pipeline.append({"$limit": k})
        pipeline.append({
            "$project": {
                **PROJECT_COMMON,
                "chunk_hash": 1,
                **({"_found_in_modes": 1, "_copies": 1} if config.HYBRID_DEDUP else {}),
                "score": "$_fusion_score",
                "scoreDetails": "$_fusion_details",
            }
        })
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
    modes: list[str] | None = None,
    query_vectors: dict[str, list[float]] | None = None,
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
            modes=modes,
            query_vectors=query_vectors,
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
    modes: list[str] | None = None,
    query_vectors: dict[str, list[float]] | None = None,
    instructions: str | None = None,
    include_candidates: bool = False,
) -> list[dict[str, Any]]:
    """Ricerca ibrida seguita da $rerank, fuori dai rami di $rankFusion.

    include_candidates restituisce tutti i candidati riordinati per ricostruire
    la classifica prima/dopo senza eseguire una seconda ricerca né un secondo rerank.
    """
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
        modes=modes,
        query_vectors=query_vectors,
    )
    if not 1 <= k <= 1000 or not 1 <= n <= k:
        raise ValueError("Il reranking richiede 1 <= Top-N <= Top-K <= 1000")
    pipeline.extend([
        {"$setWindowFields": {
            "sortBy": {"score": -1},
            "output": {"_hybrid_rank": {"$documentNumber": {}}},
        }},
        {"$sort": {"_hybrid_rank": 1}},
        {"$set": {"_rerank_text": RERANK_TEXT}},
        {
            "$rerank": {
                "query": {"text": format_rerank_query(query, instructions)},
                "model": model or config.VOYAGE_RERANK_MODEL,
                "path": "_rerank_text",
                "numDocsToRerank": k,
            }
        },
        {"$set": {"_rerank_score": {"$meta": "score"}}},
        {"$unset": "_rerank_text"},
    ])
    if not include_candidates:
        pipeline.append({"$limit": n})
    return pipeline

