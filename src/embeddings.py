"""Client Voyage AI embeddings usato in modalità manual e context.

- `embed_voyage(texts, ...)` → endpoint /embeddings (modalità manual, modello standard)
- `embed_voyage_contextualized(documents, ...)` → endpoint /contextualizedembeddings
  in modalità pre-chunked page-aware e per embedare la query

In modalità autoEmbed gli embedding li genera Atlas; questo modulo non viene chiamato.
"""
from __future__ import annotations

from typing import Literal

from . import config
from .voyage_http import post_json

_EMBED_BATCH_SIZE = 128


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {config.VOYAGE_API_KEY}",
        "Content-Type": "application/json",
    }


def _ordered_embeddings(data: dict) -> list[list[float]]:
    items = sorted(data.get("data", []), key=lambda item: item.get("index", 0))
    return [item["embedding"] for item in items]


def embed_voyage(
    texts: list[str],
    *,
    model: str | None = None,
    input_type: Literal["document", "query"] = "document",
    timeout: float = 60.0,
) -> list[list[float]]:
    """Ritorna gli embedding nello stesso ordine degli `texts`.

    `input_type`: usa "document" per gli inserts, "query" per le ricerche
    (Voyage ottimizza l'embedding in base al ruolo).
    """
    if not texts:
        return []
    if not config.VOYAGE_API_KEY:
        raise RuntimeError(
            "Modalità manual: serve VOYAGE_API_KEY nel .env (chiave 'al-...' o 'pa-...')"
        )
    model = model or config.VOYAGE_EMBED_MODEL

    url = f"{config.VOYAGE_API_BASE}/embeddings"
    embeddings: list[list[float]] = []
    for start in range(0, len(texts), _EMBED_BATCH_SIZE):
        batch = texts[start:start + _EMBED_BATCH_SIZE]
        data = post_json(
            url,
            payload={
                "input": batch,
                "model": model,
                "input_type": input_type,
                "output_dimension": config.VOYAGE_EMBED_DIMENSIONS,
                "output_dtype": "float",
                "truncation": False,
            },
            headers=_headers(),
            timeout=timeout,
            operation="embedding standard",
        )
        batch_embeddings = _ordered_embeddings(data)
        if len(batch_embeddings) != len(batch):
            raise RuntimeError(
                "Voyage AI ha restituito un numero di embedding diverso "
                "dal numero di testi inviati."
            )
        embeddings.extend(batch_embeddings)
    return embeddings


def embed_voyage_contextualized(
    documents: list[list[str]],
    *,
    model: str | None = None,
    input_type: Literal["document", "query"] = "document",
    timeout: float = 120.0,
) -> list[list[list[float]]]:
    """Contextualized chunk embeddings per documenti già suddivisi in chunk.

    `documents` è una lista di liste: ogni inner list rappresenta i chunk di UN
    documento. Voyage genera ogni vettore usando il contesto degli altri chunk
    dello stesso documento, senza modificare confini e metadati pagina.

    Ritorna: out[doc_idx][chunk_idx] = embedding (lista di float).
    """
    if not documents:
        return []
    if not config.VOYAGE_API_KEY:
        raise RuntimeError(
            "Modalità context: serve VOYAGE_API_KEY nel .env (chiave 'al-...' o 'pa-...')"
        )
    model = model or config.VOYAGE_CONTEXT_MODEL

    url = f"{config.VOYAGE_API_BASE}/contextualizedembeddings"
    data = post_json(
        url,
        payload={
            "inputs": documents,
            "model": model,
            "input_type": input_type,
            "output_dimension": config.VOYAGE_CONTEXT_DIMENSIONS,
            "output_dtype": "float",
        },
        headers=_headers(),
        timeout=timeout,
        operation="embedding contestualizzato",
    )

    out: list[list[list[float]]] = []
    for doc_obj in sorted(data.get("data", []), key=lambda item: item.get("index", 0)):
        chunk_items = sorted(
            doc_obj.get("data", []),
            key=lambda item: item.get("index", 0),
        )
        chunk_embeddings = [c["embedding"] for c in chunk_items]
        out.append(chunk_embeddings)
    return out


def embed_voyage_multimodal(
    inputs: list[list[dict[str, str]]],
    *,
    input_type: Literal["document", "query"] = "document",
    timeout: float = 180.0,
) -> list[list[float]]:
    """Crea vettori multimodali da sequenze di parti text/image_base64."""
    if not inputs:
        return []
    if not config.VOYAGE_API_KEY:
        raise RuntimeError("Modalità multimodal: serve VOYAGE_API_KEY nel .env")
    vectors: list[list[float]] = []
    batch_size = 32  # margine per payload immagine entro i limiti API
    for start in range(0, len(inputs), batch_size):
        batch = inputs[start:start + batch_size]
        data = post_json(
            f"{config.VOYAGE_API_BASE}/multimodalembeddings",
            payload={
                "inputs": [{"content": parts} for parts in batch],
                "model": config.VOYAGE_MULTIMODAL_MODEL,
                "input_type": input_type,
            },
            headers=_headers(),
            timeout=timeout,
            operation="embedding multimodale",
        )
        batch_vectors = _ordered_embeddings(data)
        if len(batch_vectors) != len(batch):
            raise RuntimeError("Voyage AI ha restituito un numero inatteso di vettori multimodali.")
        vectors.extend(batch_vectors)
    for index, vector in enumerate(vectors):
        if len(vector) != config.VOYAGE_MULTIMODAL_DIMENSIONS:
            raise RuntimeError(f"Embedding multimodale #{index}: dimensione inattesa {len(vector)}.")
    return vectors
