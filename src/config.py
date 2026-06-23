"""Caricamento configurazione da .env e factory per il client MongoDB."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

load_dotenv()

# Root del progetto (cartella che contiene src/, prompts/, schemas/, …)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

MONGODB_URI: str = os.environ["MONGODB_URI"]
MONGODB_DB: str = os.getenv("MONGODB_DB", "Rag_PDF")
MONGODB_COLLECTION: str = os.getenv("MONGODB_COLLECTION", "pdf_chunks")
MONGODB_GRIDFS_BUCKET: str = os.getenv("MONGODB_GRIDFS_BUCKET", "pdf_files")

VECTOR_INDEX_NAME: str = os.getenv("VECTOR_INDEX_NAME", "pdf_chunks_vector_idx")
SEARCH_INDEX_NAME: str = os.getenv("SEARCH_INDEX_NAME", "pdf_chunks_text_idx")

# Voyage API:
#  - VOYAGE_API_KEY: la chiave. Se "al-..." → endpoint Atlas. Se "pa-..." → endpoint Voyage.
#  - VOYAGE_API_BASE: base URL. Default Atlas (compatibile con chiavi "al-").
# Se la chiave manca, l'app salta il rerank.
VOYAGE_API_KEY: str | None = os.getenv("VOYAGE_API_KEY") or None
VOYAGE_API_BASE: str = os.getenv("VOYAGE_API_BASE", "https://ai.mongodb.com/v1").rstrip("/")
VOYAGE_EMBED_MODEL: str = os.getenv("VOYAGE_EMBED_MODEL", "voyage-4-large")
VOYAGE_RERANK_MODEL: str = os.getenv("VOYAGE_RERANK_MODEL", "rerank-2.5")
# Modalità embedding:
#   "auto"    → Atlas autoEmbed (vettori solo nell'indice, no campo embedding nei doc)
#   "manual"  → Python chiama Voyage /embeddings (vettori in campo `embedding`)
#   "context" → Python chiama Voyage /contextualizedembeddings con
#               chunk locali page-aware raggruppati per documento
EMBED_MODE: str = os.getenv("EMBED_MODE", "auto").lower()
VOYAGE_EMBED_DIMENSIONS: int = int(os.getenv("VOYAGE_EMBED_DIMENSIONS", "2048"))
# voyage-context-4: ultima generazione, 120K token per documento
# (1024 dim default, supporta anche 256/512/2048)
VOYAGE_CONTEXT_MODEL: str = os.getenv("VOYAGE_CONTEXT_MODEL", "voyage-context-4")
VOYAGE_CONTEXT_DIMENSIONS: int = int(os.getenv("VOYAGE_CONTEXT_DIMENSIONS", "1024"))

LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "anthropic").lower()

ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY") or None
ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "mistral-nemo:latest")
OLLAMA_VISION_MODEL: str = os.getenv("OLLAMA_VISION_MODEL", "llava:7b")

GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY") or None
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

CHUNK_SIZE_TOKENS: int = int(os.getenv("CHUNK_SIZE_TOKENS", "512"))
CHUNK_OVERLAP_TOKENS: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "64"))
INGESTION_DUPLICATE_POLICY: str = os.getenv("INGESTION_DUPLICATE_POLICY", "skip").lower()
INGESTION_PIPELINE_VERSION: str = os.getenv("INGESTION_PIPELINE_VERSION", "v3")
CHUNKER_VERSION: str = os.getenv("CHUNKER_VERSION", "token-window-v2")

RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "50"))
RETRIEVAL_NUM_CANDIDATES_MULTIPLIER: int = int(
    os.getenv("RETRIEVAL_NUM_CANDIDATES_MULTIPLIER", "10")
)
RETRIEVAL_NUM_CANDIDATES_MIN: int = int(os.getenv("RETRIEVAL_NUM_CANDIDATES_MIN", "100"))
RERANK_TOP_N: int = int(os.getenv("RERANK_TOP_N", "8"))
# Sotto questa soglia il top chunk rerank è considerato "off-topic" → no LLM call
RERANK_RELEVANCE_THRESHOLD: float = float(os.getenv("RERANK_RELEVANCE_THRESHOLD", "0.15"))

TESSERACT_CMD: str | None = os.getenv("TESSERACT_CMD") or None


# ─── System prompt del LLM ────────────────────────────────────────────────
# Vive in `prompts/system_prompt.txt` per essere editabile senza toccare
# il codice. Override opzionale via env var LLM_SYSTEM_PROMPT (stringa
# inline) o LLM_SYSTEM_PROMPT_FILE (path a file alternativo).

_DEFAULT_SYSTEM_PROMPT_FILE = _PROJECT_ROOT / "prompts" / "system_prompt.txt"


def _load_system_prompt() -> str:
    # 1. env inline ha precedenza assoluta
    inline = os.getenv("LLM_SYSTEM_PROMPT")
    if inline:
        return inline
    # 2. env path a file alternativo
    path_env = os.getenv("LLM_SYSTEM_PROMPT_FILE")
    if path_env:
        p = Path(path_env)
        if not p.is_absolute():
            p = _PROJECT_ROOT / p
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
    # 3. file di default `prompts/system_prompt.txt`
    if _DEFAULT_SYSTEM_PROMPT_FILE.exists():
        return _DEFAULT_SYSTEM_PROMPT_FILE.read_text(encoding="utf-8").strip()
    # 4. fallback hardcoded (sopravvivenza se cancellano il file)
    return (
        "Sei un assistente che risponde basandosi solo sui PASSAGGI forniti. "
        "Cita le fonti con [#N]. Se non c'è la risposta nei passaggi, di' "
        "'Non ho informazioni sufficienti nei passaggi forniti per rispondere.'"
    )


LLM_SYSTEM_PROMPT: str = _load_system_prompt()


@lru_cache(maxsize=1)
def get_client() -> MongoClient:
    return MongoClient(MONGODB_URI)


def get_db() -> Database:
    return get_client()[MONGODB_DB]


def get_collection() -> Collection:
    return get_db()[MONGODB_COLLECTION]
