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
MONGODB_DB: str = os.getenv("MONGODB_DB", "rag")
MONGODB_COLLECTION: str = os.getenv("MONGODB_COLLECTION", "pdf_chunks")
MONGODB_GRIDFS_BUCKET: str = os.getenv("MONGODB_GRIDFS_BUCKET", "pdf_files")

VECTOR_INDEX_NAME: str = os.getenv("VECTOR_INDEX_NAME", "pdf_chunks_vector_idx")
VECTOR_INDEX_MANUAL_NAME: str = os.getenv("VECTOR_INDEX_MANUAL_NAME", "pdf_chunks_manual_idx")
VECTOR_INDEX_CONTEXT_NAME: str = os.getenv("VECTOR_INDEX_CONTEXT_NAME", "pdf_chunks_context_idx")
VECTOR_INDEX_MULTIMODAL_NAME: str = os.getenv("VECTOR_INDEX_MULTIMODAL_NAME", "pdf_chunks_multimodal_idx")
SEARCH_INDEX_NAME: str = os.getenv("SEARCH_INDEX_NAME", "pdf_chunks_text_idx")

# Voyage API:
#  - VOYAGE_API_KEY: la chiave. Se "al-..." → endpoint Atlas. Se "pa-..." → endpoint Voyage.
#  - VOYAGE_API_BASE: base URL. Default Atlas (compatibile con chiavi "al-").
# Il reranking nativo usa l'autorizzazione Atlas; la chiave serve per le API dirette.
VOYAGE_API_KEY: str | None = os.getenv("VOYAGE_API_KEY") or None
VOYAGE_API_BASE: str = os.getenv("VOYAGE_API_BASE", "https://ai.mongodb.com/v1").rstrip("/")
VOYAGE_EMBED_MODEL: str = os.getenv("VOYAGE_EMBED_MODEL", "voyage-4-large")
VOYAGE_RERANK_MODEL: str = os.getenv("VOYAGE_RERANK_MODEL", "rerank-3")
RERANK_BACKEND: str = os.getenv("RERANK_BACKEND", "native").lower()
if RERANK_BACKEND not in {"native", "voyage_api"}:
    raise ValueError("RERANK_BACKEND deve essere native oppure voyage_api")
# Modalità embedding:
#   "auto"    → Atlas autoEmbed (vettori solo nell'indice, no campo embedding nei doc)
#   "manual"  → Python chiama Voyage /embeddings (vettori in campo `embedding`)
#   "context" → Python chiama Voyage /contextualizedembeddings con
#               chunk locali page-aware raggruppati per documento
EMBED_MODE: str = os.getenv("EMBED_MODE", "auto").lower()
VOYAGE_EMBED_DIMENSIONS: int = int(os.getenv("VOYAGE_EMBED_DIMENSIONS", "2048"))
# voyage-context-4: fino a 120K con auto-chunking backend; qui usiamo chunk
# locali per mantenere le citazioni pagina, con limite pre-chunked di 32K.
VOYAGE_CONTEXT_MODEL: str = os.getenv("VOYAGE_CONTEXT_MODEL", "voyage-context-4")
VOYAGE_CONTEXT_DIMENSIONS: int = int(os.getenv("VOYAGE_CONTEXT_DIMENSIONS", "1024"))
VOYAGE_MULTIMODAL_MODEL: str = os.getenv("VOYAGE_MULTIMODAL_MODEL", "voyage-multimodal-3.5")
VOYAGE_MULTIMODAL_DIMENSIONS: int = 1024  # dimensione predefinita dell'endpoint multimodale

LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "anthropic").lower()

ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY") or None
ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
# Modello per le decisioni dell'agente (pianificatore e valutatore). Vuoto = stesso modello della risposta.
ANTHROPIC_AGENT_MODEL: str = os.getenv("ANTHROPIC_AGENT_MODEL") or ANTHROPIC_MODEL
# Gateway compatibile con l'API OpenAI "Responses" (per esempio un gateway aziendale): LLM_PROVIDER=gateway
AI_GATEWAY_BASE_URL: str = os.getenv("AI_GATEWAY_BASE_URL", "").rstrip("/")
AI_GATEWAY_API_KEY: str = os.getenv("AI_GATEWAY_API_KEY", "")
AI_GATEWAY_API_KEY_2: str = os.getenv("AI_GATEWAY_API_KEY_2", "")  # di riserva, se la prima è rifiutata
AI_GATEWAY_MODEL: str = os.getenv("AI_GATEWAY_MODEL", "")
AI_GATEWAY_AGENT_MODEL: str = os.getenv("AI_GATEWAY_AGENT_MODEL") or AI_GATEWAY_MODEL
# OpenAI (stessa API "Responses" del gateway): LLM_PROVIDER=openai
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-5-mini")
OPENAI_AGENT_MODEL: str = os.getenv("OPENAI_AGENT_MODEL") or OPENAI_MODEL
# Provider per le decisioni dell'agente (pianificatore, valutatore). Vuoto = LLM_PROVIDER.
AGENT_LLM_PROVIDER: str = (os.getenv("AGENT_LLM_PROVIDER") or LLM_PROVIDER).lower()

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "mistral-nemo:latest")
# Contesto di Ollama (prompt + risposta, in token). Lo stesso valore per tutte le chiamate:
# se cambia tra una chiamata e l'altra, Ollama ricarica il modello.
OLLAMA_NUM_CTX: int = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
# Quanto resta caricato il modello dopo l'ultima chiamata (evita il caricamento a freddo).
OLLAMA_KEEP_ALIVE: str = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
# Modello Ollama per le decisioni dell'agente. Vuoto = stesso modello della risposta.
OLLAMA_AGENT_MODEL: str = os.getenv("OLLAMA_AGENT_MODEL") or OLLAMA_MODEL

GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY") or None
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

CHUNK_SIZE_TOKENS: int = int(os.getenv("CHUNK_SIZE_TOKENS", "512"))
CHUNK_OVERLAP_TOKENS: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "64"))
INGESTION_DUPLICATE_POLICY: str = os.getenv("INGESTION_DUPLICATE_POLICY", "skip").lower()
INGESTION_PIPELINE_VERSION: str = os.getenv("INGESTION_PIPELINE_VERSION", "v4")
CHUNKER_VERSION: str = os.getenv("CHUNKER_VERSION", "token-window-v2")

RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "50"))
# Ricerca agentica: candidati per ogni ricerca e passaggi riordinati quando si uniscono più ricerche.
AGENT_TOP_K: int = int(os.getenv("AGENT_TOP_K", "20"))
AGENT_UNION_RERANK: int = int(os.getenv("AGENT_UNION_RERANK", "30"))
RETRIEVAL_NUM_CANDIDATES_MULTIPLIER: int = int(
    os.getenv("RETRIEVAL_NUM_CANDIDATES_MULTIPLIER", "10")
)
RETRIEVAL_NUM_CANDIDATES_MIN: int = int(os.getenv("RETRIEVAL_NUM_CANDIDATES_MIN", "100"))
RERANK_TOP_N: int = int(os.getenv("RERANK_TOP_N", "8"))
# Sotto questa soglia il top chunk rerank è considerato "off-topic" → no LLM call
RERANK_RELEVANCE_THRESHOLD: float = float(os.getenv("RERANK_RELEVANCE_THRESHOLD", "0.15"))
# Lunghezza massima della risposta: oltre questo limite la risposta è troncata e viene segnalata.
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1500"))
# Peso del ramo testuale in $rankFusion quando si cerca in più modalità:
#   "balanced" → il testo pesa quanto la somma dei rami vettoriali
#   "equal"    → ogni ramo pesa 1 (comportamento precedente)
# Default "equal": con "balanced" e due modalità il testo pesa 2 e, con $limit dopo la fusione,
# anche il 20° risultato testuale batte il 1° semantico: la semantica sparisce dai candidati
# (misurato su Atlas con la modalità «tutte»: la pagina giusta usciva dai candidati).
HYBRID_TEXT_WEIGHT_MODE: str = os.getenv("HYBRID_TEXT_WEIGHT_MODE", "equal").lower()
if HYBRID_TEXT_WEIGHT_MODE not in {"balanced", "equal"}:
    raise ValueError("HYBRID_TEXT_WEIGHT_MODE deve essere balanced oppure equal")
# Unisce le copie dello stesso passaggio salvate in modalità diverse prima del
# reranking. "false" ripristina il comportamento precedente.
HYBRID_DEDUP: bool = os.getenv("HYBRID_DEDUP", "true").lower() not in {"0", "false", "no"}

def vector_index_name(mode: str) -> str:
    return {
        "auto": VECTOR_INDEX_NAME,
        "manual": VECTOR_INDEX_MANUAL_NAME,
        "context": VECTOR_INDEX_CONTEXT_NAME,
        "multimodal": VECTOR_INDEX_MULTIMODAL_NAME,
    }[mode]

# Cifratura lato client (CSFLE esplicita) della prima domanda di ogni sessione.
# La chiave master locale (96 byte) vive in un file fuori dal progetto: senza file la
# demo funziona lo stesso e salva il campo in chiaro.
CSFLE_MASTER_KEY_PATH: Path = Path(os.getenv(
    "CSFLE_MASTER_KEY_PATH", "~/.config/rag-demo/csfle-master-key.bin")).expanduser()
CSFLE_KEY_VAULT_NAMESPACE: str = os.getenv("CSFLE_KEY_VAULT_NAMESPACE", "encryption.__keyVault")
CSFLE_KEY_ALT_NAME: str = os.getenv("CSFLE_KEY_ALT_NAME", "rag-demo-sessioni")

TESSERACT_CMD: str | None = os.getenv("TESSERACT_CMD") or None
OFFICE_CONVERTER: str | None = os.getenv("OFFICE_CONVERTER") or None


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
