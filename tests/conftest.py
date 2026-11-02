"""Configurazione dei test offline: nessuna connessione ad Atlas, Voyage o LLM.

- MONGODB_URI fittizio, così `src.config` si importa senza .env reale.
- tiktoken può dover scaricare il vocabolario: se non è disponibile in locale
  viene sostituito da un tokenizer a parole, sufficiente per testare il chunking.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MONGODB_URI", "mongodb://127.0.0.1:1/?serverSelectionTimeoutMS=200")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-non-usata")
os.environ.setdefault("VOYAGE_API_KEY", "test-key-non-usata")
os.environ.setdefault("LLM_PROVIDER", "anthropic")
os.environ.setdefault("RERANK_BACKEND", "native")
os.environ.setdefault("HYBRID_TEXT_WEIGHT_MODE", "balanced")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tiktoken  # noqa: E402


class _WordEncoding:
    """Tokenizer a parole: un token per parola, decodifica con spazi."""

    def __init__(self) -> None:
        self._ids: dict[str, int] = {}
        self._words: list[str] = []

    def encode(self, text: str) -> list[int]:
        out = []
        for word in text.split():
            if word not in self._ids:
                self._ids[word] = len(self._words)
                self._words.append(word)
            out.append(self._ids[word])
        return out

    def decode(self, tokens: list[int]) -> str:
        return " ".join(self._words[token] for token in tokens)


try:
    tiktoken.get_encoding("cl100k_base")
except Exception:
    _fake = _WordEncoding()
    tiktoken.get_encoding = lambda name: _fake  # type: ignore[assignment]
