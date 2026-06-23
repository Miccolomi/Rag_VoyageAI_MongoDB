"""Generation groundata. Supporta due provider via `LLM_PROVIDER`:

- "anthropic" → Claude via SDK ufficiale (`anthropic`)
- "ollama"    → modello locale via HTTP API Ollama (`/api/chat`)

Esposte due interfacce:
- generate_answer(): chiamata bloccante, ritorna dict con `answer` completo
- generate_answer_stream(): ritorna un iteratore di stringhe (token) compatibile
  con `st.write_stream()` di Streamlit. Le metriche finali sono esposte come
  attributi (.usage, .full_text, .provider, .model) leggibili dopo che il
  consumer ha esaurito l'iteratore.

Il system prompt è identico tra i due provider — impone risposte basate
ESCLUSIVAMENTE sui passaggi forniti.
"""
from __future__ import annotations

import json
import time
from typing import Any, Callable, Iterator

import httpx
from anthropic import Anthropic

from . import config

# Il system prompt vive in `prompts/system_prompt.txt` (override opzionale
# via env LLM_SYSTEM_PROMPT / LLM_SYSTEM_PROMPT_FILE). Caricato in config.
_SYSTEM = config.LLM_SYSTEM_PROMPT

_GEMINI_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_GEMINI_MAX_ATTEMPTS = 4


def _format_context(chunks: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for i, c in enumerate(chunks, start=1):
        head = f"[#{i}] file={c.get('filename')} p.{c.get('page_number')}"
        if c.get("title"):
            head += f" titolo={c['title']!r}"
        if c.get("content_type"):
            head += f" tipo={c['content_type']}"
        blocks.append(f"{head}\n{c.get('content', '')}")
    return "\n\n---\n\n".join(blocks)


def _user_message(query: str, chunks: list[dict[str, Any]]) -> str:
    context = _format_context(chunks) if chunks else "(nessun passaggio recuperato)"
    return (
        f"<domanda>\n{query}\n</domanda>\n\n"
        f"<passaggi>\n{context}\n</passaggi>\n\n"
        "Rispondi seguendo rigorosamente il SYSTEM prompt. "
        "Le citazioni Markdown sono obbligatorie per ogni affermazione supportata."
    )


def _generate_anthropic(query: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    if not config.ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY non impostata nel .env")
    client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=1024,
        system=_SYSTEM,
        messages=[{"role": "user", "content": _user_message(query, chunks)}],
    )
    text = "".join(
        b.text for b in resp.content if getattr(b, "type", None) == "text"
    )
    return {
        "answer": text,
        "provider": "anthropic",
        "model": config.ANTHROPIC_MODEL,
        "usage": {
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
        },
    }


def _generate_gemini(query: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    if not config.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY non impostata nel .env")
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{config.GEMINI_MODEL}:generateContent")
    payload = {
        "system_instruction": {"parts": [{"text": _SYSTEM}]},
        "contents": [{"role": "user",
                      "parts": [{"text": _user_message(query, chunks)}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 1024},
    }
    # API key in header (NON in URL → non appare nei log/errori)
    headers = {"x-goog-api-key": config.GEMINI_API_KEY}
    with httpx.Client(timeout=120.0) as client:
        for attempt in range(1, _GEMINI_MAX_ATTEMPTS + 1):
            try:
                r = client.post(url, headers=headers, json=payload)
            except httpx.TransportError as exc:
                if attempt == _GEMINI_MAX_ATTEMPTS:
                    raise RuntimeError(
                        "Gemini non è raggiungibile dopo diversi tentativi. "
                        "Controlla la connessione e riprova tra poco."
                    ) from exc
                time.sleep(2 ** (attempt - 1))
                continue

            if r.status_code not in _GEMINI_RETRYABLE_STATUS:
                r.raise_for_status()
                data = r.json()
                break

            if attempt == _GEMINI_MAX_ATTEMPTS:
                if r.status_code == 429:
                    raise RuntimeError(
                        "Gemini ha raggiunto il limite di richieste (429). "
                        "Attendi qualche minuto e riprova."
                    )
                raise RuntimeError(
                    f"Gemini è temporaneamente non disponibile ({r.status_code}) "
                    "anche dopo diversi tentativi. Riprova tra poco."
                )

            retry_after = r.headers.get("retry-after")
            try:
                delay = float(retry_after) if retry_after else 2 ** (attempt - 1)
            except ValueError:
                delay = 2 ** (attempt - 1)
            time.sleep(min(max(delay, 1.0), 10.0))
    text = "".join(
        part.get("text", "")
        for c in data.get("candidates", [])
        for part in c.get("content", {}).get("parts", [])
    )
    usage = data.get("usageMetadata", {})
    return {
        "answer": text,
        "provider": "gemini",
        "model": config.GEMINI_MODEL,
        "usage": {
            "input_tokens": usage.get("promptTokenCount"),
            "output_tokens": usage.get("candidatesTokenCount"),
        },
    }


def _generate_ollama(query: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    url = f"{config.OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": config.OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": _user_message(query, chunks)},
        ],
        "options": {
            "temperature": 0.1,  # molto bassa per massimizzare la fedeltà ai passaggi
            "top_p": 0.9,
            "num_ctx": 8192,     # context window per chunk + prompt
        },
    }
    with httpx.Client(timeout=300.0) as client:
        try:
            r = client.post(url, json=payload)
        except httpx.ConnectError as e:
            raise RuntimeError(
                f"Impossibile contattare Ollama su {config.OLLAMA_BASE_URL}. "
                "Verifica che il server sia avviato (`ollama serve`)."
            ) from e
        r.raise_for_status()
        data = r.json()
    text = (data.get("message") or {}).get("content", "")
    return {
        "answer": text,
        "provider": "ollama",
        "model": config.OLLAMA_MODEL,
        "usage": {
            "input_tokens": data.get("prompt_eval_count"),
            "output_tokens": data.get("eval_count"),
        },
    }


def generate_answer(query: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Versione bloccante. Ritorna dict completo (answer + usage + provider + model)."""
    if config.LLM_PROVIDER == "ollama":
        return _generate_ollama(query, chunks)
    if config.LLM_PROVIDER == "anthropic":
        return _generate_anthropic(query, chunks)
    if config.LLM_PROVIDER == "gemini":
        return _generate_gemini(query, chunks)
    raise ValueError(
        f"LLM_PROVIDER non valido: '{config.LLM_PROVIDER}' "
        f"(attesi: anthropic | ollama | gemini)"
    )


# ────────────────────────────────────────────────────── streaming interface ──

class AnswerStream:
    """Iteratore di token streaming compatibile con `st.write_stream()`.

    Yields stringhe (token man mano che arrivano dal LLM). Al termine
    dell'iterazione, gli attributi `.full_text`, `.usage`, `.provider`, `.model`
    sono valorizzati e leggibili.
    """
    def __init__(self, gen_factory: Callable[[], Iterator], *, provider: str, model: str):
        self._gen_factory = gen_factory
        self.provider = provider
        self.model = model
        self.usage: dict = {"input_tokens": None, "output_tokens": None}
        self.full_text = ""

    def __iter__(self):
        parts: list[str] = []
        for item in self._gen_factory():
            if isinstance(item, dict):
                # Metadata finale (usage) — non yieldare al consumer Streamlit
                self.usage = item
            else:
                parts.append(item)
                yield item
        self.full_text = "".join(parts)


def _stream_anthropic(query: str, chunks: list[dict[str, Any]]) -> Iterator:
    if not config.ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY non impostata nel .env")
    client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    with client.messages.stream(
        model=config.ANTHROPIC_MODEL,
        max_tokens=1024,
        system=_SYSTEM,
        messages=[{"role": "user", "content": _user_message(query, chunks)}],
    ) as stream:
        for text_chunk in stream.text_stream:
            if text_chunk:
                yield text_chunk
        final = stream.get_final_message()
        yield {
            "input_tokens": final.usage.input_tokens,
            "output_tokens": final.usage.output_tokens,
        }


def _stream_gemini(query: str, chunks: list[dict[str, Any]]) -> Iterator:
    # Per la demo preferiamo stabilità a streaming token-by-token: l'endpoint
    # SSE di Gemini può rendere risposte parziali difficili da distinguere.
    result = _generate_gemini(query, chunks)
    yield result["answer"]
    yield result["usage"]


def _stream_ollama(query: str, chunks: list[dict[str, Any]]) -> Iterator:
    url = f"{config.OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": config.OLLAMA_MODEL,
        "stream": True,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": _user_message(query, chunks)},
        ],
        "options": {"temperature": 0.1, "top_p": 0.9, "num_ctx": 8192},
    }
    try:
        with httpx.stream("POST", url, json=payload, timeout=300.0) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                msg = data.get("message") or {}
                token = msg.get("content", "")
                if token:
                    yield token
                if data.get("done"):
                    yield {
                        "input_tokens": data.get("prompt_eval_count"),
                        "output_tokens": data.get("eval_count"),
                    }
                    break
    except httpx.ConnectError as e:
        raise RuntimeError(
            f"Impossibile contattare Ollama su {config.OLLAMA_BASE_URL}. "
            "Verifica che il server sia avviato (`ollama serve`)."
        ) from e


def generate_answer_stream(query: str, chunks: list[dict[str, Any]]) -> AnswerStream:
    """Versione streaming. Ritorna AnswerStream (iterabile) per `st.write_stream()`."""
    if config.LLM_PROVIDER == "ollama":
        return AnswerStream(
            lambda: _stream_ollama(query, chunks),
            provider="ollama", model=config.OLLAMA_MODEL,
        )
    if config.LLM_PROVIDER == "anthropic":
        return AnswerStream(
            lambda: _stream_anthropic(query, chunks),
            provider="anthropic", model=config.ANTHROPIC_MODEL,
        )
    if config.LLM_PROVIDER == "gemini":
        return AnswerStream(
            lambda: _stream_gemini(query, chunks),
            provider="gemini", model=config.GEMINI_MODEL,
        )
    raise ValueError(
        f"LLM_PROVIDER non valido: '{config.LLM_PROVIDER}' "
        f"(attesi: anthropic | ollama | gemini)"
    )


def llm_available() -> bool:
    """True se il provider configurato è utilizzabile (chiavi/server presenti)."""
    if config.LLM_PROVIDER == "anthropic":
        return bool(config.ANTHROPIC_API_KEY)
    if config.LLM_PROVIDER == "gemini":
        return bool(config.GEMINI_API_KEY)
    if config.LLM_PROVIDER == "ollama":
        try:
            with httpx.Client(timeout=2.0) as client:
                return client.get(f"{config.OLLAMA_BASE_URL}/api/tags").status_code == 200
        except Exception:
            return False
    return False
