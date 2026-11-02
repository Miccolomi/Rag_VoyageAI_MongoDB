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
import re
import time
from typing import Any, Callable, Iterator

import httpx
from anthropic import Anthropic

from . import config

# Il system prompt vive in `prompts/system_prompt.txt` (override opzionale
# via env LLM_SYSTEM_PROMPT / LLM_SYSTEM_PROMPT_FILE). Caricato in config.
def _system_prompt() -> str:
    return config._load_system_prompt()

_GEMINI_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_GEMINI_MAX_ATTEMPTS = 4


# Indirizzi web nei documenti (es. "www.3m.com" in un 10-K): vengono tolti dai
# passaggi prima di inviarli al modello, così non può riportarli nella risposta.
_WEB_IN_PASSAGE = re.compile(r"(?:https?://|ftp://|www\.)[^\s<>\"')\]]+|mailto:[^\s<>\"')\]]+", re.I)
WEB_PLACEHOLDER = "[indirizzo web omesso]"


def strip_web_addresses(text: str) -> str:
    return _WEB_IN_PASSAGE.sub(WEB_PLACEHOLDER, text or "")


def _format_context(chunks: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for i, c in enumerate(chunks, start=1):
        head = f"[#{i}] file={c.get('filename')}"
        if c.get("page_number") is not None:
            head += f" p.{c['page_number']}"
        else:
            head += " (riferimento al documento; pagina non disponibile)"
        if c.get("title"):
            head += f" titolo={c['title']!r}"
        if c.get("content_type"):
            head += f" tipo={c['content_type']}"
        blocks.append(f"{head}\n{strip_web_addresses(str(c.get('content', '')))}")
    return "\n\n---\n\n".join(blocks)


def _user_message(query: str, chunks: list[dict[str, Any]]) -> str:
    context = _format_context(chunks) if chunks else "(nessun passaggio recuperato)"
    return (
        f"DOMANDA:\n{query}\n\nPASSAGGI:\n{context}\n\n"
        "Rispondi alla domanda con le fonti che la supportano."
    )



def _generate_anthropic(query: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    if not config.ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY non impostata nel .env")
    client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    system_text = _system_prompt()
    user_text = _user_message(query, chunks)
    resp = client.messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=config.LLM_MAX_TOKENS,
        temperature=0,
        system=system_text,
        messages=[{"role": "user", "content": user_text}],
    )
    text = "".join(
        b.text for b in resp.content if getattr(b, "type", None) == "text"
    )
    return {
        "answer": text,
        "truncated": getattr(resp, "stop_reason", None) == "max_tokens",
        "stop_reason": getattr(resp, "stop_reason", None),
        "provider": "anthropic",
        "request_messages": [{"role": "system", "content": system_text}, {"role": "user", "content": user_text}],
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
        "system_instruction": {"parts": [{"text": _system_prompt()}]},
        "contents": [{"role": "user",
                      "parts": [{"text": _user_message(query, chunks)}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": config.LLM_MAX_TOKENS},
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
    finish = next((c.get("finishReason") for c in data.get("candidates", []) if c.get("finishReason")), None)
    return {
        "answer": text,
        "truncated": finish == "MAX_TOKENS",
        "stop_reason": finish,
        "provider": "gemini",
        "request_messages": [
            {"role": "system", "content": payload["system_instruction"]["parts"][0]["text"]},
            {"role": "user", "content": payload["contents"][0]["parts"][0]["text"]},
        ],
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
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": _user_message(query, chunks)},
        ],
        "keep_alive": config.OLLAMA_KEEP_ALIVE,
        "options": {
            "temperature": 0,  # stessa domanda, stessi passaggi: stessa risposta
            "top_p": 0.9,
            "num_ctx": config.OLLAMA_NUM_CTX,  # stesso valore delle decisioni dell'agente: niente ricarichi
            "num_predict": config.LLM_MAX_TOKENS,
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
        "truncated": data.get("done_reason") == "length",
        "stop_reason": data.get("done_reason"),
        "provider": "ollama",
        "request_messages": payload["messages"],
        "model": config.OLLAMA_MODEL,
        "usage": {
            "input_tokens": data.get("prompt_eval_count"),
            "output_tokens": data.get("eval_count"),
        },
    }


def _generate_gateway(query: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    from . import ai_gateway
    system_text = _system_prompt()
    user_text = _user_message(query, chunks)
    result = ai_gateway.respond(system_text, user_text, model=config.AI_GATEWAY_MODEL,
                                max_tokens=config.LLM_MAX_TOKENS)
    return {
        **result,
        "provider": "gateway",
        "request_messages": [{"role": "system", "content": system_text}, {"role": "user", "content": user_text}],
    }


def generate_answer(query: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Versione bloccante. Ritorna dict completo (answer + usage + provider + model)."""
    if config.LLM_PROVIDER == "gateway":
        return _generate_gateway(query, chunks)
    if config.LLM_PROVIDER == "ollama":
        return _generate_ollama(query, chunks)
    if config.LLM_PROVIDER == "anthropic":
        return _generate_anthropic(query, chunks)
    if config.LLM_PROVIDER == "gemini":
        return _generate_gemini(query, chunks)
    raise ValueError(
        f"LLM_PROVIDER non valido: '{config.LLM_PROVIDER}' "
        f"(attesi: anthropic | ollama | gemini | gateway)"
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
        max_tokens=config.LLM_MAX_TOKENS,
        temperature=0,
        system=_system_prompt(),
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
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": _user_message(query, chunks)},
        ],
        "options": {"temperature": 0, "top_p": 0.9, "num_ctx": config.OLLAMA_NUM_CTX,
                    "num_predict": config.LLM_MAX_TOKENS},
        "keep_alive": config.OLLAMA_KEEP_ALIVE,
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


def _stream_gateway(query: str, chunks: list[dict[str, Any]]) -> Iterator:
    result = _generate_gateway(query, chunks)
    yield result["answer"]
    yield result["usage"]


def generate_answer_stream(query: str, chunks: list[dict[str, Any]]) -> AnswerStream:
    """Versione streaming. Ritorna AnswerStream (iterabile) per `st.write_stream()`."""
    if config.LLM_PROVIDER == "gateway":
        return AnswerStream(lambda: _stream_gateway(query, chunks), provider="gateway", model=config.AI_GATEWAY_MODEL)
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
        f"(attesi: anthropic | ollama | gemini | gateway)"
    )


def llm_available() -> bool:
    """True se il provider configurato è utilizzabile (chiavi/server presenti)."""
    if config.LLM_PROVIDER == "anthropic":
        return bool(config.ANTHROPIC_API_KEY)
    if config.LLM_PROVIDER == "gateway":
        return bool(config.AI_GATEWAY_API_KEY or config.AI_GATEWAY_API_KEY_2)
    if config.LLM_PROVIDER == "gemini":
        return bool(config.GEMINI_API_KEY)
    if config.LLM_PROVIDER == "ollama":
        try:
            with httpx.Client(timeout=2.0) as client:
                return client.get(f"{config.OLLAMA_BASE_URL}/api/tags").status_code == 200
        except Exception:
            return False
    return False
