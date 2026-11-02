"""Client per l'AI Gateway aziendale (API OpenAI "Responses").

Endpoint: {AI_GATEWAY_BASE_URL}/responses, autenticazione Bearer.
Il gateway può esporre modelli con o senza "reasoning": alcuni parametri
(temperature, reasoning, formato JSON) non sono accettati da tutti i modelli.
Se il gateway rifiuta un parametro con un 400, la richiesta viene ripetuta
senza quel parametro; l'esito viene ricordato per le chiamate successive.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

from . import config

log = logging.getLogger(__name__)

# Parametri che il modello ha già rifiutato in questa esecuzione: non li si rimanda.
_UNSUPPORTED: set[str] = set()


def _keys() -> list[str]:
    return [key for key in (config.AI_GATEWAY_API_KEY, config.AI_GATEWAY_API_KEY_2) if key]


def available() -> bool:
    return bool(_keys())


def _text_from(data: dict[str, Any]) -> str:
    if isinstance(data.get("output_text"), str):
        return data["output_text"]
    parts = []
    for item in data.get("output") or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if content.get("type") in {"output_text", "text"} and isinstance(content.get("text"), str):
                parts.append(content["text"])
    return "".join(parts)


def _rejected_parameter(response: httpx.Response, candidates: list[str]) -> str | None:
    """Il parametro opzionale che il gateway ha rifiutato, dal campo `param` o dal messaggio."""
    try:
        error = (response.json() or {}).get("error") or {}
    except ValueError:
        error = {}
    param = str(error.get("param") or "")
    message = str(error.get("message") or response.text)
    for name in candidates:
        if param.split(".")[0] == name or f"'{name}" in message or f'"{name}' in message:
            return name
    return None


def respond(system: str, user: str, *, model: str, max_tokens: int, json_mode: bool = False,
            timeout: float = 120.0) -> dict[str, Any]:
    """Una chiamata al modello. Restituisce answer, truncated, stop_reason, usage, model."""
    keys = _keys()
    if not keys:
        raise RuntimeError("AI_GATEWAY_API_KEY non impostata nel .env")
    optional: dict[str, Any] = {
        "temperature": 0,
        # Con i modelli "reasoning" il ragionamento consuma token di uscita: lo si tiene breve.
        "reasoning": {"effort": "low"},
    }
    if json_mode:
        optional["text"] = {"format": {"type": "json_object"}}
    url = f"{config.AI_GATEWAY_BASE_URL}/responses"
    last_error = ""
    for key in keys:
        for _ in range(len(optional) + 1):
            payload: dict[str, Any] = {
                "model": model,
                "instructions": system,
                "input": [{"role": "user", "content": user}],
                # Margine per l'eventuale ragionamento interno del modello.
                "max_output_tokens": max(max_tokens * 3, 1500),
                **{name: value for name, value in optional.items() if name not in _UNSUPPORTED},
            }
            response = httpx.post(url, json=payload, timeout=timeout,
                                  headers={"Authorization": f"Bearer {key}"})
            if response.status_code == 400:
                rejected = _rejected_parameter(response, [n for n in optional if n not in _UNSUPPORTED])
                if rejected:
                    log.info("Il modello %s non accetta il parametro %s: lo tolgo", model, rejected)
                    _UNSUPPORTED.add(rejected)
                    continue
            if response.status_code in {401, 403, 429} and key != keys[-1]:
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                break  # prova con la seconda chiave
            if response.status_code == 403 and "RBAC" in response.text:
                raise RuntimeError("AI Gateway: accesso negato dalla rete (RBAC). Serve la VPN aziendale attiva.")
            response.raise_for_status()
            data = response.json()
            details = data.get("incomplete_details") or {}
            usage = data.get("usage") or {}
            return {
                "answer": _text_from(data),
                "truncated": data.get("status") == "incomplete" and details.get("reason") == "max_output_tokens",
                "stop_reason": details.get("reason") or data.get("status"),
                "model": data.get("model") or model,
                "usage": {"input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens")},
            }
    raise RuntimeError(f"AI Gateway non disponibile con le chiavi configurate ({last_error})")
