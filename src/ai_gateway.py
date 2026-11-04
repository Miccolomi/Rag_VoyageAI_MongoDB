"""Client per l'API OpenAI "Responses": AI Gateway aziendale (provider gateway) oppure OpenAI (provider openai).

Endpoint: {base}/responses, autenticazione Bearer.
Il gateway può esporre modelli con o senza "reasoning": alcuni parametri
(temperature, reasoning, formato JSON) non sono accettati da tutti i modelli.
Se il gateway rifiuta un parametro con un 400, la richiesta viene ripetuta
senza quel parametro; l'esito viene ricordato per le chiamate successive.
"""
from __future__ import annotations

import logging
import re
from typing import Any

import httpx

from . import config

log = logging.getLogger(__name__)

# Parametri che il modello ha già rifiutato in questa esecuzione: non li si rimanda.
_UNSUPPORTED: set[str] = set()


def _endpoint(provider: str = "gateway") -> tuple[str, list[str], str, str]:
    """Indirizzo, chiavi, nome mostrato e variabile della chiave del provider."""
    if provider == "openai":
        return config.OPENAI_BASE_URL, [key for key in (config.OPENAI_API_KEY,) if key], "OpenAI", "OPENAI_API_KEY"
    return (config.AI_GATEWAY_BASE_URL, [key for key in (config.AI_GATEWAY_API_KEY, config.AI_GATEWAY_API_KEY_2) if key],
            "AI Gateway", "AI_GATEWAY_API_KEY")


def _keys(provider: str = "gateway") -> list[str]:
    return _endpoint(provider)[1]


def available(provider: str = "gateway") -> bool:
    return bool(_keys(provider))


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


def _error(response: httpx.Response) -> dict[str, Any]:
    try:
        error = (response.json() or {}).get("error") or {}
    except (ValueError, AttributeError):
        error = {}
    return error if isinstance(error, dict) else {"message": str(error)}


def _error_message(response: httpx.Response) -> str:
    """Il motivo scritto dal gateway, per mostrarlo invece del solo codice HTTP."""
    message = str(_error(response).get("message") or response.text or "nessun dettaglio")
    return " ".join(message.split())[:300]


def _rejected_parameter(response: httpx.Response, candidates: list[str]) -> str | None:
    """Il parametro opzionale che il gateway ha rifiutato, dal campo `param` o dal messaggio."""
    error = _error(response)
    param = str(error.get("param") or "")
    message = str(error.get("message") or response.text)
    for name in candidates:
        quoted = f"'{name}" in message or f'"{name}' in message
        # "reasoning.effort is not supported": nome citato senza virgolette (non per "text", parola troppo comune).
        bare = name != "text" and re.search(rf"\b{name}\b", message, re.I)
        if param.split(".")[0] == name or quoted or bare:
            return name
    return None


def respond(system: str, user: str, *, model: str, max_tokens: int, json_mode: bool = False,
            timeout: float = 120.0, provider: str = "gateway") -> dict[str, Any]:
    """Una chiamata al modello. Restituisce answer, truncated, stop_reason, usage, model."""
    base_url, keys, name, key_var = _endpoint(provider)
    if not keys:
        raise RuntimeError(f"{key_var} non impostata nel .env")
    optional: dict[str, Any] = {
        "temperature": 0,
        # Con i modelli "reasoning" il ragionamento consuma token di uscita: lo si tiene breve.
        "reasoning": {"effort": "low"},
    }
    if json_mode:
        optional["text"] = {"format": {"type": "json_object"}}
    url = f"{base_url}/responses"
    last_error = ""
    for key in keys:
        minimal = False  # ultimo tentativo: nessun parametro facoltativo
        for _ in range(len(optional) + 2):
            active = {} if minimal else {name: value for name, value in optional.items() if name not in _UNSUPPORTED}
            payload: dict[str, Any] = {
                "model": model,
                "instructions": system,
                "input": [{"role": "user", "content": user}],
                # Margine per l'eventuale ragionamento interno del modello.
                "max_output_tokens": max(max_tokens * 3, 1500),
                **active,
            }
            response = httpx.post(url, json=payload, timeout=timeout,
                                  headers={"Authorization": f"Bearer {key}"})
            if response.status_code == 400:
                rejected = _rejected_parameter(response, list(active))
                if rejected:
                    log.info("Il modello %s non accetta il parametro %s: lo tolgo", model, rejected)
                    _UNSUPPORTED.add(rejected)
                    continue
                if active:
                    # 400 senza indicare il parametro: si riprova una volta con la richiesta minima.
                    log.warning("%s 400 (%s): riprovo senza parametri facoltativi", name, _error_message(response))
                    minimal = True
                    continue
            if response.status_code in {401, 403, 429} and key != keys[-1]:
                last_error = f"HTTP {response.status_code}: {_error_message(response)}"
                break  # prova con la seconda chiave
            if provider == "gateway" and response.status_code == 403 and "RBAC" in response.text:
                raise RuntimeError("AI Gateway: accesso negato dalla rete (RBAC). Serve la VPN aziendale attiva.")
            if response.status_code >= 400:
                raise RuntimeError(f"{name} ha rifiutato la richiesta (HTTP {response.status_code}, "
                                   f"modello {model}): {_error_message(response)}")
            if minimal:
                # La richiesta minima è passata: quei parametri non si rimandano più a questo gateway.
                _UNSUPPORTED.update(optional)
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
    raise RuntimeError(f"{name} non disponibile con le chiavi configurate ({last_error})")
