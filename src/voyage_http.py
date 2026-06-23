"""HTTP condiviso per le API Voyage AI con retry sugli errori temporanei."""
from __future__ import annotations

import time
from typing import Any

import httpx

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_MAX_ATTEMPTS = 4


def post_json(
    url: str,
    *,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
    operation: str,
) -> dict[str, Any]:
    """Esegue una POST Voyage con backoff e restituisce la risposta JSON."""
    with httpx.Client(timeout=timeout) as client:
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                response = client.post(url, json=payload, headers=headers)
            except httpx.TransportError as exc:
                if attempt == _MAX_ATTEMPTS:
                    raise RuntimeError(
                        f"Voyage AI non raggiungibile durante {operation} "
                        "dopo diversi tentativi."
                    ) from exc
                time.sleep(2 ** (attempt - 1))
                continue

            if response.status_code not in _RETRYABLE_STATUS:
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    detail = response.text[:500].strip()
                    raise RuntimeError(
                        f"Voyage AI ha rifiutato {operation} "
                        f"({response.status_code}): {detail or 'nessun dettaglio'}"
                    ) from exc
                return response.json()

            if attempt == _MAX_ATTEMPTS:
                raise RuntimeError(
                    f"Voyage AI temporaneamente non disponibile durante {operation} "
                    f"({response.status_code}) dopo {_MAX_ATTEMPTS} tentativi."
                )

            retry_after = response.headers.get("retry-after")
            try:
                delay = float(retry_after) if retry_after else 2 ** (attempt - 1)
            except ValueError:
                delay = 2 ** (attempt - 1)
            time.sleep(min(max(delay, 1.0), 10.0))

    raise RuntimeError(f"Voyage AI: risposta non disponibile durante {operation}.")
