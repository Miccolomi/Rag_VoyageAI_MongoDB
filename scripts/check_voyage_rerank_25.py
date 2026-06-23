"""Verifica che rerank-2.5 sia disponibile sull'endpoint Atlas con la chiave al-.

Confronto: stesso payload su rerank-2 (baseline) e rerank-2.5.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

API_KEY = os.getenv("VOYAGE_API_KEY", "").strip()
API_BASE = os.getenv("VOYAGE_API_BASE", "https://ai.mongodb.com/v1").rstrip("/")
URL = f"{API_BASE}/rerank"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

QUERY = "Qual è l'obiettivo della piattaforma di metering?"
DOCS = [
    "La piattaforma di metering raccoglie misure di energia elettrica in tempo reale.",
    "Le ricette tipiche della cucina napoletana includono la pizza margherita.",
    "Audit e tracciabilità del sistema di metering permettono di verificare l'integrità.",
]


def _call(model: str) -> dict:
    payload = {"query": QUERY, "documents": DOCS, "model": model, "top_k": 3}
    print(f"\n──── {model} ────")
    try:
        with httpx.Client(timeout=30.0) as client:
            r = client.post(URL, json=payload, headers=HEADERS)
        print(f"status: {r.status_code}")
        try:
            body = r.json()
        except Exception:
            body = {"raw": r.text[:300]}
        if r.status_code != 200:
            print(f"body:   {json.dumps(body)[:400]}")
            return {"ok": False, "status": r.status_code, "body": body}
        for item in body.get("data", []):
            idx = item["index"]
            score = item.get("relevance_score")
            preview = DOCS[idx][:60]
            print(f"  rank{item.get('index', '?')}: score={score:.4f} · "
                  f"doc[{idx}] = '{preview}…'")
        return {"ok": True, "status": r.status_code, "body": body}
    except Exception as e:
        print(f"net err: {type(e).__name__}: {e}")
        return {"ok": False, "status": None, "body": str(e)}


def main() -> int:
    if not API_KEY:
        print("ERROR: VOYAGE_API_KEY mancante")
        return 1
    print(f"endpoint: {URL}")
    print(f"key: {API_KEY[:3]}…\n")

    r25 = _call("rerank-2.5")
    r2 = _call("rerank-2")
    r25l = _call("rerank-2.5-lite")

    print("\n" + "=" * 50)
    print(f" rerank-2.5:       {'✅' if r25['ok'] else '❌ ' + str(r25['status'])}")
    print(f" rerank-2:         {'✅' if r2['ok'] else '❌ ' + str(r2['status'])}")
    print(f" rerank-2.5-lite:  {'✅' if r25l['ok'] else '❌ ' + str(r25l['status'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
