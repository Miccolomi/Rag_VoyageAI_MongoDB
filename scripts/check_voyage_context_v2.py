"""Verifica differenze context-3 vs context-4 e limite effettivo aggiornato.

Test:
1. context-4 funziona con la nostra chiave al- sull'endpoint Atlas?
2. context-3 limite reale ora: 32K (test del 2026-06-09) o 120K (doc 2026-06-11)?
3. context-4 limite reale: 120K dichiarato?
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import tiktoken
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

API_KEY = os.getenv("VOYAGE_API_KEY", "").strip()
API_BASE = os.getenv("VOYAGE_API_BASE", "https://ai.mongodb.com/v1").rstrip("/")
URL = f"{API_BASE}/contextualizedembeddings"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
TOK = tiktoken.get_encoding("cl100k_base")


def _call(payload: dict, label: str) -> dict:
    print(f"\n──── {label} ────")
    try:
        with httpx.Client(timeout=120.0) as client:
            r = client.post(URL, json=payload, headers=HEADERS)
        print(f"status: {r.status_code}")
        body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"raw": r.text[:300]}
        if r.status_code != 200:
            print(f"body:   {json.dumps(body)[:400]}")
            return {"ok": False, "status": r.status_code, "body": body}
        # parse nested data
        embeddings_count = sum(
            len(d.get("data", [])) for d in body.get("data", [])
        )
        usage = body.get("usage", {})
        first_dim = None
        for d in body.get("data", []):
            for c in d.get("data", []):
                first_dim = len(c["embedding"])
                break
            if first_dim:
                break
        print(f"chunk ricevuti: {embeddings_count} · dim: {first_dim} · usage: {usage}")
        return {"ok": True, "status": r.status_code, "body": body}
    except Exception as e:
        print(f"net err: {type(e).__name__}: {e}")
        return {"ok": False, "status": None, "body": str(e)}


def _make_payload(model: str, target_tokens: int) -> tuple[dict, int]:
    """Costruisce un payload con ~target_tokens token totali per modello."""
    chunk = ("La piattaforma di metering raccoglie misure di energia "
             "elettrica in tempo reale da contatori distribuiti sul "
             "territorio nazionale. " * 30)
    per_chunk = len(TOK.encode(chunk))
    n = max(1, target_tokens // per_chunk)
    chunks = [chunk] * n
    total = per_chunk * n
    return {
        "inputs": [chunks],
        "model": model,
        "input_type": "document",
    }, total


def main() -> int:
    if not API_KEY:
        print("ERROR: VOYAGE_API_KEY mancante")
        return 1
    print(f"endpoint: {URL}")
    print(f"key kind: {API_KEY[:3]}…\n")

    # ── Q1: context-4 funziona con la nostra chiave? ──
    small = {
        "inputs": [["Capitolo 1: introduzione.",
                    "Capitolo 2: dettagli.",
                    "Capitolo 3: conclusioni."]],
        "model": "voyage-context-4",
        "input_type": "document",
    }
    r_c4 = _call(small, "Q1 · voyage-context-4 chiamata base")

    # Stesso payload con context-3 per confronto
    small3 = dict(small, model="voyage-context-3")
    r_c3 = _call(small3, "Q1bis · voyage-context-3 chiamata base (sanity check)")

    # ── Q2: context-3 con payload ~50K token (tra il vecchio limite 32K e nuovo 120K) ──
    p50_3, tot_50_3 = _make_payload("voyage-context-3", 50000)
    print(f"\n(payload context-3 target ~50K → effettivi ~{tot_50_3} tok)")
    r_c3_50 = _call(p50_3, "Q2 · context-3 a ~50K token")

    # ── Q3: context-4 con payload ~50K token ──
    p50_4, tot_50_4 = _make_payload("voyage-context-4", 50000)
    print(f"\n(payload context-4 target ~50K → effettivi ~{tot_50_4} tok)")
    r_c4_50 = _call(p50_4, "Q3 · context-4 a ~50K token")

    # ── Q4: context-3 con payload ~130K token (oltre nuovo limite dichiarato) ──
    p130_3, tot_130_3 = _make_payload("voyage-context-3", 130000)
    print(f"\n(payload context-3 target ~130K → effettivi ~{tot_130_3} tok)")
    r_c3_130 = _call(p130_3, "Q4 · context-3 a ~130K token (oltre dichiarato)")

    # ── Summary ──
    print("\n" + "=" * 60)
    print(" SUMMARY")
    print("=" * 60)
    print(f" context-4 base:           {'✅' if r_c4['ok'] else '❌ ' + str(r_c4['status'])}")
    print(f" context-3 base:           {'✅' if r_c3['ok'] else '❌ ' + str(r_c3['status'])}")
    print(f" context-3 a ~50K tok:     {'✅' if r_c3_50['ok'] else '❌ ' + str(r_c3_50['status'])}")
    print(f" context-4 a ~50K tok:     {'✅' if r_c4_50['ok'] else '❌ ' + str(r_c4_50['status'])}")
    print(f" context-3 a ~130K tok:    {'✅' if r_c3_130['ok'] else '❌ ' + str(r_c3_130['status'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
