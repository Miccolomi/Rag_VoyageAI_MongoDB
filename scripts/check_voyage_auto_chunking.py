"""Verifica l'ipotesi 32K (pre-chunked) vs 120K (auto-chunking).

Test:
- A) ~50K token come SINGOLA stringa + enable_auto_chunking=true → 120K dovrebbe accettare
- B) ~130K token come singola stringa + auto_chunking → al limite 120K dovrebbe fallire
- C) cosa torna in risposta? embeddings + i testi dei chunk creati da Voyage?
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
MODEL = "voyage-context-4"


def _call(payload: dict, label: str) -> dict:
    print(f"\n──── {label} ────")
    try:
        with httpx.Client(timeout=180.0) as client:
            r = client.post(URL, json=payload, headers=HEADERS)
        print(f"status: {r.status_code}")
        try:
            body = r.json()
        except Exception:
            body = {"raw": r.text[:300]}
        if r.status_code != 200:
            print(f"body:   {json.dumps(body)[:400]}")
            return {"ok": False, "status": r.status_code, "body": body}
        return {"ok": True, "status": r.status_code, "body": body}
    except Exception as e:
        print(f"net err: {type(e).__name__}: {e}")
        return {"ok": False, "status": None, "body": str(e)}


def _build_big_string(target_tokens: int) -> tuple[str, int]:
    base = ("La piattaforma di metering raccoglie misure di energia "
            "elettrica in tempo reale da contatori distribuiti sul "
            "territorio nazionale. " * 30)
    per_block = len(TOK.encode(base))
    n = max(1, target_tokens // per_block)
    s = base * n
    return s, per_block * n


def _inspect_response(body: dict) -> None:
    """Quanti chunk ha creato Voyage? Ritorna anche i testi se presenti."""
    print(f"top-level keys: {list(body.keys())}")
    usage = body.get("usage", {})
    print(f"usage: {usage}")
    docs = body.get("data", [])
    for di, d in enumerate(docs):
        inner = d.get("data", [])
        print(f"  doc[{di}]: {len(inner)} chunk · keys per chunk: "
              f"{list(inner[0].keys()) if inner else '[]'}")
        for ci, c in enumerate(inner[:3]):
            keys = {k: type(v).__name__ for k, v in c.items()}
            text_preview = ""
            if "text" in c:
                t = c["text"]
                text_preview = f" · text(len={len(t)}): {t[:80]!r}…"
            print(f"    chunk[{ci}]: keys={keys}{text_preview}")
        if len(inner) > 3:
            print(f"    … e altri {len(inner)-3} chunk")


def main() -> int:
    if not API_KEY:
        print("ERROR: VOYAGE_API_KEY mancante")
        return 1
    print(f"endpoint: {URL}  ·  model: {MODEL}\n")

    # ── A) ~50K SINGLE STRING + auto_chunking ──
    s50, tot50 = _build_big_string(50000)
    print(f"\n=== A) single string ~50K tok (effettivi: {tot50}) + auto_chunking ===")
    payload_a = {
        "inputs": [[s50]],
        "model": MODEL,
        "input_type": "document",
        "enable_auto_chunking": True,
    }
    r_a = _call(payload_a, "A · single 50K + auto_chunking")
    if r_a["ok"]:
        _inspect_response(r_a["body"])

    # ── B) ~130K SINGLE STRING + auto_chunking ──
    s130, tot130 = _build_big_string(130000)
    print(f"\n=== B) single string ~130K tok (effettivi: {tot130}) + auto_chunking ===")
    payload_b = {
        "inputs": [[s130]],
        "model": MODEL,
        "input_type": "document",
        "enable_auto_chunking": True,
    }
    r_b = _call(payload_b, "B · single 130K + auto_chunking")
    if r_b["ok"]:
        _inspect_response(r_b["body"])

    # ── C) Sanity: ~50K single string SENZA auto_chunking (atteso 400) ──
    print(f"\n=== C) sanity: single string ~50K SENZA auto_chunking ===")
    payload_c = {
        "inputs": [[s50]],
        "model": MODEL,
        "input_type": "document",
    }
    r_c = _call(payload_c, "C · single 50K SENZA auto_chunking")

    # ── Summary ──
    print("\n" + "=" * 60)
    print(" SUMMARY")
    print("=" * 60)
    print(f" A) ~50K + auto_chunking:    "
          f"{'✅ sblocca il limite 32K' if r_a['ok'] else '❌ ' + str(r_a['status'])}")
    print(f" B) ~130K + auto_chunking:   "
          f"{'✅ va oltre dichiarato!' if r_b['ok'] else '❌ ' + str(r_b['status']) + ' (limite ~120K?)'}")
    print(f" C) ~50K NO auto_chunking:   "
          f"{'😱 inatteso 200' if r_c['ok'] else '✅ atteso ' + str(r_c['status'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
