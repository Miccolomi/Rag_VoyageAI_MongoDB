"""Step 1 — verifica preliminare per integrare voyage-context-3.

Risponde a 3 domande prima di scrivere codice:
1. La chiave VOYAGE_API_KEY corrente copre l'endpoint /contextualizedembeddings
   sul VOYAGE_API_BASE configurato?
2. Quale dimensione vettoriale torna di default?
3. Come si comporta l'endpoint su payload sovradimensionato?

NIENTE codice di produzione: è uno script one-shot da lanciare a mano.
Uso:  python scripts/check_voyage_context.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

API_KEY = os.getenv("VOYAGE_API_KEY", "").strip()
API_BASE = os.getenv("VOYAGE_API_BASE", "https://ai.mongodb.com/v1").rstrip("/")
MODEL = "voyage-context-3"

# Endpoint alternativo Voyage diretto (chiavi pa-…)
VOYAGE_DIRECT = "https://api.voyageai.com/v1"


def _short(s: str, n: int = 200) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[:n] + "…"


def _call(base: str, payload: dict, *, label: str) -> dict:
    url = f"{base}/contextualizedembeddings"
    print(f"\n──── {label} ────")
    print(f"POST {url}")
    print(f"payload: {json.dumps(payload)[:200]}…")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    try:
        with httpx.Client(timeout=30.0) as client:
            r = client.post(url, json=payload, headers=headers)
        print(f"status:  {r.status_code}")
        try:
            data = r.json()
        except Exception:
            data = {"raw": r.text[:500]}
        if r.status_code != 200:
            print(f"body:    {_short(json.dumps(data))}")
            return {"ok": False, "status": r.status_code, "body": data}
        return {"ok": True, "status": r.status_code, "body": data}
    except Exception as e:
        print(f"network error: {type(e).__name__}: {e}")
        return {"ok": False, "status": None, "body": str(e)}


def main() -> int:
    if not API_KEY:
        print("ERROR: VOYAGE_API_KEY non impostata in .env")
        return 1

    key_kind = "al- (Atlas)" if API_KEY.startswith("al-") else (
        "pa- (Voyage diretto)" if API_KEY.startswith("pa-") else "sconosciuto"
    )
    print("==========================================================")
    print(f" VOYAGE_API_KEY  = ({key_kind}) {API_KEY[:6]}…")
    print(f" VOYAGE_API_BASE = {API_BASE}")
    print(f" MODEL           = {MODEL}")
    print("==========================================================")

    # ── Q1: l'endpoint configurato accetta la nostra chiave + voyage-context-3? ──
    payload_basic = {
        "inputs": [["Capitolo 1: introduzione al sistema.",
                    "Capitolo 2: dettagli implementativi e algoritmi.",
                    "Capitolo 3: conclusioni e riferimenti."]],
        "model": MODEL,
        "input_type": "document",
    }
    r1 = _call(API_BASE, payload_basic,
               label=f"Q1 · endpoint configurato ({API_BASE})")

    # Se Q1 fallisce, prova endpoint Voyage diretto come fallback informativo
    r1_alt = None
    if not r1["ok"] and API_BASE != VOYAGE_DIRECT:
        r1_alt = _call(VOYAGE_DIRECT, payload_basic,
                       label=f"Q1bis · fallback endpoint Voyage diretto ({VOYAGE_DIRECT})")

    # ── Q2: quale dimensione vettoriale torna? ──
    print("\n──── Q2 · dimensioni vettore ────")
    succeeded = r1 if r1["ok"] else (r1_alt if r1_alt and r1_alt["ok"] else None)
    if not succeeded:
        print("⚠ nessuna chiamata è riuscita → impossibile leggere le dimensioni")
    else:
        body = succeeded["body"]
        # Voyage SDK ritorna: {results:[{embeddings:[[...], [...], ...]}]}
        # API REST ritorna shape simile, da scoprire:
        print(f"keys top-level body: {list(body.keys())}")
        # Struttura voyage-context-3 (nested): data[doc].data[chunk].embedding
        dims_found: list[int] = []
        try:
            for doc_obj in body.get("data", []):
                # primo livello: lista di documenti
                inner = doc_obj.get("data") if isinstance(doc_obj, dict) else None
                if isinstance(inner, list):
                    for chunk_obj in inner:
                        emb = chunk_obj.get("embedding")
                        if isinstance(emb, list):
                            dims_found.append(len(emb))
                elif isinstance(doc_obj, dict) and "embedding" in doc_obj:
                    # forma flat (fallback per embed standard)
                    dims_found.append(len(doc_obj["embedding"]))
        except Exception as e:
            print(f"parsing err: {e}")
        # usage info se presente
        usage = body.get("usage", {})
        if usage:
            print(f"usage: {usage}")
        if dims_found:
            print(f"vettori ricevuti: {len(dims_found)} · dim per vettore: {dims_found[0]}")
            if len(set(dims_found)) == 1:
                print(f"✓ tutti i vettori hanno {dims_found[0]} dimensioni")
            else:
                print(f"⚠ dimensioni eterogenee: {set(dims_found)}")
        else:
            print("⚠ struttura risposta non riconosciuta, dump primi 500 char:")
            print(_short(json.dumps(body), 500))

    # ── Q3: comportamento su overflow 32K token? ──
    # tiktoken cl100k_base ≈ stima conservativa per il tokenizer Voyage.
    print("\n──── Q3 · comportamento su overflow (~40K token totali) ────")
    try:
        import tiktoken
        tok = tiktoken.get_encoding("cl100k_base")
        # genera testo realistico ~40K token (oltre il limite 32K dichiarato)
        # un chunk lungo ripetuto in 40 chunk
        chunk_text = ("La piattaforma di metering raccoglie misure di energia "
                      "elettrica in tempo reale da contatori distribuiti sul "
                      "territorio nazionale, normalizza i flussi e li espone "
                      "tramite API documentate. " * 30)
        single_tok = len(tok.encode(chunk_text))
        n_chunks = max(1, 40000 // single_tok)
        big_chunks = [chunk_text] * n_chunks
        total_est = sum(len(tok.encode(c)) for c in big_chunks)
        print(f"payload stimato: {n_chunks} chunk × ~{single_tok} tok = ~{total_est} tok totali")
    except Exception as e:
        print(f"(tiktoken non disponibile: {e}; uso fallback su lunghezza)")
        big_chunks = ["lorem ipsum " * 4000] * 12  # ~48K token stima grossolana
    payload_big = {
        "inputs": [big_chunks],
        "model": MODEL,
        "input_type": "document",
    }
    target = API_BASE if r1["ok"] else (VOYAGE_DIRECT if r1_alt and r1_alt["ok"] else API_BASE)
    r3 = _call(target, payload_big, label=f"Q3 · overflow su {target}")
    if r3["ok"]:
        # quanti chunk sono tornati? il numero indica se ha troncato silently
        b = r3["body"]
        try:
            returned = sum(
                len(doc_obj.get("data", [])) for doc_obj in b.get("data", [])
            )
            print(f"chunk inviati: {len(big_chunks)} · chunk ricevuti: {returned}")
            if returned < len(big_chunks):
                print(f"⚠ TRONCAMENTO SILENZIOSO: ricevuti meno chunk del previsto")
            usage = b.get("usage", {})
            if usage:
                print(f"usage: {usage}")
        except Exception as e:
            print(f"parsing risposta overflow: {e}")

    # ── Summary ──
    print("\n==========================================================")
    print(" SUMMARY")
    print("==========================================================")
    print(f" Q1: endpoint `{API_BASE}` con chiave {key_kind}:")
    print(f"     {'✅ funziona' if r1['ok'] else '❌ NON funziona (status=' + str(r1['status']) + ')'}")
    if r1_alt:
        print(f"     fallback `{VOYAGE_DIRECT}`: "
              f"{'✅ funziona' if r1_alt['ok'] else '❌ NON funziona (status=' + str(r1_alt['status']) + ')'}")
    if succeeded and dims_found:
        print(f" Q2: dimensione vettore di default = {dims_found[0]}")
    else:
        print(f" Q2: non determinata")
    print(f" Q3: overflow handling — status {r3['status']}, "
          f"{'errore strutturato' if not r3['ok'] else 'accettato (sorprendente)'}")
    return 0 if r1["ok"] or (r1_alt and r1_alt["ok"]) else 2


if __name__ == "__main__":
    sys.exit(main())
