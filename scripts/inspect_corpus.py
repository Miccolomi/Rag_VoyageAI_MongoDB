"""Inspector del corpus reale in MongoDB: documenti, chunk, contenuti.

Output:
- lista documenti distinti con chunk count
- per ogni documento: 5 chunk random + i primi/ultimi
- ricerca per keyword in tutti i chunk
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config


def _wrap(s: str, w: int = 100, indent: str = "    ") -> str:
    s = (s or "").strip().replace("\n", " ")
    return textwrap.fill(s, width=w, initial_indent=indent, subsequent_indent=indent)


def main() -> int:
    coll = config.get_collection()
    total = coll.estimated_document_count()
    print(f"\n{'=' * 80}\n  COLLECTION `{config.MONGODB_COLLECTION}` · {total} chunk totali\n{'=' * 80}")

    # 1. Documenti distinti
    docs = list(coll.aggregate([
        {"$group": {
            "_id": "$document_id",
            "filename":    {"$first": "$filename"},
            "title":       {"$first": "$title"},
            "chunks":      {"$sum": 1},
            "total_pages": {"$first": "$total_pages"},
        }},
        {"$sort": {"chunks": -1}},
    ]))
    print(f"\n📂 {len(docs)} documenti distinti:\n")
    for d in docs:
        print(f"  • {d['filename']}  ({d['chunks']} chunk · {d.get('total_pages', '?')} pagine)")

    # 2. Per ogni documento, mostra 6 chunk: primo, ultimo, 4 random
    for d in docs:
        print(f"\n{'─' * 80}\n  📄 {d['filename']}\n{'─' * 80}")
        chunks = list(coll.find(
            {"document_id": d["_id"]},
            {"chunk_index": 1, "content": 1, "page_number": 1, "content_type": 1}
        ).sort("chunk_index", 1))

        if len(chunks) <= 8:
            sample = chunks
        else:
            # primo, ultimo, 6 distribuiti
            n = len(chunks)
            indices = sorted({0, n - 1} | {int(n * k / 7) for k in range(1, 7)})
            sample = [chunks[i] for i in indices]

        for c in sample:
            idx = c.get("chunk_index", "?")
            page = c.get("page_number")
            ctype = c.get("content_type", "?")
            content = c.get("content", "")
            # Trunca a ~200 char
            preview = content[:300] + ("…" if len(content) > 300 else "")
            print(f"  [chunk #{idx} · page={page} · type={ctype}]")
            print(_wrap(preview, w=78))
            print()

    # 3. Ricerca per keyword utili — vediamo cosa c'è
    print(f"\n{'=' * 80}\n  🔎 KEYWORD SEARCH (per capire cosa è citato)\n{'=' * 80}\n")
    keywords = [
        "CEI 11-4", "CEI 211-4", "ALLEGATO A.3",
        "DPCM", "decreto", "induzione", "telecomunicazione",
        "cavo", "obiettivo", "metering", "misurazione", "sanzioni",
        "responsabilità", "obblighi", "valori", "soglia", "limite",
        "schermatura", "messa a terra", "alta tensione",
    ]
    for kw in keywords:
        # case-insensitive substring search
        n = coll.count_documents({"content": {"$regex": kw, "$options": "i"}})
        marker = "✓" if n > 0 else "✗"
        print(f"  {marker} '{kw}': {n} chunk")

    return 0


if __name__ == "__main__":
    sys.exit(main())
