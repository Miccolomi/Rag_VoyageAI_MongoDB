"""Valutazione del retrieval sulle domande annotate di FinanceBench.

Per ogni domanda la cui fonte è presente nella collection misura in che
posizione compare la fonte corretta, per configurazione:

  text            solo $search (BM25)
  vector:<mode>   solo $vectorSearch per una modalità di embedding
  hybrid          $rankFusion (testo + tutte le modalità presenti)
  hybrid_rerank   $rankFusion + $rerank (o Voyage API, secondo RERANK_BACKEND)
  hybrid_rerank_docfilter
                  come sopra, con filtro sul file corretto: è il limite
                  superiore di ciò che un agente ottiene estraendo azienda e
                  anno dalla domanda e trasformandoli in pre-filter

Due livelli di corrispondenza:
  documento  il risultato viene dal file giusto (robusto)
  pagina     file giusto e pagina tra quelle di evidenza annotate

Uso (dalla radice del progetto, con lo stesso .env della demo):
  python -m scripts.evaluate_retrieval --limit 30
  python -m scripts.evaluate_retrieval --configs text,hybrid,hybrid_rerank --k 1 5 10

FinanceBench indica le pagine di evidenza con `evidence_page_num`. Lo script
assume che siano numerate da 0 (pagina 1 = 0); se i risultati a livello
pagina sembrano sfasati di uno, rilancia con --page-base 1 e confronta.

Il materiale di riferimento (risposte attese) non viene mai inviato a
modelli: lo script usa solo domande, nome del file e pagine di evidenza.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.rerank import rerank  # noqa: E402
from src.retrieval import (  # noqa: E402
    EMBED_MODES, build_native_rerank_pipeline, query_vector_for_mode, search,
)

DEFAULT_QUESTIONS = (Path(__file__).resolve().parent.parent
                     / "data" / "financebench_100" / "questions_reference.jsonl")
ALL_CONFIGS = ["text", "vector", "hybrid", "hybrid_rerank", "hybrid_rerank_docfilter"]


def load_questions(path: Path, available_files: set[str], page_base: int) -> list[dict[str, Any]]:
    """Domande con fonte presente nella collection; pagine convertite a base 1."""
    questions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        filename = f"{row.get('doc_name', '')}.pdf"
        if filename not in available_files:
            continue
        pages = set()
        for evidence in row.get("evidence") or []:
            page = evidence.get("evidence_page_num")
            if isinstance(page, int) and (evidence.get("doc_name") in (None, row.get("doc_name"))):
                pages.add(page + (1 - page_base))
        questions.append({
            "id": row.get("financebench_id") or f"q{len(questions) + 1}",
            "question": row.get("question", ""),
            "filename": filename,
            "pages": pages,
            "question_type": row.get("question_type"),
        })
    return questions


def first_rank(results: list[dict[str, Any]], filename: str, pages: set[int] | None) -> int | None:
    for rank, doc in enumerate(results, start=1):
        if doc.get("filename") != filename:
            continue
        if pages is None or doc.get("page_number") in pages:
            return rank
    return None


def query_vectors(question: str, modes: list[str]) -> dict[str, list[float]]:
    targets = [mode for mode in modes if mode != "auto"]
    if not targets:
        return {}
    with ThreadPoolExecutor(max_workers=len(targets)) as pool:
        futures = {mode: pool.submit(query_vector_for_mode, question, mode) for mode in targets}
        return {mode: future.result() for mode, future in futures.items()}


def hybrid_rerank(coll, question: str, *, top_k: int, top_n: int, modes: list[str],
                  vectors: dict[str, list[float]], filters: dict | None = None) -> list[dict[str, Any]]:
    if config.RERANK_BACKEND == "native":
        return list(coll.aggregate(build_native_rerank_pipeline(
            question, filters=filters, top_k=top_k, top_n=top_n, embed_mode="all",
            modes=modes, query_vectors=vectors,
        )))
    candidates = search(coll, question, filters=filters, top_k=top_k, embed_mode="all",
                        retrieval_mode="hybrid", modes=modes, query_vectors=vectors)
    return rerank(question, candidates, top_n=min(top_n, len(candidates)))


def run(args: argparse.Namespace) -> int:
    coll = config.get_collection()
    available_files = set(coll.distinct("filename"))
    present_modes = set(coll.distinct("embedding_mode"))
    modes = [mode for mode in EMBED_MODES if mode in present_modes]
    if args.modes:
        modes = [mode for mode in args.modes.split(",") if mode in modes]
    if not modes:
        print("Nessuna modalità di embedding con documenti nella collection.")
        return 1

    questions = load_questions(Path(args.questions), available_files, args.page_base)
    if args.limit:
        questions = questions[:args.limit]
    if not questions:
        print("Nessuna domanda ha la propria fonte nella collection. Hai caricato i PDF di FinanceBench?")
        return 1

    configs = args.configs.split(",") if args.configs else ALL_CONFIGS
    unknown = set(configs) - set(ALL_CONFIGS)
    if unknown:
        print(f"Configurazioni sconosciute: {sorted(unknown)}")
        return 1
    max_k = max(args.k)
    print(f"{len(questions)} domande · modalità {modes} · configurazioni {configs} · "
          f"rerank {config.RERANK_BACKEND}/{config.VOYAGE_RERANK_MODEL} · "
          f"pesi fusione {config.HYBRID_TEXT_WEIGHT_MODE}\n", flush=True)

    rows: list[dict[str, Any]] = []
    for number, item in enumerate(questions, start=1):
        question = item["question"]
        try:
            vectors = query_vectors(question, modes)
        except Exception as exc:
            print(f"[{number}] embedding della domanda non riuscito: {exc}")
            continue
        runs: list[tuple[str, Any]] = []
        if "text" in configs:
            runs.append(("text", lambda: search(coll, question, top_k=args.top_k, embed_mode="all",
                                                retrieval_mode="text", modes=modes)))
        if "vector" in configs:
            for mode in modes:
                runs.append((f"vector:{mode}", lambda mode=mode: search(
                    coll, question, top_k=args.top_k, embed_mode=mode, retrieval_mode="vector",
                    modes=[mode], query_vectors=vectors)))
        if "hybrid" in configs:
            runs.append(("hybrid", lambda: search(coll, question, top_k=args.top_k, embed_mode="all",
                                                  retrieval_mode="hybrid", modes=modes, query_vectors=vectors)))
        if "hybrid_rerank" in configs:
            runs.append(("hybrid_rerank", lambda: hybrid_rerank(
                coll, question, top_k=args.top_k, top_n=max_k, modes=modes, vectors=vectors)))
        if "hybrid_rerank_docfilter" in configs:
            runs.append(("hybrid_rerank_docfilter", lambda: hybrid_rerank(
                coll, question, top_k=args.top_k, top_n=max_k, modes=modes, vectors=vectors,
                filters={"filename": item["filename"]})))

        for name, call in runs:
            started = time.perf_counter()
            try:
                results = call()
                error = ""
            except Exception as exc:
                results, error = [], f"{type(exc).__name__}: {str(exc)[:200]}"
            elapsed = (time.perf_counter() - started) * 1000
            rows.append({
                "question_id": item["id"], "question_type": item["question_type"],
                "config": name, "doc_rank": first_rank(results, item["filename"], None),
                "page_rank": first_rank(results, item["filename"], item["pages"]) if item["pages"] else None,
                "has_pages": bool(item["pages"]), "results": len(results),
                "latency_ms": round(elapsed, 1), "error": error,
            })
        print(f"[{number}/{len(questions)}] {item['id']} · {item['filename']}", flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detail_path = out_dir / f"eval_{stamp}_dettaglio.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["config"])
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows, args.k)
    summary_path = out_dir / f"eval_{stamp}_sintesi.md"
    summary_path.write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"Dettaglio: {detail_path}\nSintesi:   {summary_path}")
    return 0


def summarize(rows: list[dict[str, Any]], ks: list[int]) -> str:
    """Tabella Markdown: recall@k (documento e pagina), MRR pagina, latenza mediana."""
    configs = list(dict.fromkeys(row["config"] for row in rows))
    header = (["Configurazione", "Domande"] + [f"Doc@{k}" for k in ks] + [f"Pag@{k}" for k in ks]
              + ["MRR pag", "Latenza mediana (ms)", "Errori"])
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for name in configs:
        subset = [row for row in rows if row["config"] == name]
        ok = [row for row in subset if not row["error"]]
        paged = [row for row in ok if row["has_pages"]]
        cells = [name, str(len(ok))]
        for k in ks:
            hits = sum(1 for row in ok if row["doc_rank"] is not None and row["doc_rank"] <= k)
            cells.append(f"{hits / len(ok):.0%}" if ok else "—")
        for k in ks:
            hits = sum(1 for row in paged if row["page_rank"] is not None and row["page_rank"] <= k)
            cells.append(f"{hits / len(paged):.0%}" if paged else "—")
        mrr = (sum(1 / row["page_rank"] for row in paged if row["page_rank"]) / len(paged)) if paged else None
        cells.append(f"{mrr:.3f}" if mrr is not None else "—")
        cells.append(f"{statistics.median(row['latency_ms'] for row in ok):.0f}" if ok else "—")
        cells.append(str(len(subset) - len(ok)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    parser.add_argument("--configs", default="", help="Elenco separato da virgole; default tutte")
    parser.add_argument("--modes", default="", help="Limita le modalità di embedding (es. auto,manual)")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 5, 10, 20])
    parser.add_argument("--top-k", type=int, default=50, help="Candidati per ramo, come Top-K nella demo")
    parser.add_argument("--limit", type=int, default=0, help="Numero massimo di domande")
    parser.add_argument("--page-base", type=int, choices=[0, 1], default=0,
                        help="Base della numerazione di evidence_page_num (default 0)")
    parser.add_argument("--out", default="results")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
