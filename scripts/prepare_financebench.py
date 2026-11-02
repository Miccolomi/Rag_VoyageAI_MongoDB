"""Download 100 public FinanceBench PDFs locally; never connect to Atlas or Voyage."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen


REPOSITORY = "patronus-ai/financebench"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "financebench_100"


def fetch(url: str) -> bytes:
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers={"User-Agent": "RAG-demo-corpus-preparation"}), timeout=45) as response:
                return response.read()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    raise RuntimeError("Download fallito")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = args.output.resolve()
    pdf_dir = root / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        tree = json.loads(fetch(f"https://api.github.com/repos/{REPOSITORY}/git/trees/main?recursive=1"))
        if tree.get("truncated"):
            raise RuntimeError("Elenco GitHub incompleto: nessuna selezione effettuata.")
        revision = tree["sha"]
        available = {
            Path(entry["path"]).stem: entry for entry in tree["tree"]
            if entry["type"] == "blob" and entry["path"].startswith("pdfs/")
            and entry["path"].lower().endswith(".pdf")
        }
        base = f"https://raw.githubusercontent.com/{REPOSITORY}/{revision}"
        annotations = fetch(f"{base}/data/financebench_open_source.jsonl")
        questions = [json.loads(line) for line in annotations.decode().splitlines() if line.strip()]
        # Keep annotated sources first; fill with earlier/later reports as distractors.
        annotated = {q["doc_name"] for q in questions}
        required = {"3M_2018_10K", "ADOBE_2022_10K", "AMAZON_2022_10K"}
        missing = required - available.keys()
        if missing:
            raise RuntimeError(f"Fonti iniziali mancanti: {sorted(missing)}")
        selected = sorted(required)
        selected += sorted((annotated & available.keys()) - set(selected))
        selected += sorted(available.keys() - set(selected))
        selected = selected[:100]
        if len(selected) != 100:
            raise RuntimeError("Meno di 100 PDF disponibili.")
        manifest = {
            "source": f"https://github.com/{REPOSITORY}",
            "revision": revision,
            "prepared_at": datetime.now(timezone.utc).isoformat(),
            "documents": [
                {"filename": Path(available[name]["path"]).name,
                 "url": f"{base}/{available[name]['path']}",
                 "git_blob_sha": available[name]["sha"],
                 "size_bytes": available[name]["size"],
                 "has_annotated_question": name in annotated}
                for name in selected
            ],
        }
        # Reference material deliberately lives OUTSIDE the folder to upload.
        (root / "questions_reference.jsonl").write_bytes(annotations)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    def download(entry: dict) -> dict:
        filename = entry["filename"]
        if Path(filename).name != filename or not filename.lower().endswith(".pdf"):
            raise ValueError("Nome PDF non valido nel manifest")
        if not entry["url"].startswith(f"https://raw.githubusercontent.com/{REPOSITORY}/"):
            raise ValueError("Origine non valida nel manifest")
        target = pdf_dir / filename

        def matches(data: bytes) -> bool:
            blob = b"blob " + str(len(data)).encode() + b"\0" + data
            return (data.startswith(b"%PDF-") and len(data) == entry["size_bytes"]
                    and hashlib.sha1(blob).hexdigest() == entry["git_blob_sha"])

        data = target.read_bytes() if target.exists() else b""
        if not matches(data):
            data = fetch(entry["url"])
            if not matches(data):
                raise ValueError(f"PDF incompleto o diverso dalla fonte: {filename}")
            partial = target.with_suffix(".pdf.part")
            partial.write_bytes(data)
            partial.replace(target)
        return {**entry, "sha256": hashlib.sha256(data).hexdigest()}

    completed, failures = [], []
    print(f"Preparazione di {len(manifest['documents'])} PDF in {pdf_dir}", flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(download, entry): entry for entry in manifest["documents"]}
        for number, future in enumerate(as_completed(futures), 1):
            try:
                completed.append(future.result())
            except Exception as exc:
                failures.append({"filename": futures[future]["filename"], "error": str(exc)})
            if number % 10 == 0 or number == len(futures):
                print(f"{number}/{len(futures)} esaminati · {len(completed)} pronti · {len(failures)} errori", flush=True)
    by_name = {entry["filename"]: entry for entry in completed}
    manifest["documents"] = [by_name.get(entry["filename"], entry) for entry in manifest["documents"]]
    manifest["download_errors"] = failures
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit("Download incompleto: dettagli nel manifest. Riesegui per scaricare i file mancanti.")
    total_mb = sum(entry["size_bytes"] for entry in completed) / (1024 * 1024)
    print(f"PRONTI: {len(completed)} PDF · {total_mb:.1f} MB. In Upload seleziona SOLO: {pdf_dir}", flush=True)


if __name__ == "__main__":
    main()
