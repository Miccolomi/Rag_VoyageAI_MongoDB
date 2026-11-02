# RAG with MongoDB Atlas and Voyage AI

[Guida completa in italiano](README.it.md)

**Documented state: October 20, 2026.** Current decisions and evidence: [DEMO_PLAN.md](DEMO_PLAN.md).

A demonstration PoC for ingesting documents, comparing indexing strategies,
retrieving relevant passages and generating answers with source references.
It supports PDF and Office documents, including visual pages/slides, plus text
formats supported by the parser.

This guide describes repository behavior, not a completed deployment check.
The Italian guide also records the How it Works page structure and open issues.

The agreed demo objective and implementation roadmap are recorded in
[DEMO_PLAN.md](DEMO_PLAN.md) (Italian): 100 similar documents, folder upload,
and 5 questions with verified answers and sources, including at least 2 trick
questions. Evaluation uses a manual reference sheet. Strategy comparisons remain
verbal; no automated benchmark dashboard or new multimodal work is in scope.
The 100-PDF corpus is prepared locally and folder upload is implemented; the
five-question reference sheet is available in [DEMO_QUERIES.md](DEMO_QUERIES.md),
with answers verified against the original PDF tables. The user reported successful answers before question 5; question 5 failed and
was inspected directly. See the reference sheet for evidence and limitations.
Do not ingest the reference sheet. No bulk Atlas ingestion
or end-to-end upload test has been run as part of this change.

To prepare the files again, run `python scripts/prepare_financebench.py`. In
Upload choose text ingestion, GridFS, then **Una cartella**, and select
`data/financebench_100/pdfs`. Keep duplicate handling on skip and leave shared
title/author/year overrides empty. Click **Avvia ingestion**. Each file is processed sequentially, with progress and a downloadable
summary of successes, skips and errors. Keep the page open: this is not a
background job. Reference annotations and the manifest stay outside `pdfs`
and must not be ingested. Atlas indexing is asynchronous.

## Architecture: ingestion and querying

**Ingestion:** extract text, tables and metadata; use OCR where needed; split
content into chunks; render pages/slides for visual ingestion; generate
embeddings in the app or delegate them to Atlas. Store chunks and metadata in
the application collection and original files in GridFS. Atlas indexes update
asynchronously.

**Querying:** apply selected filters; retrieve through Atlas Search and Atlas
Vector Search; combine rank positions with `$rankFusion`; rerank candidates
through Atlas `$rerank` after `$rankFusion` in the same aggregation; pass selected
text to the configured LLM. The app shows
citations and provides access to originals. The prompt asks the LLM to use the
retrieved passages; citations do not guarantee factual accuracy.

The agentic page defaults to one search followed by user feedback. A refinement
requires a new question and can include context, document filters and search
settings; the feedback and links between searches are saved in MongoDB. The
optional automatic mode can perform additional searches within a configured limit.

The Query page also runs diagnostic searches to explain each channel. The
complete interaction involves additional database and external service calls.
Native reranking uses Atlas project enablement and database authentication;
app-generated embeddings still require a Voyage API key. The previous HTTP
path is available with `RERANK_BACKEND=voyage_api`; only that path skips
reranking if the API key is missing. Native errors have no silent fallback.

## Original files, GridFS and object storage

Original files are the complete uploaded files preserved in their binary format
so they can be downloaded and opened with the appropriate application.
[GridFS](https://www.mongodb.com/docs/manual/core/gridfs/) stores and retrieves
binary files of any format in MongoDB, including PDF, Word, Excel, PowerPoint
and images. It divides files into binary blocks and stores their metadata; the
driver reassembles the file on retrieval. These storage blocks serve a different
purpose from the passages extracted for search. Storing a format does not imply
that the demo's parsers support extracting its content.

The demo implements GridFS to keep binaries in MongoDB. Alternatively, files
can live in **object storage**, with searchable content, metadata and file
references in MongoDB. That alternative would require a dedicated application
integration.

## Four embedding strategies

Models below are code defaults; configuration can override them.

| Mode | Input | Embedding generation | Vector storage |
|---|---|---|---|
| `auto` | Chunk text, `voyage-4-large` | Atlas autoEmbed | Atlas index |
| `manual` | Chunk text, `voyage-4-large` | App calls Voyage | `embedding` |
| `context` | Chunks grouped with document context, `voyage-context-4` | App calls Voyage | `embedding_context` |
| `multimodal` | Text and rendered page/slide images, `voyage-multimodal-3.5` | App calls Voyage | `embedding_multimodal` |

The Upload page offers text ingestion using the selected text mode from Admin,
or text-and-visual ingestion using `multimodal`. Switching modes does not
convert existing chunks: ingest the document in each mode to compare them.
Versions of the same file can coexist; `skip` and `replace` apply to the active
mode. The Query page always searches all populated modes, independently of the
upload setting, without a mode selector. Optional document filters apply to
both text and vector branches.

Contextualized embedding preserves local chunk boundaries and page references.
The current implementation rejects a document when its estimated pre-chunked
input exceeds 32K tokens; it does not automatically split it into more groups.

## Parsers and visual content

PyMuPDF reads/renders PDF files, pdfplumber extracts tables, and Tesseract provides
OCR when installed. MarkItDown extracts text and structure from Office and other
supported formats. LibreOffice (`soffice`) converts Word/PowerPoint into PDF
pages for rendering.

LibreOffice is required for **visual Office ingestion**, not PDF ingestion.
Text-only Office ingestion also attempts conversion, with a MarkItDown fallback.
Voyage embeds the prepared text/images; it does not replace these parsers.

**Current multimodal limitation:** embeddings include images, but reranking and
answer generation receive text and source references only. A visually relevant
page with little extracted text may lose rank, and the LLM does not see its
image. Visual retrieval is implemented; image understanding through the entire
answer pipeline is not. See `src/rerank.py` and `src/generation.py`.

## Run locally and prepare the database

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# First setup only, if .env does not exist:
cp .env.example .env
python -m uvicorn server:app --port 8000
# then open localhost:8000
```

Before starting, configure `.env` with MongoDB connectivity, the Voyage key and
matching API endpoint, and the chosen LLM provider. Set `OFFICE_CONVERTER` or
`TESSERACT_CMD` if those executables are outside PATH. Install OCR language data
as needed. Verify network access, index creation permissions and Atlas feature
requirements for the selected modes. The app does not provision the cluster or
external services. Local secrets, virtual environments and uploaded documents
are excluded from the repository.

1. Open **Admin → Prepara database e indici**.
2. Read the report for connectivity, collections, the document lookup index, GridFS indexes and Atlas
   index creation requests for all four embedding modes.
3. Refresh Atlas status and wait for the active vector and shared text indexes
   to be ready. An accepted creation request is not completed indexing.
4. Inspect generated MongoDB commands in the index-code expander.
5. Upload a document, inspect the ingestion summary, then use Query.

Default resources:

- `pdf_chunks`: content chunks, metadata and app-generated vectors.
- `document_id_1_embedding_mode_1`: ordinary index on document identity and
  embedding mode for ingestion duplicate checks and document lookups.
- `pdf_files.files`, `pdf_files.chunks`: GridFS original files and ordinary indexes.
- `pdf_chunks_text_idx`: shared text search index.
- `pdf_chunks_vector_idx`, `pdf_chunks_manual_idx`, `pdf_chunks_context_idx`,
  `pdf_chunks_multimodal_idx`: vector indexes for the four strategies.

Setup creates missing resources and compares existing index definitions.
It reuses an equivalent existing document lookup index when available.
The separate **Admin → Verifica / Crea indici** button checks the same lookup
index alongside the Atlas indexes for the selected mode. Its **Drop & recreate**
option applies to those Atlas indexes, not the ordinary lookup index.
It updates mismatched definitions with `updateSearchIndex` and leaves matching
indexes untouched. The shared text index is processed once. Progress and the
report show per-index actions, field differences, previous/expected definitions,
MongoDB commands, errors and Atlas status. Incompatible index types are reported;
drop/recreate remains explicit. Updating an index does not recompute app-generated
vectors after a model or dimension change.

Atlas changes are asynchronous. During an update, `queryable=true` may refer to
the previous definition; wait for `READY` before using the new one.

Office whole-document excerpts and MarkItDown fallbacks no longer receive a
synthetic page 1. Admin also repairs existing `office_structure` and
`creator=MarkItDown` page metadata without changing content or vectors.
Rendered PDF page/slide references are preserved. Missing-page citations point
to the document only. `python -m src.init_db` prepares collections and GridFS
indexes; Admin includes the citation metadata repair and Atlas index setup.

## Filters and reranking

Query exposes document-field filters. `$vectorSearch` requires corresponding
`filter` index fields; `$search` uses compatible native filters and residual
`$match` conditions. These filters do not inspect vector coordinates.
The rerank threshold checks the highest score and can stop generation; it is
neither a per-passage threshold nor a probability that the answer is correct.

Atlas also offers richer vector pre-filters through its `vectorSearch` operator
inside `$search`, as described in the [hybrid search documentation](https://www.mongodb.com/docs/vector-search/hybrid-search/hybrid-search-overview/).
This app uses the separate `$vectorSearch` stage. Native
[`$rerank`](https://www.mongodb.com/docs/vector-search/query/aggregation-stages/rerank/)
is the default implementation (`RERANK_BACKEND=native`), placed after fusion,
not inside its input branches. Fusion scores and details are preserved separately
from rerank scores. All Top-K candidates are returned for the before/after
explanation; only Top-N reaches the LLM. The cluster was verified as 9.0.4 with
native rerank-3 enabled on October 19, 2026. Official pages currently differ on
minimum version requirements; verify the chosen model on the target deployment.

**Instruction-guided reranking:** Voyage supports instructions appended or
prepended to the query, such as prioritizing actionable procedures for a
troubleshooting question. This adapts candidate ordering to the user's purpose
using already indexed documents. Instructions influence relevance; mandatory
exclusions require explicit filters. See the
[Voyage documentation](https://www.mongodb.com/docs/voyageai/models/rerankers/?client=python).
How it Works highlights this capability with a visible example. The Query page
provides **Istruzioni al reranker (opzionale)** below the question. The field starts
empty. Empty or whitespace-only input preserves the original query; otherwise
`src/rerank.py` formats the instructions for native `$rerank.query.text`, or
the HTTP API's `query` field when that backend is configured.
Retrieval and generation keep the original question. Reranking details show
the submitted question and instructions. Instructions are not applied when reranking is
skipped. An automatic comparison on identical candidates remains future work.
Benefits on this corpus need measurement; no claim of competitive exclusivity
is made here.

The `author` field is mapped as `token` for Search `equals`/`in` and is
included in the multimodal vector index filters. Admin reconciles old schemas.

## Demonstration and next steps

Use the same corpus, questions, filters and parameters for each comparison.
Observe the expected source's rank, relevance, stage timings and citation
accuracy. Do not compare raw scores from different stages as a shared scale.
Include exact terms, paraphrases, context-dependent passages, visual pages,
filtered queries and an unrelated question. These are suggested scenarios,
not benchmark results.

The reorganized How it Works page separates ingestion and querying, summarizes
the four embedding modes, and includes an illustrative walkthrough with sources
to look for. Technical details remain in expanders. Repeated feature lists,
static “KPI” cards and the generic traditional-RAG comparison have been removed.
See the [Italian project guide](README.it.md#pagina-how-it-works) for the page
structure and remaining technical work. The visual downstream limitation remains open; the author index mappings are aligned by setup.

## Query page

The **Domanda** selector contains the five agreed FinanceBench questions and
**Domanda libera**. Free text starts empty (the old Doganella default is cleared),
and user drafts survive switching to a preset. Scenario selection and its cards
are removed. Only question text is sent; expected answers and trap descriptions
are never added. Query 2 has been removed.

Only **Esegui** starts retrieval and generation: browsing rankings, passages and
the pipeline does not call Atlas, Voyage or the model again.

Query still searches every populated embedding mode with one text branch and
one vector branch per mode. The displayed pipeline is the actual Atlas request.
RRF combines ranks with weight 1, deduplicating by `_id`; Top-K limits each branch
and the final fusion. Native reranking returns all candidates for comparison;
only Top-N reaches the LLM. Missing required indexes stop execution. The scope
inventory is cached for 30 seconds and refreshed on execution.

The result layout is: accepted answer, downloads of cited originals, Trust Panel,
file/page generation details, then five explanation tabs:

1. Query analysis and executed MongoDB pipeline.
2. Separate BM25 and textual-vector diagnostic lists. BM25 highlights words;
   vector passages are plain text, with subsequent rerank position and citation
   status. These diagnostic searches are separate from the main execution.
   The multimodal diagnostic table is omitted; its main search branch stays active.
3. Official MongoDB RRF formula, explained weights/constant 60, illustrative
   example and actual branch ranks. NA/N/A means absence, fixing the old false
   shared-result counts. Internal chunk-number column and text inspector removed.
4. Before/after rerank positions, movements, LLM inclusion and citations. Native
   mode provides all Top-K positions; API mode only the returned positions.
   No fabricated explanations for individual scores; no separate Rank Journey.
5. Simple table of passages actually supplied: reference, file, page and cited
   yes/no. The prompt expander shows only the recorded SYSTEM instructions,
   deduplicated across attempts; USER messages remain hidden.
   Old runs without recorded system prompts are explicitly identified.

### Generation, citations and limitations

The runtime observed in this task remains Ollama `mistral-nemo:latest`, Top-K 50,
Top-N 8, context 8192, temperature 0.1 and top_p 0.9. Qwen3.6 and larger context/
Top-N were proposed but not activated. Native reranking defaults to `rerank-3`.
Configuration environment variables may override repository defaults.

The general prompt lives in `prompts/system_prompt.txt` and is reread per request,
with existing inline/file overrides respected. It covers evidence, tables,
source limitations and internal citations. It contains no reference answers.
The special increase/decrease and subtraction instructions added after the Adobe
failure were removed at the user's request: do not tune the demo to its questions.

`generate_reviewed_answer` makes at most two attempts on the same passages.
External/non-internal links, missing citations and out-of-range references fail
validation. If the final attempt fails, the generated answer is withheld instead
of being rendered with a warning. Raw attempts remain in memory for diagnosis.
An insufficient-evidence response may still be shown. Citation checks do not
validate arithmetic, factual accuracy or whether a passage supports a claim.
The Ollama `/api/chat` path supplies no browsing tools. A generated external URL
is not evidence of web access; Atlas/Voyage calls still use configured services.

Historical checks: a fixed public 3M context gave correct answers in 2/3 paired
runs with the old prompt and 3/3 with the revised general prompt. This does not
prove general reliability or success with the requested cash-flow source: page
46 was outside the eight supplied passages. On October 20, Adobe question 5
produced +476 and an external link despite receiving the correct 2021/2022 values
(4,822 and 4,756). It remains a failed case; no new LLM run has validated the later
citation-blocking changes. Details and manual results are in
[DEMO_PLAN.md](DEMO_PLAN.md) and [DEMO_QUERIES.md](DEMO_QUERIES.md).

Restart the web server (or run it with `--reload`) after Python-module or
configuration changes. Prompt-file edits are read per call.

## Client-side encryption of agent memory (CSFLE)

The first question of each session (`agentic_sessions.first_question`) is encrypted by
the application before it reaches Atlas, using explicit Client-Side Field Level
Encryption with the randomized algorithm. Atlas, backups and cluster administrators
see only binary data (BSON subtype 6); the application, which holds the key, reads it.

- Master key: 96 bytes in a local file outside the project (`CSFLE_MASTER_KEY_PATH`,
  default `~/.config/rag-demo/csfle-master-key.bin`); production deployments should use a KMS.
- Data key: `encryption.__keyVault`, alt name `rag-demo-sessioni`.
- Setup once per cluster: `.venv/bin/python -m src.field_encryption setup`.
- Only the first question is encrypted; the same text remains in plain form in
  `memory.messages`, `memory.turn_summaries` and `agentic_runs`.
- Automatic encryption is not used: MongoDB documents that `$rerank` is not supported
  by clients configured for automatic encryption.
- Two machines on the same cluster need the same master key file.

## Project structure

- `server.py`: web server (Starlette), JSON APIs and event streams.
- `web/`: the pages (`index.html`, `style.css`, `app.js`) and the architecture diagram.
- `src/web_services.py`: page services (step-by-step Query, Upload, Admin, health checks).
- `src/agent.py`, `src/agentic_search.py`, `src/agentic_memory.py`: agentic search and its MongoDB memory.
- `src/retrieval_details.py`: passage identity and fusion branch ranks.
- `src/search_scope.py`: shared search inventory, scope and index explanations.
- `src/answer_quality.py`: answer generation, role-tag cleanup and citation checks.
- `src/`: ingestion, embedding, retrieval, reranking, generation and configuration.
- `src/init_db.py`, `src/indexes.py`: collection and index setup.
- `src/field_encryption.py`: client-side encryption (CSFLE) of session first questions.
- `schemas/`: Atlas Search and Vector Search index definitions.
- `prompts/`: LLM instructions for answers with sources.
- `tests/`: offline tests (mongomock); `scripts/`: FinanceBench corpus preparation
  and one-off Voyage API checks.

Update both guides when setup, modes, dependencies or limitations change.
