# RAG Demo · MongoDB + Voyage AI

*[Versione italiana](README.it.md)*

A demo that answers questions about 100 public financial documents from 34 US companies (mostly
10-K annual reports, plus some quarterly reports and earnings releases), **citing the pages** each
piece of information comes from.

It shows two approaches:

- **Query**: classic RAG. One search, one answer. You can see step by step how MongoDB picks the
  passages: keyword search, semantic search, fusion of the two rankings and a final reranking with a
  Voyage AI model.
- **Agentic search**: an agent that searches again, in a targeted way, when the first search is not
  enough. It splits multi-part questions, asks which year you mean when values change from year to
  year, and states what it could not find.

Everything happens inside **MongoDB Atlas**: documents, text search, vector search, fusion,
reranking, conversation memory and original files. That is the point of the demo: MongoDB is the
platform, not just the database.

> The application interface is in Italian. You can ask questions in English or Italian.

---

## Contents

1. [What you need](#1-what-you-need)
2. [Installation](#2-installation)
3. [Configuration: the .env file](#3-configuration-the-env-file)
4. [First start](#4-first-start)
5. [Prepare the database and indexes](#5-prepare-the-database-and-indexes)
6. [Download and load the 100 documents](#6-download-and-load-the-100-documents)
7. [Ask your first questions](#7-ask-your-first-questions)
8. [The demo pages](#8-the-demo-pages)
9. [Optional features](#9-optional-features)
10. [Troubleshooting](#10-troubleshooting)
11. [How it is built](#11-how-it-is-built)
12. [Data and licences](#12-data-and-licences)

---

## 1. What you need

| What | Why | Required |
|---|---|---|
| **Python 3.11 or 3.12** | runs the application | yes |
| **A MongoDB Atlas cluster** | stores documents, indexes and memory | yes |
| **An API key for a language model** (for example Anthropic) | writes the answers | yes |
| **A Voyage AI key** | only for the manual, context and multimodal modes | no |
| **Git** | to download the project | recommended |
| **Tesseract**, **LibreOffice** | OCR for scanned pages, visual pages of Word and PowerPoint files | no |

**The Atlas cluster.** The demo uses recent MongoDB features: `$rankFusion` (MongoDB 8.1 or later),
Vector Search with automatic embeddings (`autoEmbed`) and the `$rerank` stage. It is tested on
MongoDB Atlas 9.0. If something is missing in your cluster, the app's component check (see
[step 4](#4-first-start)) tells you straight away.

**Costs.** Atlas, Voyage AI and the language model may have usage costs. Loading 100 reports means
computing embeddings for about 33,000 passages.

**Python.** The demo is tested with Python 3.11 and 3.12. With very recent versions some libraries
(for example the one that reads PDFs) may not have a ready-made package yet, and installation fails.
With `uv` you do not need to worry: the command in step 2 downloads Python 3.12 by itself.

> The commands in this guide are for macOS and Linux. On Windows replace `.venv/bin/python` with
> `.venv\Scripts\python`.

---

## 2. Installation

**1. Download the project** and enter the folder:

```bash
git clone <repository-url>
cd <folder-name>
```

**2. Create the Python environment and install the libraries.** Pick one of the two ways.

With [uv](https://docs.astral.sh/uv/), faster:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

With the standard Python tools, if you already have Python 3.11 or 3.12 (check with
`python3 --version`; you can use `python3.12` instead of `python3`):

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

**3. Check that everything is in place** with the automated tests. They connect neither to Atlas
nor to any model, so they work even before configuration. They need two extra libraries, listed in
`requirements-dev.txt`:

```bash
uv pip install --python .venv/bin/python -r requirements-dev.txt   # or: .venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
```

They should all be `passed`.

---

## 3. Configuration: the .env file

Settings and keys live in a file called `.env` in the project folder. Create it from the template:

```bash
cp .env.example .env
```

Then open `.env` in a text editor. The file is split into numbered, commented sections (comments are
in Italian). **You only need to fill in the first three**; every line starting with `#` already has
a default value.

**Section 1: MongoDB Atlas.** The connection string of your cluster.

1. In Atlas: *Database* → your cluster → **Connect** → *Drivers* → copy the `mongodb+srv://...`
   string.
2. Replace user and password with those of a database user (*Database Access*).
3. In *Network Access* add your computer's IP address.

```
MONGODB_URI=mongodb+srv://user:password@cluster.xxxxx.mongodb.net/
```

**Section 2: Voyage AI.** If for now you only use the **auto** mode (recommended to start), you can
leave it empty: Atlas computes embeddings and reranking. The other modes need a Voyage AI key:

- created from your Atlas project: it starts with `al-` and works with the default address;
- created on the Voyage AI website: it starts with `pa-` and also needs
  `VOYAGE_API_BASE=https://api.voyageai.com/v1`.

```
VOYAGE_API_KEY=al-...
```

**Section 3: the model that writes the answers.** The simplest option is Anthropic:

```
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
```

Alternatively `gemini` (Google) or `ollama` (a model on your own computer, no keys).

> **The `.env` file holds your keys: never share or publish it.** It is already excluded from git by
> `.gitignore`. `.env.example`, on the other hand, is the key-free template and is part of the
> project.

---

## 4. First start

```bash
.venv/bin/python -m uvicorn server:app --port 8000
```

Open your browser at **http://localhost:8000**.

At the top right there is a status indicator. Click it: the app checks the Atlas connection, the
documents, the indexes, hybrid search with reranking, encryption and the answer model one by one,
and tells you what works and what does not. On first start it is normal for documents and indexes
to be missing: you prepare them in the next two steps.

To stop the app press `Ctrl+C` in the terminal.

---

## 5. Prepare the database and indexes

Do this once, on a new cluster.

1. Open the **Admin** page.
2. In the *Indici di ricerca* (search indexes) box press **Verifica e prepara gli indici**.

The app creates the collections, the indexes for the original files (GridFS), the text index and one
vector index for each of the four modes. It does not delete anything. Atlas takes a few minutes to
make the indexes ready: in the table their status changes to `READY`.

---

## 6. Download and load the 100 documents

The PDFs are **not included** in the project: they are 180 MB and belong to the public FinanceBench
dataset. A script downloads them, always the same 100 files, checking that each one is identical to
the original.

**1. Download the PDFs** (about one minute):

```bash
.venv/bin/python scripts/prepare_financebench.py
```

You will find 100 PDFs in `data/financebench_100/pdfs`. If the download stops, run the same command
again: it resumes with the missing files.

**2. Load them into the demo:**

1. Open the **Upload** page.
2. Choose the **auto** mode.
3. Press **una cartella** (a folder) and select `data/financebench_100/pdfs`. Select `pdfs` only,
   not the folder above it, which also contains service files.
4. Press **Avvia il caricamento** (start loading).

For each document the app extracts the text (tables included), splits it into passages, stores them
in MongoDB and keeps the original PDF in GridFS. With 100 reports this takes a while: keep the page
open. You can already ask questions about the documents loaded so far.

---

## 7. Ask your first questions

Open **Query** or **Ricerca agentica** (agentic search). Below the question box there are some
ready-made examples; click one and start the search. For example:

- *What were 3M's capital expenditures in 2018, in millions of dollars?*
- *How much did Adobe spend on research and development in fiscal 2022?*
- *How many manufacturing plants did 3M have in 2018, and how much did Adobe spend on R&D in fiscal
  2022?* (a two-part question: see how the agent splits it)
- *How do you make a Neapolitan pizza?* (off-topic: the app does not make up an answer)

The example questions are designed for these documents: for other companies and years the answers
depend on what the 100 files contain.

The **Dove cercare** (where to search) selector below the question decides which passages to search:
one indexing mode or **tutte** (all of them).

---

## 8. The demo pages

| Page | What it shows |
|---|---|
| **MongoDB** | why MongoDB is the platform of the demo: document model, search, AI, security |
| **Come funziona** (how it works) | the three paths: question, agentic search, document loading |
| **Upload** | document loading, with the four indexing modes |
| **Query** | classic RAG, step by step |
| **Ricerca agentica** | the agent, with the timeline of every decision |
| **Admin** | status, indexes, documents per mode, reset |

### Query

1. MongoDB searches in parallel **by keywords** (`$search`, BM25) and **by meaning**
   (`$vectorSearch`).
2. `$rankFusion` merges the two rankings.
3. `$rerank` reorders the candidates with a Voyage AI model.
4. The best passages go to the model, which writes **a single answer** with citations.

After the answer, *Dentro la ricerca* (inside the search) shows what each engine found, how the
rankings were merged, what the reranking changed and the MQL pipeline that ran. Every number in the
answer is checked against the cited passage. With the **Equilibrio della fusione** (fusion balance)
slider you can shift the weight between keywords and meaning and immediately see how the sources
change, without calling the model.

If the best passage is below the **minimum relevance threshold**, the model is not called.

### Agentic search

The agent runs a first search and **checks element by element** whether the evidence is enough. If
something is missing it runs targeted searches, up to 5 additional ones (configurable in the
settings). It stops earlier if a search brings no new passages or if two different searches find
nothing: in that case it does not make up an answer and does not show unrelated sources.

- **Multi-part questions**: one search per part.
- **Different values in different years**, with a question that has no year: it asks which year you
  mean.
- **Conversation**: it remembers the previous questions of the session, so you can ask "and in
  2019?".
- **Computed numbers**: if the answer adds up two values from the sources, it says so and shows the
  calculation.

Every step is stored in MongoDB (`agentic_runs`) and can be reviewed from the session questions menu.

### The four indexing modes

| Mode | Who computes the vectors | When to use it |
|---|---|---|
| **auto** | MongoDB Atlas, with Voyage AI | to start: no embedding code |
| **manual** | the application, with the Voyage AI API | to choose model and chunking |
| **context** | the application, with `voyage-context` | each passage takes the rest of the document into account |
| **multimodal** | the application, with `voyage-multimodal` | slides, charts, scans: image and text of the page |

In **manual** the Upload page lets you choose the chunking strategy (by structure or fixed window),
the passage size and the overlap.

**Limit of the context mode:** it works with documents up to about 32,000 tokens. 10-K reports are
longer, so for this dataset use auto or manual.

---

## 9. Optional features

**Client-side field level encryption (CSFLE).** The first question of each agent session can be
stored encrypted: MongoDB keeps only the encrypted value and only the application holding the key can
read it. To enable it, once:

```bash
.venv/bin/python -m src.field_encryption setup
```

The command creates a master key in `~/.config/rag-demo/`, outside the project, and a data key in
Atlas. Without this step the demo works and stores the question in clear text. The
**🔒 Come MongoDB conserva i dati** button, below the agent's answer, shows the same document as seen
by the application and as seen by the database.

**OCR and Office documents.** For scanned pages install Tesseract. For visual pages of Word and
PowerPoint files in multimodal mode install LibreOffice. If they are not in your PATH, set their path
in `.env` (section 6).

**Other models.** In `.env` (sections 3 and 4) you can choose another provider, another model, or a
different model for the agent's decisions. The instructions given to the model are in
`prompts/system_prompt.txt`.

---

## 10. Troubleshooting

| Message or symptom | What to do |
|---|---|
| Installation fails while "building" a package (`pymupdf`, `cryptography`) | your Python version is too recent: recreate the environment with `uv venv --python 3.12 .venv` |
| `KeyError: 'MONGODB_URI'` on start | the `.env` file or its `MONGODB_URI` line is missing ([step 3](#3-configuration-the-env-file)) |
| Atlas not reachable | check user and password in the string and that your IP is in *Network Access* |
| "Non ci sono documenti caricati in modalità …" | load documents in that mode, or pick another one in *Dove cercare* |
| "Indici non interrogabili" | Admin → *Verifica e prepara gli indici*, then wait for `READY` |
| "Il modello configurato non è disponibile" | check `LLM_PROVIDER` and the matching key in `.env` |
| "Modalità manual: serve VOYAGE_API_KEY" | add the Voyage key to `.env`, or use the auto mode |
| Error in context mode on a long document | that is the ~32,000-token limit: use auto or manual |
| Pages do not show the latest changes | reload the browser with `Cmd+Shift+R` (`Ctrl+Shift+R` on Windows) |

---

## 11. How it is built

```
server.py               web server (Starlette): pages and API
web/                    interface: index.html, style.css, app.js (plain JavaScript, no framework)
prompts/                instructions for the model
scripts/                download of the 100 FinanceBench PDFs
src/
  config.py             reads the .env file
  ingestion.py          document reading, chunking, storage in MongoDB and GridFS
  indexes.py            index definitions and creation
  retrieval.py          search pipeline: $search + $vectorSearch → $rankFusion → $rerank
  web_services.py       page logic (Query, Upload, Admin, component check)
  agent.py              agentic search engine
  agentic_search.py     planning, evidence grading, targeted searches
  citation_check.py     checks the numbers in the answer against the sources
  agentic_memory.py     sessions and history in MongoDB
  field_encryption.py   client-side field level encryption (CSFLE)
tests/                  automated tests (use neither Atlas nor the models)
```

**Data in MongoDB** (database `rag`):

| Collection | Content |
|---|---|
| `pdf_chunks` | one document per passage: text, metadata, page, mode, vector if any |
| `pdf_files.files`, `pdf_files.chunks` | the original files (GridFS) |
| `agentic_sessions` | conversations with the agent |
| `agentic_runs` | every agentic search, step by step |
| `app_settings` | the last search mode chosen |
| `encryption.__keyVault` | the encryption data key (CSFLE only) |

---

## 12. Data and licences

The documents come from **FinanceBench** by Patronus AI
([repository](https://github.com/patronus-ai/financebench)): public financial documents of
US-listed companies. They are not included in this project; the script downloads them from a
specific version of that repository (list in `scripts/financebench_100.json`). Check FinanceBench's
terms of use before using them outside a demo.
