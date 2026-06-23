# RAG with MongoDB Atlas and Voyage AI

[Italiano](README.it.md)

An end-to-end demonstration application for ingesting documents, retrieving
their content through hybrid search, and generating grounded answers from
sources stored in MongoDB Atlas.

## What It Does

- Ingests PDF and Office documents and extracts text and useful metadata.
- Splits documents into chunks while preserving page and content-type data.
- Supports MongoDB Atlas-managed embeddings, application-generated embeddings,
  and contextualized embeddings with Voyage AI.
- Performs keyword, semantic, or hybrid search.
- Combines result rankings with `$rankFusion`.
- Applies metadata filters directly within the retrieval pipelines.
- Reorders candidates using Voyage AI reranking models.
- Generates grounded answers using only the retrieved passages.
- Keeps in-text citations separate from downloadable original documents.
- Shows the contribution of each retrieval stage on the Query page.

## Architecture

The same data platform manages documents, metadata, text search, vector search,
filters, and rank fusion. The application uses Voyage AI for embeddings and
reranking, together with a configurable LLM provider that composes the final
answer from the selected passages.

The main pipeline is:

1. document extraction and normalization;
2. chunking and metadata enrichment;
3. embedding generation or automated embedding management;
4. keyword and semantic retrieval;
5. rank fusion and reranking;
6. context construction;
7. grounded answer generation and access to original sources.

## Run Locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

Configure `.env` with your MongoDB Atlas connection and the API keys for the
providers you intend to use. The `.env` file, virtual environments, and locally
uploaded documents are excluded from the repository.

## Project Structure

- `app.py`: Streamlit interface and demo orchestration.
- `src/`: ingestion, embeddings, retrieval, reranking, and configuration.
- `schemas/`: Atlas Search and Vector Search index definitions.
- `prompts/`: system prompt used for grounded generation.
- `tests/`: pipeline tests.
- `scripts/`: diagnostic tools for configuration and models.
