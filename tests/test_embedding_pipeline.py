from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from src import config
from src.embeddings import embed_voyage, embed_voyage_contextualized
from src.ingestion import ingest_pdf
from src.retrieval import build_native_rerank_pipeline, build_pipeline
from src.voyage_http import post_json


class _FakeCollection:
    def __init__(self):
        self.inserted: list[dict] = []

    def count_documents(self, *_args, **_kwargs):
        return 0

    def insert_many(self, docs):
        self.inserted.extend(docs)


def _metadata() -> dict:
    return {
        "title": "Documento test",
        "author": None,
        "subject": None,
        "keywords": None,
        "creator": None,
        "producer": None,
        "creation_date": None,
        "modification_date": None,
        "total_pages": 2,
        "format": "PDF 1.7",
    }


class EmbeddingClientTests(unittest.TestCase):
    @patch("src.embeddings.post_json")
    def test_manual_embedding_sends_explicit_dimensions(self, post_json):
        post_json.return_value = {
            "data": [
                {"index": 0, "embedding": [0.1] * config.VOYAGE_EMBED_DIMENSIONS},
                {"index": 1, "embedding": [0.2] * config.VOYAGE_EMBED_DIMENSIONS},
            ]
        }

        result = embed_voyage(["uno", "due"], input_type="document")

        self.assertEqual(len(result), 2)
        payload = post_json.call_args.kwargs["payload"]
        self.assertEqual(payload["output_dimension"], config.VOYAGE_EMBED_DIMENSIONS)
        self.assertEqual(payload["input_type"], "document")
        self.assertFalse(payload["truncation"])

    @patch("src.embeddings.post_json")
    def test_context_embedding_preserves_prechunked_shape(self, post_json):
        post_json.return_value = {
            "data": [{
                "index": 0,
                "data": [
                    {"index": 0, "embedding": [0.1] * config.VOYAGE_CONTEXT_DIMENSIONS},
                    {"index": 1, "embedding": [0.2] * config.VOYAGE_CONTEXT_DIMENSIONS},
                ],
            }]
        }

        result = embed_voyage_contextualized(
            [["chunk pagina uno", "chunk pagina due"]],
            input_type="document",
        )

        self.assertEqual(len(result[0]), 2)
        payload = post_json.call_args.kwargs["payload"]
        self.assertEqual(
            payload["inputs"],
            [["chunk pagina uno", "chunk pagina due"]],
        )
        self.assertEqual(payload["output_dimension"], config.VOYAGE_CONTEXT_DIMENSIONS)
        self.assertNotIn("enable_auto_chunking", payload)


class ContextIngestionTests(unittest.TestCase):
    @patch("src.ingestion.embed_voyage_contextualized")
    @patch("src.ingestion.parse_document")
    def test_context_mode_keeps_page_metadata(
        self,
        parse_document,
        contextualized_embed,
    ):
        parse_document.return_value = (
            _metadata(),
            [
                {
                    "page_number": 1,
                    "text": "Contenuto significativo della prima pagina.",
                    "tables": [],
                    "content_type": "text",
                },
                {
                    "page_number": 2,
                    "text": "Contenuto significativo della seconda pagina.",
                    "tables": [],
                    "content_type": "text",
                },
            ],
        )
        contextualized_embed.return_value = [[
            [0.1] * config.VOYAGE_CONTEXT_DIMENSIONS,
            [0.2] * config.VOYAGE_CONTEXT_DIMENSIONS,
        ]]
        collection = _FakeCollection()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pdf"
            path.write_bytes(b"fake-pdf")
            summary = ingest_pdf(
                path,
                db=object(),
                collection=collection,
                storage="s3",
                mode="context",
            )

        self.assertEqual(summary["chunks_inserted"], 2)
        self.assertEqual(
            [doc["page_number"] for doc in collection.inserted],
            [1, 2],
        )
        self.assertTrue(all(
            doc["context_strategy"] == "page_aware_contextualized_chunks"
            for doc in collection.inserted
        ))
        self.assertTrue(all(
            doc["embedding_dimensions"] == config.VOYAGE_CONTEXT_DIMENSIONS
            for doc in collection.inserted
        ))
        self.assertTrue(all(
            len(doc["embedding"]) == config.VOYAGE_CONTEXT_DIMENSIONS
            for doc in collection.inserted
        ))


class VoyageRetryTests(unittest.TestCase):
    @patch("src.voyage_http.time.sleep")
    @patch("src.voyage_http.httpx.Client")
    def test_retries_temporary_503(self, client_cls, sleep):
        request = httpx.Request("POST", "https://ai.mongodb.com/v1/embeddings")
        unavailable = httpx.Response(503, request=request)
        success = httpx.Response(
            200,
            request=request,
            json={"data": [{"index": 0, "embedding": [0.1]}]},
        )
        client = client_cls.return_value.__enter__.return_value
        client.post.side_effect = [unavailable, success]

        result = post_json(
            str(request.url),
            payload={"input": ["test"]},
            headers={"Authorization": "Bearer test"},
            timeout=1.0,
            operation="test",
        )

        self.assertEqual(result["data"][0]["embedding"], [0.1])
        self.assertEqual(client.post.call_count, 2)
        sleep.assert_called_once()


class RetrievalProjectionTests(unittest.TestCase):
    def test_hybrid_projection_contains_only_runtime_fields(self):
        pipeline = build_pipeline(
            "domanda",
            top_k=5,
            embed_mode="context",
            retrieval_mode="hybrid",
            query_vector=[0.1] * config.VOYAGE_CONTEXT_DIMENSIONS,
        )
        projection = pipeline[-1]["$project"]

        self.assertEqual(
            set(projection),
            {
                "document_id",
                "filename",
                "page_number",
                "chunk_index",
                "content_type",
                "content",
                "title",
                "category",
                "gridfs_file_id",
                "score",
                "scoreDetails",
            },
        )
        self.assertNotIn("embedding_model", projection)
        self.assertNotIn("ingested_at", projection)

    def test_native_rerank_follows_rank_fusion(self):
        pipeline = build_native_rerank_pipeline(
            "domanda",
            top_k=50,
            top_n=8,
            embed_mode="context",
            query_vector=[0.1] * config.VOYAGE_CONTEXT_DIMENSIONS,
        )

        stage_names = [next(iter(stage)) for stage in pipeline]
        self.assertIn("$rankFusion", stage_names)
        self.assertIn("$rerank", stage_names)
        self.assertGreater(
            stage_names.index("$rerank"),
            stage_names.index("$rankFusion"),
        )
        rerank_stage = next(stage["$rerank"] for stage in pipeline if "$rerank" in stage)
        self.assertEqual(rerank_stage["model"], config.VOYAGE_RERANK_MODEL)
        self.assertEqual(rerank_stage["numDocsToRerank"], 50)
        self.assertEqual(rerank_stage["path"], "content")


if __name__ == "__main__":
    unittest.main()
