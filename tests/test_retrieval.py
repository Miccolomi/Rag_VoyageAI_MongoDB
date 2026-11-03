import json

import pytest

from src import config, retrieval
from src.retrieval import build_native_rerank_pipeline, build_pipeline, fusion_weights

VECTORS = {"manual": [0.1] * 4, "context": [0.2] * 4}
MODES = ["auto", "manual", "context"]


def _stage_names(pipeline):
    return [next(iter(stage)) for stage in pipeline]


def test_pesi_bilanciati_testo_uguale_somma_vettoriali(monkeypatch):
    monkeypatch.setattr(config, "HYBRID_TEXT_WEIGHT_MODE", "balanced")
    weights = fusion_weights(["vector_auto", "vector_manual", "vector_context", "text"])
    assert weights == {"vector_auto": 1.0, "vector_manual": 1.0, "vector_context": 1.0, "text": 3.0}


def test_pesi_uguali_se_richiesto(monkeypatch):
    monkeypatch.setattr(config, "HYBRID_TEXT_WEIGHT_MODE", "equal")
    assert fusion_weights(["vector_auto", "text"])["text"] == 1.0


def test_pesi_espliciti_prevalgono():
    assert fusion_weights(["vector", "text"], {"text": 0.5})["text"] == 0.5


def test_pipeline_ibrida_deduplica_prima_della_proiezione():
    pipeline = build_pipeline("capex 3M 2018", top_k=20, embed_mode="all", modes=MODES,
                              query_vectors=VECTORS)
    names = _stage_names(pipeline)
    assert names[0] == "$rankFusion"
    # i metadati di fusione vengono copiati prima di $group
    assert names.index("$set") < names.index("$group")
    assert names[-1] == "$project"
    group = pipeline[names.index("$group")]["$group"]
    assert group["_id"] == {"document_id": "$document_id", "chunk": {"$ifNull": ["$chunk_hash", "$_id"]}}
    project = pipeline[-1]["$project"]
    assert project["score"] == "$_fusion_score"
    assert project["scoreDetails"] == "$_fusion_details"
    branches = pipeline[0]["$rankFusion"]["input"]["pipelines"]
    assert set(branches) == {"vector_auto", "vector_manual", "vector_context", "text"}


def test_ramo_vettoriale_usa_il_campo_della_modalita():
    pipeline = build_pipeline("q", top_k=10, embed_mode="all", modes=MODES, query_vectors=VECTORS)
    branches = pipeline[0]["$rankFusion"]["input"]["pipelines"]
    assert branches["vector_auto"][0]["$vectorSearch"]["path"] == "content"
    assert branches["vector_manual"][0]["$vectorSearch"]["path"] == "embedding"
    assert branches["vector_context"][0]["$vectorSearch"]["path"] == "embedding_context"


def test_rerank_nativo_dopo_la_deduplica():
    pipeline = build_native_rerank_pipeline("q", top_k=30, top_n=8, embed_mode="all",
                                            modes=MODES, query_vectors=VECTORS)
    names = _stage_names(pipeline)
    assert names.index("$group") < names.index("$rerank")
    assert names[-1] == "$limit" and pipeline[-1]["$limit"] == 8
    json.dumps(pipeline)  # serializzabile così com'è, come nella vista mongosh


def test_pipeline_solo_testo_invariata():
    pipeline = build_pipeline("q", top_k=10, embed_mode="all", modes=MODES, retrieval_mode="text")
    assert "$group" not in _stage_names(pipeline)


def test_nessuna_chiamata_embedding_se_i_vettori_sono_forniti(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("embedding non atteso")
    monkeypatch.setattr(retrieval, "embed_query", fail)
    build_pipeline("q", top_k=10, embed_mode="all", modes=MODES, query_vectors=VECTORS)


def test_deduplica_tiene_la_copia_migliore_e_registra_le_modalita():
    import mongomock
    from src.retrieval import dedup_stages
    coll = mongomock.MongoClient()["db"]["c"]
    coll.insert_many([
        {"_id": "a_auto", "document_id": "d", "chunk_hash": "h1", "embedding_mode": "auto", "_fusion_score": 0.02},
        {"_id": "a_manual", "document_id": "d", "chunk_hash": "h1", "embedding_mode": "manual", "_fusion_score": 0.05},
        {"_id": "b_auto", "document_id": "d", "chunk_hash": "h2", "embedding_mode": "auto", "_fusion_score": 0.04},
        {"_id": "c_auto", "document_id": "e", "chunk_hash": "h1", "embedding_mode": "auto", "_fusion_score": 0.01},
    ])
    out = list(coll.aggregate(dedup_stages(10)))
    assert [doc["_id"] for doc in out] == ["a_manual", "b_auto", "c_auto"]
    assert sorted(out[0]["_found_in_modes"]) == ["auto", "manual"]
    assert out[0]["_copies"] == 2
    assert len(list(coll.aggregate(dedup_stages(2)))) == 2


def test_deduplica_disattivabile(monkeypatch):
    monkeypatch.setattr(config, "HYBRID_DEDUP", False)
    pipeline = build_pipeline("q", top_k=10, embed_mode="all", modes=MODES, query_vectors=VECTORS)
    names = _stage_names(pipeline)
    assert "$group" not in names and names[-2:] == ["$limit", "$project"]


def test_filtro_della_modalita_non_ripetuto_nel_ramo_vettoriale():
    from src import retrieval
    pipe = retrieval.build_pipeline("capex", embed_mode="auto")
    stage = pipe[0]["$rankFusion"]["input"]["pipelines"]["vector"][0]["$vectorSearch"]
    assert stage["filter"] == {"embedding_mode": "auto"}
    pipe = retrieval.build_pipeline("capex", embed_mode="auto", filters={"year": 2018})
    stage = pipe[0]["$rankFusion"]["input"]["pipelines"]["vector"][0]["$vectorSearch"]
    assert stage["filter"] == {"$and": [{"year": 2018}, {"embedding_mode": "auto"}]}


def test_pesi_uguali_con_piu_modalita(monkeypatch):
    # Con "balanced" il testo a peso 2 escludeva tutti i risultati semantici dai primi k.
    monkeypatch.setattr(config, "HYBRID_TEXT_WEIGHT_MODE", "equal")
    assert set(fusion_weights(["vector_auto", "vector_manual", "text"]).values()) == {1.0}


def test_cursore_semantico_diventa_pesi():
    from src.retrieval import semantic_weights
    assert semantic_weights(None) is None and semantic_weights(50) is None
    assert semantic_weights(0) == {"text": 2.0, "vector": 0.0}
    assert semantic_weights(80) == {"text": 0.4, "vector": 1.6}
    with pytest.raises(ValueError):
        semantic_weights(120)


def test_pesi_passati_a_rankfusion_anche_con_rerank_nativo():
    from src.retrieval import semantic_weights
    pipeline = build_native_rerank_pipeline("capex", top_k=10, top_n=5, embed_mode="all", modes=["auto", "manual"],
                                            query_vectors={"manual": [0.1] * 4}, weights=semantic_weights(20))
    weights = pipeline[0]["$rankFusion"]["combination"]["weights"]
    assert weights == {"vector_auto": 0.4, "vector_manual": 0.4, "text": 1.6}
