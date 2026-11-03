"""Verifica dei componenti: ogni controllo chiama davvero il componente."""
import mongomock

from src import config, search_scope, web_services


def test_verifica_componenti(monkeypatch):
    db = mongomock.MongoClient()["rag"]
    monkeypatch.setattr(config, "get_db", lambda: db)
    monkeypatch.setattr(config, "get_collection", lambda: db["pdf_chunks"])
    monkeypatch.setattr(search_scope, "load_active_mode", lambda default=None: "auto")
    monkeypatch.setattr(db, "command", lambda *a, **k: {"ok": 1}, raising=False)
    import src.generation as gen
    monkeypatch.setattr(gen, "generate_answer", lambda q, c: (_ for _ in ()).throw(RuntimeError("RBAC: access denied")))
    web_services._HEALTH.clear()
    out = web_services.health(force=True)
    names = {c["name"]: c for c in out["checks"]}
    assert out["ok"] is False
    assert names["Documenti"]["ok"] is False and "RBAC" in names["Modello della risposta"]["detail"]


def test_errori_mostrati_senza_indirizzi_web():
    exc = RuntimeError("Command failed. See https://www.mongodb.com/docs/x for details; anche www.esempio.it")
    message = web_services.public_error(exc)
    assert message.startswith("RuntimeError: Command failed.")
    assert "http" not in message and "www." not in message


def test_flusso_ndjson_toglie_gli_indirizzi_dagli_errori():
    import asyncio, json
    import server

    def events():
        yield {"type": "error", "message": "Atlas: vedi https://example.com/errore"}
        raise RuntimeError("rotto, vedi http://example.com")

    async def collect():
        return [chunk async for chunk in server._stream(events()).body_iterator]
    lines = [json.loads(line) for line in asyncio.run(collect())]
    assert all("http" not in e["message"] for e in lines) and len(lines) == 2


def test_static_files_are_revalidated():
    from starlette.testclient import TestClient
    import server
    response = TestClient(server.app).get("/web/app.js")
    assert response.status_code == 200 and response.headers["cache-control"] == "no-cache"


def test_query_rifiuta_modalita_sconosciuta():
    from starlette.testclient import TestClient
    import server
    response = TestClient(server.app).post("/api/query", json={"question": "capex?", "mode": "xyz"})
    assert response.status_code == 400 and "sconosciuta" in response.json()["error"]


def test_upload_non_cambia_la_modalita_di_ricerca(monkeypatch):
    from src import ingestion, search_scope, web_services
    monkeypatch.setattr(search_scope, "save_active_mode", lambda mode: (_ for _ in ()).throw(AssertionError("non deve")))
    monkeypatch.setattr(web_services.config, "get_db", lambda: None)
    monkeypatch.setattr(web_services.config, "get_collection", lambda: None)
    monkeypatch.setattr(ingestion, "ingest_pdf", lambda *a, **k: {"chunks_inserted": 3, "pages_processed": 1})
    events = list(web_services.ingest_files([("a.pdf", b"x")], mode="manual", metadata={}))
    assert events[-2]["outcome"] == "inserted"


def test_elimina_documento_solo_da_una_modalita(monkeypatch):
    import mongomock
    from src import web_services
    coll = mongomock.MongoClient()["rag"]["pdf_chunks"]
    coll.insert_many([{"document_id": "d1", "embedding_mode": "auto", "filename": "a.pdf"},
                      {"document_id": "d1", "embedding_mode": "manual", "filename": "a.pdf"},
                      {"document_id": "d1", "embedding_mode": "manual", "filename": "a.pdf"}])
    monkeypatch.setattr(web_services.config, "get_collection", lambda: coll)
    docs = web_services.documents()
    assert {m["mode"]: m["chunks"] for m in docs[0]["by_mode"]} == {"auto": 1, "manual": 2}
    assert web_services.delete_document("d1", "manual") == {"chunks_deleted": 2}
    assert coll.count_documents({"document_id": "d1", "embedding_mode": "auto"}) == 1
