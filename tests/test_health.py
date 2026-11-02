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
