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
