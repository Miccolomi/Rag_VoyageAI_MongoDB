"""Cifratura CSFLE della prima domanda: offline, con un cifratore finto al posto di libmongocrypt."""
import json

import mongomock
import pytest
from bson.binary import Binary
from starlette.testclient import TestClient

from src import agentic_memory, config, field_encryption, search_scope


class FakeClientEncryption:
    """Come ClientEncryption: restituisce Binary di sottotipo 6 e li decifra."""

    def __init__(self):
        self.plain: dict[bytes, str] = {}

    def encrypt(self, value, algorithm, key_alt_name=None):
        assert algorithm.endswith("Random") and key_alt_name == config.CSFLE_KEY_ALT_NAME
        cipher = Binary(b"\x02" + str(len(self.plain)).encode() + b"cifrato" * 8, 6)
        self.plain[bytes(cipher)] = value
        return cipher

    def decrypt(self, value):
        return self.plain[bytes(value)]


@pytest.fixture()
def fake_crypto(monkeypatch, tmp_path):
    key = tmp_path / "chiave.bin"
    key.write_bytes(b"k" * 96)
    monkeypatch.setattr(config, "CSFLE_MASTER_KEY_PATH", key)
    fake = FakeClientEncryption()
    monkeypatch.setattr(field_encryption, "client_encryption", lambda: fake)
    field_encryption._KEY_READY.set()
    yield fake
    field_encryption._KEY_READY.clear()


def test_senza_chiave_resta_in_chiaro():
    assert not field_encryption.is_configured()
    assert field_encryption.encrypt_text("ciao") == "ciao"
    assert field_encryption.stored_view("ciao") == {"encrypted": False, "value": "ciao"}


def test_cifra_e_decifra(fake_crypto):
    cipher = field_encryption.encrypt_text("Capex 3M 2018?")
    assert field_encryption.is_encrypted(cipher)
    assert field_encryption.decrypt_text(cipher) == "Capex 3M 2018?"
    view = field_encryption.stored_view(cipher)
    assert view["encrypted"] and view["subtype"] == 6 and view["bytes"] == len(cipher)
    assert "Capex" not in json.dumps(view)


def test_chiave_mancante_non_rompe_la_lettura(fake_crypto, monkeypatch):
    cipher = field_encryption.encrypt_text("segreto")
    monkeypatch.setattr(field_encryption, "client_encryption", lambda: (_ for _ in ()).throw(RuntimeError("no")))
    assert field_encryption.decrypt_text(cipher) is None


def test_chiave_master_creata_una_volta(monkeypatch, tmp_path):
    path = tmp_path / "sub" / "master.bin"
    monkeypatch.setattr(config, "CSFLE_MASTER_KEY_PATH", path)
    assert field_encryption.create_master_key() is True
    assert len(path.read_bytes()) == 96 and (path.stat().st_mode & 0o777) == 0o600
    first = path.read_bytes()
    assert field_encryption.create_master_key() is False and path.read_bytes() == first


def test_prima_domanda_salvata_cifrata(fake_crypto):
    db = mongomock.MongoClient()["rag"]
    sessions, runs = db["agentic_sessions"], db["agentic_runs"]
    sid = agentic_memory.create_session(sessions)["_id"]
    for q in ("Capex 3M 2018?", "E nel 2017?"):
        run_id = agentic_memory.start_run(runs, sessions, sid, q)
        agentic_memory.finish_run(runs, sessions, session_id=sid, run_id=run_id, question=q,
                                  answer="ok", sources=[], status="complete")
    stored = sessions.find_one({"_id": sid})["first_question"]
    assert field_encryption.is_encrypted(stored)
    assert field_encryption.decrypt_text(stored) == "Capex 3M 2018?"  # resta la prima


def test_se_la_cifratura_fallisce_la_risposta_si_salva(fake_crypto, monkeypatch):
    monkeypatch.setattr(field_encryption, "encrypt_text", lambda v: (_ for _ in ()).throw(RuntimeError("kms")))
    db = mongomock.MongoClient()["rag"]
    sessions, runs = db["agentic_sessions"], db["agentic_runs"]
    sid = agentic_memory.create_session(sessions)["_id"]
    run_id = agentic_memory.start_run(runs, sessions, sid, "q")
    agentic_memory.finish_run(runs, sessions, session_id=sid, run_id=run_id, question="q",
                              answer="ok", sources=[], status="complete")
    doc = sessions.find_one({"_id": sid})
    assert doc["first_question"] == "q" and doc["memory"]["turn_count"] == 1
    assert doc["encryption_error"].startswith("RuntimeError: kms")


def test_api_elenco_e_vista_protezione(fake_crypto, monkeypatch):
    db = mongomock.MongoClient()["rag"]
    monkeypatch.setattr(config, "get_db", lambda: db)
    monkeypatch.setattr(search_scope, "load_active_mode", lambda default=None: "auto")
    cipher = field_encryption.encrypt_text("Capex 3M 2018?")
    db["agentic_sessions"].insert_one({"_id": "s1", "title": "Sessione", "first_question": cipher,
                                       "memory": {"turn_count": 1}})
    import server
    c = TestClient(server.app)
    row = c.get("/api/sessions").json()[0]
    assert row["first_question"] == "Capex 3M 2018?" and row["encrypted"] is True
    view = c.get("/api/sessions/s1/protection").json()
    assert view["app_view"] == "Capex 3M 2018?" and view["encrypted"] is True
    assert view["stored_view"]["subtype"] == 6 and "Capex" not in json.dumps(view["stored_view"])
    assert c.get("/api/sessions/nessuna/protection").status_code == 404


def test_api_ultima_sessione_cifrata(fake_crypto, monkeypatch):
    db = mongomock.MongoClient()["rag"]
    monkeypatch.setattr(config, "get_db", lambda: db)
    import server
    c = TestClient(server.app)
    assert c.get("/api/protection/latest").json()["empty"] is True
    db["agentic_sessions"].insert_many([
        {"_id": "vecchia", "first_question": "in chiaro", "created_at": 2},
        {"_id": "cifrata", "first_question": field_encryption.encrypt_text("Capex?"), "created_at": 1}])
    view = c.get("/api/protection/latest").json()
    assert view["id"] == "cifrata" and view["app_view"] == "Capex?"


def test_senza_chiave_il_motivo_resta_nella_sessione():
    db = mongomock.MongoClient()["rag"]
    sessions, runs = db["agentic_sessions"], db["agentic_runs"]
    sid = agentic_memory.create_session(sessions)["_id"]
    run_id = agentic_memory.start_run(runs, sessions, sid, "q")
    agentic_memory.finish_run(runs, sessions, session_id=sid, run_id=run_id, question="q",
                              answer="ok", sources=[], status="complete")
    doc = sessions.find_one({"_id": sid})
    assert doc["first_question"] == "q" and "Chiave master" in doc["encryption_error"]
