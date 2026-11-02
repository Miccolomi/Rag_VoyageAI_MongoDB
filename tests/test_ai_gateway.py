import httpx
import pytest

from src import ai_gateway, config


class _Resp:
    def __init__(self, status, payload=None, text=""):
        self.status_code, self._payload, self.text = status, payload, text or str(payload)

    def json(self):
        if self._payload is None:
            raise ValueError
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("x", request=None, response=self)


@pytest.fixture(autouse=True)
def keys(monkeypatch):
    monkeypatch.setattr(config, "AI_GATEWAY_API_KEY", "k1")
    monkeypatch.setattr(config, "AI_GATEWAY_API_KEY_2", "k2")
    ai_gateway._UNSUPPORTED.clear()


OK = {"status": "completed", "model": "modello-test",
      "output": [{"type": "reasoning"}, {"type": "message", "content": [{"type": "output_text", "text": "ok [#1]"}]}],
      "usage": {"input_tokens": 10, "output_tokens": 3}}


def test_risposta_letta_dal_formato_responses(monkeypatch):
    monkeypatch.setattr(ai_gateway.httpx, "post", lambda *a, **k: _Resp(200, OK))
    out = ai_gateway.respond("s", "u", model="modello-test", max_tokens=100)
    assert out["answer"] == "ok [#1]" and not out["truncated"]
    assert out["usage"] == {"input_tokens": 10, "output_tokens": 3}


def test_parametro_rifiutato_viene_tolto_e_ricordato(monkeypatch):
    sent = []

    def post(url, json=None, **k):
        sent.append(json)
        if "temperature" in json:
            return _Resp(400, {"error": {"message": "Unsupported parameter: 'temperature'", "param": "temperature"}})
        return _Resp(200, OK)
    monkeypatch.setattr(ai_gateway.httpx, "post", post)
    ai_gateway.respond("s", "u", model="m", max_tokens=100, json_mode=True)
    assert "temperature" in sent[0] and "temperature" not in sent[1]
    assert sent[1]["text"] == {"format": {"type": "json_object"}}
    ai_gateway.respond("s", "u", model="m", max_tokens=100)
    assert "temperature" not in sent[-1]   # non lo si rimanda più


def test_seconda_chiave_se_la_prima_e_rifiutata(monkeypatch):
    used = []

    def post(url, json=None, headers=None, **k):
        used.append(headers["Authorization"])
        return _Resp(401, {"error": {"message": "bad key"}}) if headers["Authorization"].endswith("k1") else _Resp(200, OK)
    monkeypatch.setattr(ai_gateway.httpx, "post", post)
    assert ai_gateway.respond("s", "u", model="m", max_tokens=10)["answer"] == "ok [#1]"
    assert used == ["Bearer k1", "Bearer k2"]


def test_rete_aziendale_mancante_messaggio_chiaro(monkeypatch):
    monkeypatch.setattr(ai_gateway.httpx, "post", lambda *a, **k: _Resp(403, None, "RBAC: access denied"))
    with pytest.raises(RuntimeError, match="VPN"):
        ai_gateway.respond("s", "u", model="m", max_tokens=10)


def test_risposta_troncata(monkeypatch):
    monkeypatch.setattr(ai_gateway.httpx, "post", lambda *a, **k: _Resp(200, {
        **OK, "status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}))
    assert ai_gateway.respond("s", "u", model="m", max_tokens=10)["truncated"]
