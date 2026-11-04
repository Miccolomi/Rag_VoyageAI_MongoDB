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
    monkeypatch.setattr(config, "AI_GATEWAY_BASE_URL", "https://gateway.example/v1")
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


def test_400_senza_parametro_riprova_con_richiesta_minima(monkeypatch):
    sent = []

    def post(url, json=None, **k):
        sent.append(json)
        if any(name in json for name in ("temperature", "reasoning", "text")):
            return _Resp(400, {"error": {"message": "Invalid request for this model"}})
        return _Resp(200, OK)
    monkeypatch.setattr(ai_gateway.httpx, "post", post)
    assert ai_gateway.respond("s", "u", model="m", max_tokens=10, json_mode=True)["answer"] == "ok [#1]"
    assert not any(name in sent[-1] for name in ("temperature", "reasoning", "text"))


def test_400_mostra_il_motivo_del_gateway(monkeypatch):
    monkeypatch.setattr(ai_gateway.httpx, "post", lambda *a, **k: _Resp(400, {"error": {"message": "The model gpt-x does not exist"}}))
    with pytest.raises(RuntimeError, match="does not exist"):
        ai_gateway.respond("s", "u", model="gpt-x", max_tokens=10)


def test_parametro_citato_senza_virgolette(monkeypatch):
    sent = []

    def post(url, json=None, **k):
        sent.append(json)
        if "reasoning" in json:
            return _Resp(400, {"error": {"message": "reasoning.effort is not supported with this model"}})
        return _Resp(200, OK)
    monkeypatch.setattr(ai_gateway.httpx, "post", post)
    ai_gateway.respond("s", "u", model="m", max_tokens=10)
    assert "reasoning" not in sent[-1] and "temperature" in sent[-1]


def test_openai_usa_indirizzo_e_chiave_di_openai(monkeypatch):
    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(config, "OPENAI_BASE_URL", "https://api.openai.example/v1")
    seen = {}

    def post(url, json=None, headers=None, timeout=None):
        seen.update(url=url, auth=headers["Authorization"], model=json["model"])
        return _Resp(200, OK)

    monkeypatch.setattr(ai_gateway.httpx, "post", post)
    out = ai_gateway.respond("s", "u", model="gpt-5-mini", max_tokens=10, provider="openai")
    assert out["answer"] == "ok [#1]"
    assert seen == {"url": "https://api.openai.example/v1/responses", "auth": "Bearer sk-test", "model": "gpt-5-mini"}


def test_openai_senza_chiave_lo_dice(monkeypatch):
    monkeypatch.setattr(config, "OPENAI_API_KEY", "")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        ai_gateway.respond("s", "u", model="m", max_tokens=10, provider="openai")


def test_openai_errore_con_il_suo_nome(monkeypatch):
    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(ai_gateway.httpx, "post", lambda *a, **k: _Resp(401, {"error": {"message": "Incorrect API key"}}))
    with pytest.raises(RuntimeError, match="OpenAI ha rifiutato.*Incorrect API key"):
        ai_gateway.respond("s", "u", model="m", max_tokens=10, provider="openai")


def test_generazione_con_provider_openai(monkeypatch):
    from src import generation
    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(config, "OPENAI_MODEL", "gpt-5-mini")
    calls = []
    monkeypatch.setattr(ai_gateway.httpx, "post", lambda url, **k: calls.append(url) or _Resp(200, OK))
    out = generation.generate_answer("domanda", [{"content": "testo", "filename": "a.pdf", "page_number": 1}])
    assert out["provider"] == "openai" and out["answer"] == "ok [#1]"
    assert calls[0].endswith("/responses") and generation.llm_available()
