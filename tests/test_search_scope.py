import threading
import time

from src import search_scope


def test_vettori_della_domanda_calcolati_in_parallelo(monkeypatch):
    active, peak = 0, 0
    lock = threading.Lock()

    def slow_vector(query, mode):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.2)
        with lock:
            active -= 1
        return [len(mode)]

    monkeypatch.setattr(search_scope, "embed_query", slow_vector)
    started = time.perf_counter()
    vectors = search_scope.prepare_vectors("q", ["auto", "manual", "context", "multimodal"])
    elapsed = time.perf_counter() - started
    assert set(vectors) == {"manual", "context", "multimodal"}
    assert peak == 3
    assert elapsed < 0.5


def test_didascalia_pesi():
    assert "peso 3" in search_scope.weights_caption(["auto", "manual", "context"])
