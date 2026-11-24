"""Train -> export bundle -> serve with both runtimes -> call the API."""
import numpy as np
import pytest

pytest.importorskip("torch")


def test_torch_and_onnx_agree(bundle, datasets):
    from trajclf.predictor import Predictor

    x = datasets["X_test"][:8]
    a = Predictor(bundle, "torch").logits(x)
    b = Predictor(bundle, "onnx").logits(x)
    assert np.allclose(a, b, atol=1e-4)


def test_api_predict(bundle, monkeypatch):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    monkeypatch.setenv("MODEL_DIR", str(bundle))
    monkeypatch.setenv("MODEL_RUNTIME", "torch")
    from trajclf.api.main import app

    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        info = client.get("/model").json()
        ride = {"points": np.random.rand(300, len(info["features"])).tolist()}
        r = client.post("/predict", json=ride)
        assert r.status_code == 200
        body = r.json()
        assert body["label"] in info["labels"]
        assert abs(sum(body["probabilities"].values()) - 1) < 1e-4
        assert client.post("/predict", json={"points": [[1.0], [2.0]]}).status_code == 422
        assert client.post("/predict/batch", json={"rides": [ride, ride]}).status_code == 200
        assert "api_requests_total" in client.get("/metrics").text
