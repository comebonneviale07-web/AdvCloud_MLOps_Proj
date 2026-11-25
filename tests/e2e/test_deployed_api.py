"""End-to-end test against a running API (docker compose or Kubernetes).

API_URL=http://localhost:8000 pytest -m e2e tests/e2e
"""
import json
import os
import urllib.request

import pytest

API_URL = os.environ.get("API_URL")
pytestmark = [pytest.mark.e2e, pytest.mark.skipif(not API_URL, reason="API_URL not set")]


def _get(path):
    with urllib.request.urlopen(f"{API_URL}{path}", timeout=10) as r:
        return json.loads(r.read())


def test_health_and_prediction():
    assert _get("/health")["status"] == "ok"
    info = _get("/model")
    body = json.dumps({"points": [[0.1] * len(info["features"])] * 2000}).encode()
    req = urllib.request.Request(f"{API_URL}/predict", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        assert json.loads(r.read())["label"] in info["labels"]
