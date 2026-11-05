import pytest
from fastapi.testclient import TestClient

from hallucination_lab.app import app

client = TestClient(app)


def test_ui_and_health():
    assert client.get("/health").json()["status"] == "ok"
    page = client.get("/")
    assert page.status_code == 200 and 'lang="vi"' in page.text
    assert client.get("/static/app.js").status_code == 200
    methods = client.get("/methods").json()["methods"]
    assert len(methods) == 7


@pytest.mark.parametrize(
    "claim,context,expected",
    [
        ("Việt Nam gia nhập ASEAN năm 2000.", "Việt Nam gia nhập ASEAN năm 1995.", True),
        ("Hà Nội là thủ đô của Việt Nam.", "Hà Nội là thủ đô của Việt Nam.", False),
        ("The sky is not blue.", "The sky is blue.", True),
        ("There are 999 students.", "There are 100 students.", True),
    ],
)
def test_baseline_functional(claim, context, expected):
    response = client.post("/detect", json={"claim": claim, "context": context})
    assert response.status_code == 200
    data = response.json()
    assert data["is_hallucination"] is expected
    assert data["confidence_kind"] == "heuristic"
    assert data["findings"]
    assert all(e["quote"] in context for e in data["evidence"])


def test_missing_evidence_abstains():
    data = client.post("/detect", json={"claim": "A plausible statement."}).json()
    assert data["is_hallucination"] is None
    assert data["score"] is None and data["confidence"] is None


@pytest.mark.parametrize(
    "payload",
    [
        {"claim": " "},
        {"claim": "...!!!"},
        {"claim": "x", "method": "unknown"},
        {"claim": "x", "threshold": -1},
        {"claim": "x", "threshold": 1.1},
        {"claim": "x", "samples": [" "]},
        {"claim": "x", "extra": "ignored?"},
        {"claim": "x" * 6001},
    ],
)
def test_request_validation(payload):
    assert client.post("/detect", json=payload).status_code == 422


def test_llm_not_configured_is_error_not_fake_score(monkeypatch):
    monkeypatch.delenv("JUDGE_BASE_URL", raising=False)
    monkeypatch.delenv("JUDGE_MODEL", raising=False)
    response = client.post("/detect", json={"claim": "x", "context": "y", "method": "llm_judge"})
    assert response.status_code == 503
    assert "JUDGE_BASE_URL" in response.json()["detail"]


def test_threshold_changes_decision_not_raw_score():
    payload = {"claim": "A different fact.", "context": "An unrelated source."}
    low = client.post("/detect", json=dict(payload, threshold=0.5)).json()
    high = client.post("/detect", json=dict(payload, threshold=0.9)).json()
    assert low["is_hallucination"] is True and high["is_hallucination"] is False
    assert low["score"] == high["score"]
    assert high["confidence"] == pytest.approx(1 - high["score"])


@pytest.mark.parametrize("method", ["nli", "hhem"])
def test_missing_local_models_return_configuration_error(monkeypatch, method):
    monkeypatch.setenv("ENABLE_LOCAL_MODELS", "0")
    response = client.post("/detect", json={"claim": "Claim.", "context": "Source.", "method": method})
    assert response.status_code == 503


def test_punctuation_context_is_not_evidence():
    response = client.post("/detect", json={"claim": "A fact.", "context": "...!!!"})
    assert response.status_code == 200
    assert response.json()["is_hallucination"] is None
