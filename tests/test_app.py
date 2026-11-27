import base64
import hashlib
import hmac
import json

import pytest
from fastapi.testclient import TestClient

from src import main
from src.cleaner import ThaiTextCleaner


def test_cleaner_keeps_negation():
    tokens = ThaiTextCleaner().clean("สินค้าไม่ดีเลย!!! http://x.co")
    assert "ไม่" in tokens
    assert not any("http" in t for t in tokens)


def test_cleaner_collapses_repeats():
    assert "ก" * 3 not in "".join(ThaiTextCleaner().clean("ดีมากกกกก"))


@pytest.fixture
def client():
    with TestClient(main.app) as c:
        yield c


def test_analyze_validation(client):
    assert client.post("/analyze", json={"text": ""}).status_code == 422
    assert client.post("/analyze", json={"text": "   "}).status_code == 400


def test_analyze_no_model_503(client, monkeypatch):
    monkeypatch.setattr(main, "predictor", None)
    assert client.post("/analyze", json={"text": "ดี"}).status_code == 503


def test_webhook_signature(client, monkeypatch):
    monkeypatch.setattr(main, "LINE_ACCESS_TOKEN", "tok")
    monkeypatch.setattr(main, "LINE_CHANNEL_SECRET", "secret")
    body = json.dumps({"events": []}).encode()

    assert client.post("/webhook", content=body).status_code == 401
    assert client.post("/webhook", content=body, headers={"x-line-signature": "bad"}).status_code == 401

    sig = base64.b64encode(hmac.new(b"secret", body, hashlib.sha256).digest()).decode()
    r = client.post("/webhook", content=body, headers={"x-line-signature": sig})
    assert r.status_code == 200




def test_webhook_stores_message(client, monkeypatch, tmp_path):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setattr(main, "LINE_ACCESS_TOKEN", "tok")
    monkeypatch.setattr(main, "LINE_CHANNEL_SECRET", "secret")
    monkeypatch.setattr(main, "send_line_reply", lambda *a: None)

    class P:
        def predict(self, t):
            return {"sentiment": "negative", "confidence": 0.9, "label": 0, "aspects": ["delay"], "tokens": ["ช้า"]}

    monkeypatch.setattr(main, "predictor", P())
    body = json.dumps({"events": [{"type": "message", "replyToken": "r", "timestamp": 1793500000000,
                                   "message": {"type": "text", "id": "m1", "text": "ส่งช้า"}}]}).encode()
    sig = base64.b64encode(hmac.new(b"secret", body, hashlib.sha256).digest()).decode()
    assert client.post("/webhook", content=body, headers={"x-line-signature": sig}).status_code == 200
    # same message id again -> not duplicated
    client.post("/webhook", content=body, headers={"x-line-signature": sig})
    rep = client.get("/stats/daily").json()
    assert len(rep) == 1 and rep[0]["total"] == 1 and rep[0]["top_complaints"] == {"delay": 1}
