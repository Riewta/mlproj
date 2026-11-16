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
    monkeypatch.setitem(main.state, "model", None)
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
