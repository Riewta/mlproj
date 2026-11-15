import asyncio
import base64
import hashlib
import hmac
import logging
import os
from contextlib import asynccontextmanager

import joblib
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from src.cleaner import ThaiTextCleaner

load_dotenv()
logger = logging.getLogger("sentiment")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

LINE_ACCESS_TOKEN = os.getenv("LINE_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
LINE_REPLY_URL = "https://api.line.me/v2/bot/message/reply"

state: dict = {"model": None, "tfidf": None, "cleaner": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state["model"] = joblib.load(os.path.join(MODEL_DIR, "sentiment_model.pkl"))
        state["tfidf"] = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
        state["cleaner"] = ThaiTextCleaner()
        logger.info("Models loaded successfully.")
    except Exception as e:
        logger.warning("Could not load models (%s). Train first: python -m src.train", e)
    yield


app = FastAPI(title="Thai Sentiment API", lifespan=lifespan)


def predict_sentiment(text: str) -> dict:
    model, tfidf, cleaner = state["model"], state["tfidf"], state["cleaner"]
    if model is None or tfidf is None or cleaner is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Run: python -m src.train")
    cleaned = " ".join(cleaner.clean(text))
    vec = tfidf.transform([cleaned])
    proba = model.predict_proba(vec)[0]
    label = int(model.classes_[proba.argmax()])
    return {
        "sentiment": "positive" if label == 1 else "negative",
        "confidence": round(float(proba.max()), 4),
        "label": label,
    }


def verify_line_signature(body: bytes, signature: str) -> bool:
    digest = hmac.new(LINE_CHANNEL_SECRET.encode(), body, hashlib.sha256).digest()
    return hmac.compare_digest(base64.b64encode(digest).decode(), signature)


def build_reply(user_text: str) -> str:
    try:
        result = predict_sentiment(user_text)
    except HTTPException:
        return "ขออภัย ระบบวิเคราะห์ความรู้สึกยังไม่พร้อมใช้งาน"
    positive = result["sentiment"] == "positive"
    return (
        f"{'😊' if positive else '😞'} ผลการวิเคราะห์ความรู้สึก\n"
        f"ข้อความ: {user_text}\n"
        f"ความรู้สึก: {'เชิงบวก' if positive else 'เชิงลบ'}\n"
        f"ความมั่นใจ: {result['confidence'] * 100:.1f}%"
    )


def send_line_reply(reply_token: str, text: str) -> None:
    try:
        resp = requests.post(
            LINE_REPLY_URL,
            json={"replyToken": reply_token, "messages": [{"type": "text", "text": text}]},
            headers={"Authorization": f"Bearer {LINE_ACCESS_TOKEN}"},
            timeout=10,
        )
        logger.info("LINE reply status: %s", resp.status_code)
    except requests.RequestException as e:
        logger.error("LINE reply failed: %s", e)


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)


@app.get("/")
def root():
    return {
        "status": "online",
        "models_loaded": state["model"] is not None,
        "endpoints": ["/analyze", "/webhook"],
    }


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")
    return predict_sentiment(req.text)


@app.post("/webhook")
async def handle_line_webhook(request: Request):
    if not LINE_ACCESS_TOKEN or not LINE_CHANNEL_SECRET:
        raise HTTPException(status_code=500, detail="LINE_ACCESS_TOKEN / LINE_CHANNEL_SECRET not set")

    raw = await request.body()
    if not verify_line_signature(raw, request.headers.get("x-line-signature", "")):
        raise HTTPException(status_code=401, detail="invalid signature")

    body = await request.json()
    for event in body.get("events", []):
        msg = event.get("message", {})
        if event.get("type") != "message" or msg.get("type") != "text":
            continue
        reply_text = await asyncio.to_thread(build_reply, msg["text"])
        await asyncio.to_thread(send_line_reply, event["replyToken"], reply_text)

    return {"status": "ok"}
