import asyncio
import base64
import hashlib
import hmac
import logging
import os
from datetime import datetime, timezone
from contextlib import asynccontextmanager

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field

from src import db
from src.predictor import Predictor

load_dotenv()
logger = logging.getLogger("sentiment")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

LINE_ACCESS_TOKEN = os.getenv("LINE_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
LINE_REPLY_URL = "https://api.line.me/v2/bot/message/reply"

predictor: Predictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    try:
        predictor = Predictor(MODEL_DIR)
        logger.info("Models loaded successfully.")
    except Exception as e:
        logger.warning("Could not load models (%s). Train first: python -m src.train", e)
    yield


app = FastAPI(title="Thai Sentiment API", lifespan=lifespan)


def predict_sentiment(text: str) -> dict:
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Run: python -m src.train")
    return predictor.predict(text)


def store_line_message(msg_id: str, text: str, result: dict, ts_ms: int | None) -> None:
    created = (datetime.fromtimestamp(ts_ms / 1000, timezone.utc).isoformat(timespec="seconds")
               if ts_ms else db.utc_now())
    conn = db.connect()
    try:
        db.insert_review(conn, source="line", dedupe_key=msg_id, text=text, result=result, created_at=created)
        conn.commit()
    finally:
        conn.close()


def verify_line_signature(body: bytes, signature: str) -> bool:
    digest = hmac.new(LINE_CHANNEL_SECRET.encode(), body, hashlib.sha256).digest()
    return hmac.compare_digest(base64.b64encode(digest).decode(), signature)


def build_reply(user_text: str, result: dict | None) -> str:
    if result is None:
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
        "models_loaded": predictor is not None,
        "endpoints": ["/analyze", "/webhook", "/stats/daily"],
    }


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")
    result = predict_sentiment(req.text)
    result.pop("tokens")
    return result


@app.get("/stats/daily")
def stats_daily(days: int | None = Query(default=None, ge=1, le=365)):
    conn = db.connect()
    try:
        return db.daily_report(conn, days)
    finally:
        conn.close()


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
        result = predictor.predict(msg["text"]) if predictor else None
        if result is not None:
            await asyncio.to_thread(store_line_message, msg.get("id", event.get("webhookEventId", "")),
                                    msg["text"], result, event.get("timestamp"))
        reply_text = await asyncio.to_thread(build_reply, msg["text"], result)
        await asyncio.to_thread(send_line_reply, event["replyToken"], reply_text)

    return {"status": "ok"}
