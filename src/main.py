import os
import joblib
import requests
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

def identity_tokenizer(text):
    return text.split()

import __main__
__main__.identity_tokenizer = identity_tokenizer

from src.cleaner import ThaiTextCleaner

app = FastAPI(title="Thai Sentiment API")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
MODEL_DIR = os.path.join(BASE_DIR, 'models')

model = None
tfidf = None
cleaner = None

try:
    model = joblib.load(os.path.join(MODEL_DIR, 'sentiment_model.pkl'))
    tfidf = joblib.load(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
    cleaner = ThaiTextCleaner()
    print("Models loaded successfully.")
except Exception as e:
    print(f"Warning: Could not load models ({e}). Train first with: python src/train.py")

LINE_ACCESS_TOKEN = os.getenv("LINE_ACCESS_TOKEN", "")


def predict_sentiment(text: str) -> dict:
    if model is None or tfidf is None or cleaner is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Run train.py first.")
    tokens = cleaner.clean(text)
    cleaned = " ".join(tokens)
    vec = tfidf.transform([cleaned])
    label = int(model.predict(vec)[0])
    proba = model.predict_proba(vec)[0]
    confidence = float(max(proba))
    return {
        "sentiment": "positive" if label == 1 else "negative",
        "confidence": round(confidence, 4),
        "label": label,
    }


class AnalyzeRequest(BaseModel):
    text: str


@app.get("/")
def root():
    return {
        "status": "online",
        "models_loaded": model is not None,
        "endpoints": ["/analyze", "/webhook"],
    }


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")
    return predict_sentiment(req.text)


@app.post("/webhook")
async def handle_line_webhook(request: Request):
    if not LINE_ACCESS_TOKEN:
        raise HTTPException(status_code=500, detail="LINE_ACCESS_TOKEN not set in environment")

    body = await request.json()
    events = body.get("events", [])

    for event in events:
        if event.get("type") != "message":
            continue
        msg = event.get("message", {})
        if msg.get("type") != "text":
            continue

        user_text = msg["text"]
        reply_token = event["replyToken"]

        try:
            result = predict_sentiment(user_text)
            emoji = "😊" if result["sentiment"] == "positive" else "😞"
            reply_text = (
                f"{emoji} ผลการวิเคราะห์ความรู้สึก\n"
                f"ข้อความ: {user_text}\n"
                f"ความรู้สึก: {'เชิงบวก' if result['sentiment'] == 'positive' else 'เชิงลบ'}\n"
                f"ความมั่นใจ: {result['confidence'] * 100:.1f}%"
            )
        except HTTPException:
            reply_text = "ขออภัย ระบบวิเคราะห์ความรู้สึกยังไม่พร้อมใช้งาน"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LINE_ACCESS_TOKEN}",
        }
        payload = {
            "replyToken": reply_token,
            "messages": [{"type": "text", "text": reply_text}],
        }
        resp = requests.post(
            "https://api.line.me/v2/bot/message/reply",
            json=payload,
            headers=headers,
        )
        print(f"LINE reply status: {resp.status_code}")

    return {"status": "ok"}
