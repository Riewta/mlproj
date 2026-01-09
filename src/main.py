from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
from fastapi import Request
import requests
# solve หา identity ไม่เจอ;
def identity_tokenizer(text):
    return text.split()

# ระบบมองภาพขนาดใหญ่มากขึ้นก่อนโหลด
import __main__

__main__.identity_tokenizer = identity_tokenizer

from src.cleaner import ThaiTextCleaner

app = FastAPI()
# ส่วนAi
# ใช้ path absolute
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
model_path = os.path.join(BASE_DIR, 'models', 'sentiment_model.pkl')
tfidf_path = os.path.join(BASE_DIR, 'models', 'tfidf_vectorizer.pkl')

try:
    model = joblib.load(model_path)
    tfidf = joblib.load(tfidf_path)
    cleaner = ThaiTextCleaner()
    print("AI Models Loaded Successfully!")
except Exception as e:
    print(f"Error Loading Models: {e}")


# set type of data;
class ReviewRequest(BaseModel):
    text: str


@app.get("/")
def root():
    return {"status": "online", "message": "Sentiment Analysis API is ready!"}


LINE_ACCESS_TOKEN = "21SjO2+W/RyK8CVEwWa2zUU3JOFMIFtK9ZuoAW983GSRy6tG/GCME2EuiLgw/zi94VUKdr4EDNgA4LX/5bBFJA9f3JzBZfhNodgXYPcN6Y7joUFeikg0PiHIYtneNdXOrijQZGyEweXK4BlyTLoC0wdB04t89/1O/w1cDnyilFU="


@app.post("/predict")
async def handle_line(request: Request):
    body = await request.json()

    if "events" in body and len(body["events"]) > 0:
        event = body["events"][0]

        # เช็กว่าเป็นข้อความป่าว
        if event["type"] == "message":
            user_msg = event["message"]["text"]
            reply_token = event["replyToken"]  # reply

            print(f"Naevis received: {user_msg}")

            # ตอบกลับ
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {LINE_ACCESS_TOKEN}"
            }
            payload = {
                "replyToken": reply_token,
                "messages": [
                    {
                        "type": "text",
                        "text": f"Naevis ได้รับข้อความว่า: {user_msg}"
                    }
                ]
            }
            # send back to Line
            response = requests.post("https://api.line.me/v2/bot/message/reply", json=payload, headers=headers)
            print(f"Reply Status: {response.status_code}")  # เช็กว่าส่งผ่านรึป่าว

    return {"status": "ok"}