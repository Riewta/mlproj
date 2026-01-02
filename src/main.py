from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
# solve หา identity ไม่เจอ;
def identity_tokenizer(text):
    return text.split()

# ระบบมองภาพขนาดใหญ่มากขึ้นก่อนโหลด
import __main__

__main__.identity_tokenizer = identity_tokenizer

from cleaner import ThaiTextCleaner

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


@app.post("/predict")
def predict_sentiment(request: ReviewRequest):
    try:
        # cut the word
        cleaned = " ".join(cleaner.clean(request.text))
        # turn to numb
        vectorized = tfidf.transform([cleaned])
        # predict
        prediction = model.predict(vectorized)[0]

        return {
            "text": request.text,
            "sentiment": "Positive" if prediction == 1 else "Negative",
            "confidence": "high"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))