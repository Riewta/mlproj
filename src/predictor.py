import os

import joblib

from src.aspects import detect_aspects
from src.cleaner import ThaiTextCleaner
from src.tokenizer import identity_tokenizer  # noqa: F401  (needed to unpickle tfidf)


class Predictor:
    def __init__(self, model_dir: str):
        self.model = joblib.load(os.path.join(model_dir, "sentiment_model.pkl"))
        self.tfidf = joblib.load(os.path.join(model_dir, "tfidf_vectorizer.pkl"))
        self.cleaner = ThaiTextCleaner()

    def predict(self, text: str) -> dict:
        tokens = self.cleaner.clean(text)
        vec = self.tfidf.transform([" ".join(tokens)])
        proba = self.model.predict_proba(vec)[0]
        label = int(self.model.classes_[proba.argmax()])
        return {
            "sentiment": "positive" if label == 1 else "negative",
            "confidence": round(float(proba.max()), 4),
            "label": label,
            "aspects": [a for a, hit in detect_aspects(text).items() if hit],
            "tokens": tokens,
        }
