"""Evaluate sentiment model + aspect baseline on hand-labeled reviews.

Run:  python -m src.evaluate [data/labeled.csv]
CSV columns: review, sentiment (1 pos / 0 neg), food, rider, delay, price (1/0).
Rows with empty cells are skipped.
"""
import os
import sys

import joblib
import pandas as pd
from sklearn.metrics import classification_report, f1_score

from src.aspects import ASPECTS, detect_aspects
from src.cleaner import ThaiTextCleaner

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE_DIR, "data", "labeled.csv")
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.dropna(subset=["sentiment"] + ASPECTS).reset_index(drop=True)
    if df.empty:
        raise SystemExit(f"No fully labeled rows in {path}")
    print(f"Evaluating on {len(df)} labeled reviews\n")

    model = joblib.load(os.path.join(MODEL_DIR, "sentiment_model.pkl"))
    tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    cleaner = ThaiTextCleaner()

    cleaned = [" ".join(cleaner.clean(t)) for t in df["review"]]
    pred = model.predict(tfidf.transform(cleaned))
    print("== Sentiment ==")
    print(classification_report(df["sentiment"].astype(int), pred,
                                target_names=["Negative", "Positive"], zero_division=0))

    print("== Aspects (keyword baseline) ==")
    aspect_pred = pd.DataFrame([detect_aspects(t) for t in df["review"]])
    for a in ASPECTS:
        y = df[a].astype(int)
        acc = (aspect_pred[a] == y).mean()
        f1 = f1_score(y, aspect_pred[a], zero_division=0)
        print(f"{a:6s} support={int(y.sum()):3d}  acc={acc:.2f}  f1={f1:.2f}")
    macro = sum(f1_score(df[a].astype(int), aspect_pred[a], zero_division=0) for a in ASPECTS) / len(ASPECTS)
    print(f"macro F1 = {macro:.2f}")


if __name__ == "__main__":
    main()
