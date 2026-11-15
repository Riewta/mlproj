"""Train Thai sentiment model.  Run from project root:  python -m src.train"""
import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from src.cleaner import ThaiTextCleaner
from src.tokenizer import identity_tokenizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "train.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

POSITIVE_MIN = 4  # rating >= 4 -> positive
NEGATIVE_MAX = 2  # rating <= 2 -> negative; 3 is neutral and dropped


def load_data(path: str) -> pd.DataFrame:
    """Load `review,rating` rows. Reviews may contain commas, so split on the last one."""
    with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
        lines = f.read().splitlines()

    rows = []
    for line in lines:
        review, sep, rating = line.rpartition(",")
        rating = rating.strip().strip('"')
        if sep and rating.isdigit():
            rows.append({"review": review.strip().strip('"'), "rating": int(rating)})

    if not rows:  # fallback: rating first
        for line in lines:
            rating, sep, review = line.partition(",")
            rating = rating.strip().strip('"')
            if sep and rating.isdigit():
                rows.append({"review": review.strip().strip('"'), "rating": int(rating)})

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No valid rows found in {path}")

    df = df[(df["rating"] >= POSITIVE_MIN) | (df["rating"] <= NEGATIVE_MAX)]
    df = df.drop_duplicates(subset="review")
    df["label"] = (df["rating"] >= POSITIVE_MIN).astype(int)
    return df.reset_index(drop=True)


def main() -> None:
    if not os.path.exists(DATA_PATH):
        raise SystemExit(f"Data not found: {DATA_PATH}")

    df = load_data(DATA_PATH)
    pos = int(df["label"].sum())
    print(f"Loaded {len(df)} rows | Positive: {pos} | Negative: {len(df) - pos}")

    print("Cleaning Thai text (this may take a few minutes)...")
    cleaner = ThaiTextCleaner()
    df["cleaned"] = [" ".join(cleaner.clean(t)) for t in df["review"]]
    df = df[df["cleaned"].str.len() > 0]

    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    print("Training model...")
    tfidf = TfidfVectorizer(
        tokenizer=identity_tokenizer,
        token_pattern=None,
        lowercase=False,
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    model = LogisticRegression(max_iter=1000, C=5, class_weight="balanced")
    model.fit(X_train_vec, y_train)

    print("\nModel Evaluation:")
    print(classification_report(y_test, model.predict(X_test_vec),
                                target_names=["Negative", "Positive"]))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, "sentiment_model.pkl"))
    joblib.dump(tfidf, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    print(f"Models saved to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
