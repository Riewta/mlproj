"""Train the sentiment model served by the API.  Run from project root:  python -m src.train

Uses the same splits and settings as notebooks/02_baseline.ipynb, so the numbers it prints
are the numbers in the README. Downloads Wongnai / Wisesight from Hugging Face on first run.
"""
import os

import joblib
import pandas as pd

from src.cleaner import ThaiTextCleaner
from src.data.splits import wisesight_test, wongnai_splits
from src.metrics import sentiment_metrics
from src.models.baseline import build_baseline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# chosen on the validation set in notebooks/02_baseline.ipynb (threshold stays 0.5)
C = 50
CLASS_WEIGHT = None


def main() -> None:
    data = wongnai_splits()
    data["wisesight"] = wisesight_test()

    print("Cleaning Thai text (about a minute)...")
    cleaner = ThaiTextCleaner()
    for d in data.values():
        d["clean"] = [" ".join(cleaner.clean(t)) for t in d.text]

    print(f"Training on {len(data['train']):,} reviews...")
    pipe = build_baseline(C=C, class_weight=CLASS_WEIGHT).fit(data["train"].clean, data["train"].label)

    report = {name: sentiment_metrics(d.label, pipe.predict(d.clean))
              for name, d in data.items() if name != "train"}
    print(pd.DataFrame(report).T.round(3))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipe.named_steps["clf"], os.path.join(MODEL_DIR, "sentiment_model.pkl"))
    joblib.dump(pipe.named_steps["tfidf"], os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    print(f"Models saved to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
