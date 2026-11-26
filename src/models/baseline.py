"""TF-IDF + Logistic Regression. Input is text already cleaned by ThaiTextCleaner (space-joined tokens)."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.tokenizer import identity_tokenizer


def build_baseline(C: float = 5.0, class_weight: str | None = "balanced") -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            tokenizer=identity_tokenizer,
            token_pattern=None,
            lowercase=False,
            max_features=20000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(C=C, class_weight=class_weight, max_iter=1000)),
    ])
