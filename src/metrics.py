import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def sentiment_metrics(y_true, y_pred) -> dict:
    """The numbers every model reports. Label 0 = negative, 1 = positive."""
    f1_neg, f1_pos = f1_score(y_true, y_pred, average=None, labels=[0, 1], zero_division=0)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_negative": float(f1_neg),
        "f1_positive": float(f1_pos),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def bootstrap_ci(y_true, y_pred, metric="macro_f1", n=1000, seed=0) -> tuple[float, float]:
    """95% interval of a metric by resampling the test set. Overlapping intervals = no clear winner."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        scores.append(sentiment_metrics(y_true[idx], y_pred[idx])[metric])
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))
