import pandas as pd

from src.data.splits import make_splits
from src.metrics import sentiment_metrics


def _fake():
    train = [(f"pos {i}", 5) for i in range(90)] + [(f"neg {i}", 1) for i in range(10)]
    train += [("mixed", 3), ("pos 0", 5), ("leak", 1)]  # 3 stars, duplicate, leak
    test = [("leak", 1), ("t pos", 4), ("t neg", 2)]
    rows = [(t, s, "train") for t, s in train] + [(t, s, "test") for t, s in test]
    return pd.DataFrame(rows, columns=["text", "label_raw", "split"])


def test_make_splits_drops_three_stars_duplicates_and_leaks():
    s = make_splits(_fake(), val_size=0.2)
    train_val = pd.concat([s["train"], s["val"]])
    assert "mixed" not in set(train_val.text)
    assert "leak" not in set(train_val.text)
    assert train_val.text.is_unique
    assert len(s["test"]) == 3


def test_val_is_stratified():
    s = make_splits(_fake(), val_size=0.2)
    assert s["val"].label.value_counts().to_dict() == {1: 18, 0: 2}


def test_sentiment_metrics():
    m = sentiment_metrics([0, 0, 1, 1], [0, 1, 1, 1])
    assert m["accuracy"] == 0.75
    assert round(m["f1_negative"], 3) == 0.667
    assert round(m["f1_positive"], 3) == 0.8


def test_bootstrap_ci_contains_point_estimate():
    from src.metrics import bootstrap_ci
    y = [0] * 20 + [1] * 80
    p = [0] * 15 + [1] * 85
    lo, hi = bootstrap_ci(y, p, n=200)
    assert lo <= sentiment_metrics(y, p)["macro_f1"] <= hi
