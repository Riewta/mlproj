"""Fixed train / val / test splits, shared by every model so results are comparable."""
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.load import load_wisesight, load_wongnai, stars_to_binary, wisesight_to_binary

SEED = 42


def make_splits(df: pd.DataFrame, val_size: float = 0.1, seed: int = SEED) -> dict[str, pd.DataFrame]:
    """df has text, label_raw (stars), split. Returns train / val / test with text + label.

    - 3-star reviews dropped, duplicates dropped
    - train rows that also appear in test are removed (leakage)
    - val is a stratified slice of train; test stays as published
    """
    df = df.assign(label=stars_to_binary(df.label_raw)).dropna(subset=["label"])
    df = df.drop_duplicates(["split", "text"])
    test = df[df.split == "test"]
    train = df[(df.split == "train") & ~df.text.isin(test.text)]
    train, val = train_test_split(train, test_size=val_size, stratify=train.label, random_state=seed)
    cols = ["text", "label"]
    return {
        "train": train[cols].reset_index(drop=True),
        "val": val[cols].reset_index(drop=True),
        "test": test[cols].reset_index(drop=True),
    }


def wongnai_splits(val_size: float = 0.1) -> dict[str, pd.DataFrame]:
    return make_splits(load_wongnai(), val_size)


def wisesight_test() -> pd.DataFrame:
    """Cross-domain test set: Wisesight test split, pos/neg only."""
    df = load_wisesight(["test"])
    df = df.assign(label=wisesight_to_binary(df.label_raw)).dropna(subset=["label"])
    return df[["text", "label"]].drop_duplicates("text").reset_index(drop=True)
