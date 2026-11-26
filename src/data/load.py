"""Load Wongnai and Wisesight from Hugging Face into one shape: text, label_raw, split, source."""
import pandas as pd

WISESIGHT_CLASSES = ["pos", "neu", "neg", "q"]


def _read(repo: str, filename: str) -> pd.DataFrame:
    from huggingface_hub import hf_hub_download  # imported here so the API doesn't need it

    return pd.read_parquet(hf_hub_download(repo, filename, repo_type="dataset"))


def load_wongnai(splits=("train", "test")) -> pd.DataFrame:
    frames = []
    for split in splits:
        df = _read("Wongnai/wongnai_reviews", f"data/{split}-00000-of-00001.parquet")
        frames.append(pd.DataFrame({
            "text": df["review_body"],
            "label_raw": df["star_rating"] + 1,  # HF stores 0-4, we want 1-5 stars
            "split": split,
            "source": "wongnai",
        }))
    return pd.concat(frames, ignore_index=True)


def load_wisesight(splits=("train", "validation", "test")) -> pd.DataFrame:
    frames = []
    for split in splits:
        df = _read("pythainlp/wisesight_sentiment", f"wisesight_sentiment/{split}-00000-of-00001.parquet")
        frames.append(pd.DataFrame({
            "text": df["texts"],
            "label_raw": df["category"].map(dict(enumerate(WISESIGHT_CLASSES))),
            "split": split,
            "source": "wisesight",
        }))
    return pd.concat(frames, ignore_index=True)


def stars_to_binary(stars: pd.Series) -> pd.Series:
    """4-5 stars -> 1, 1-2 stars -> 0, 3 stars -> NA (dropped later)."""
    return stars.map({1: 0, 2: 0, 4: 1, 5: 1}).astype("Int64")


def wisesight_to_binary(category: pd.Series) -> pd.Series:
    """pos -> 1, neg -> 0, neu / q -> NA (dropped later)."""
    return category.map({"pos": 1, "neg": 0}).astype("Int64")
