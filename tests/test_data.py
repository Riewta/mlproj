import pandas as pd

from src.data.load import stars_to_binary, wisesight_to_binary


def test_stars_to_binary_drops_three_stars():
    out = stars_to_binary(pd.Series([1, 2, 3, 4, 5]))
    assert out.tolist()[:2] == [0, 0]
    assert pd.isna(out[2])
    assert out.tolist()[3:] == [1, 1]


def test_wisesight_to_binary_drops_neutral_and_question():
    out = wisesight_to_binary(pd.Series(["pos", "neg", "neu", "q"]))
    assert out.tolist()[:2] == [1, 0]
    assert out[2:].isna().all()
