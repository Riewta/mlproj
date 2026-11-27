import pytest

from src.models.transformer import preprocess, truncate


def test_preprocess_lowercases_and_marks_spaces():
    assert preprocess("  MK Suki  อร่อย\nมาก ") == "mk<_>suki<_>อร่อย<_>มาก"


def test_truncate_short_input_unchanged():
    assert truncate([1, 2, 3], 5) == [1, 2, 3]


def test_truncate_head():
    assert truncate(list(range(10)), 4) == [0, 1, 2, 3]


def test_truncate_head_tail_keeps_both_ends():
    assert truncate(list(range(10)), 5, "head_tail", head=2) == [0, 1, 7, 8, 9]


def test_truncate_unknown_strategy():
    with pytest.raises(ValueError):
        truncate(list(range(10)), 5, "middle")
