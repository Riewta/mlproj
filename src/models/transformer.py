"""WangchanBERTa input preparation, shared by the training notebook and (later) the API."""
import re

MODEL_NAME = "airesearch/wangchanberta-base-att-spm-uncased"
MAX_LENGTH = 416  # sequence length used in pretraining; the tokenizer has no default limit

_SPACES = re.compile(r"\s+")


def preprocess(text: str) -> str:
    """Match the pretraining format: lowercase (uppercase becomes <unk>) and spaces as <_>."""
    return _SPACES.sub("<_>", text.strip().lower())


def truncate(ids: list[int], max_tokens: int, strategy: str = "head", head: int = 128) -> list[int]:
    """Cut token ids (without special tokens) to max_tokens.

    head:      keep the beginning
    head_tail: keep the first `head` tokens and the end (reviews often conclude at the end)
    """
    if len(ids) <= max_tokens:
        return ids
    if strategy == "head":
        return ids[:max_tokens]
    if strategy == "head_tail":
        return ids[:head] + ids[len(ids) - (max_tokens - head):]
    raise ValueError(f"unknown strategy: {strategy}")


def encode(tokenizer, texts, max_length: int = MAX_LENGTH, strategy: str = "head", head: int = 128) -> list[list[int]]:
    """Tokenize, truncate and add <s> ... </s>. Padding is left to the data collator."""
    ids = tokenizer([preprocess(t) for t in texts], add_special_tokens=False)["input_ids"]
    max_tokens = max_length - 2  # room for <s> and </s>
    cls, sep = tokenizer.cls_token_id, tokenizer.sep_token_id
    return [[cls] + truncate(x, max_tokens, strategy, head) + [sep] for x in ids]
