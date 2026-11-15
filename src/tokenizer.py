def identity_tokenizer(text: str) -> list[str]:
    """Text is pre-tokenized (space separated) by ThaiTextCleaner."""
    return text.split()
