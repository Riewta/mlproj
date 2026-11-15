import re

from pythainlp.corpus.common import thai_stopwords
from pythainlp.tokenize import word_tokenize

# Negations / intensifiers carry sentiment — never drop them as stopwords.
KEEP_WORDS = {"ไม่", "ไม่ได้", "ไม่มี", "ยัง", "มาก", "ที่สุด", "เลย", "ห่วย", "ดี"}

_URL_RE = re.compile(r"https?://\S+")
_NON_TEXT_RE = re.compile(r"[^฀-๿a-zA-Z0-9\s]")
_REPEAT_RE = re.compile(r"(.)\1{2,}")


class ThaiTextCleaner:
    def __init__(self):
        self.stop_words = set(thai_stopwords()) - KEEP_WORDS

    def clean(self, text) -> list[str]:
        text = str(text).strip()
        text = _URL_RE.sub("", text)
        text = _NON_TEXT_RE.sub("", text)
        text = _REPEAT_RE.sub(r"\1", text)
        tokens = word_tokenize(text, engine="newmm")
        return [w for w in tokens if w not in self.stop_words and not w.isspace()]


if __name__ == "__main__":
    cleaner = ThaiTextCleaner()
    sample = "สวัสดีครับบบบบ สนใจสินค้าชิ้นนี้มากกกก! ราคาเท่าไหร่ครับ?"
    print(f"Original: {sample}")
    print(f"Cleaned: {cleaner.clean(sample)}")
