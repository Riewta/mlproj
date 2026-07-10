import re
import pythainlp
from pythainlp.corpus.common import thai_stopwords


class ThaiTextCleaner:
    def __init__(self):
        self.stop_words = set(thai_stopwords())

    def _normalize_repeated_chars(self, text):
        return re.sub(r'(.)\1{2,}', r'\1', text)

    def clean(self, text):
        text = str(text).strip()
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'[^฀-๿a-zA-Z0-9\s]', '', text)
        text = self._normalize_repeated_chars(text)
        text = pythainlp.util.remove_tonemark(text)
        tokens = pythainlp.tokenize.word_tokenize(text, engine="newmm")
        cleaned_tokens = [w for w in tokens if w not in self.stop_words and not w.isspace()]
        return cleaned_tokens


if __name__ == "__main__":
    cleaner = ThaiTextCleaner()
    sample = "สวัสดีครับบบบบ สนใจสินค้าชิ้นนี้มากกกก! ราคาเท่าไหร่ครับ?"
    print(f"Original: {sample}")
    print(f"Cleaned: {cleaner.clean(sample)}")
