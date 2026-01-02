import re
import pythainlp
from pythainlp.corpus.common import thai_stopwords


class ThaiTextCleaner:
    def __init__(self):
        self.stop_words = list(thai_stopwords())

    def clean(self, text):
        text = str(text).strip()
        text = re.sub(r'[^\u0E00-\u0E7Fa-zA-Z0-9\s]', '', text)

        # delete repeating word
        text = pythainlp.util.remove_tonemark(text) # this also delete วรรณยุกต์ when the case that need only meaning

        # cut Th words
        tokens = pythainlp.tokenize.word_tokenize(text, engine="newmm")

        # delete stop word and space
        cleaned_tokens = [w for w in tokens if w not in self.stop_words and not w.isspace()]

        return cleaned_tokens


# testing
if __name__ == "__main__":
    cleaner = ThaiTextCleaner()
    sample = "สวัสดีครับบบบบ สนใจสินค้าชิ้นนี้มากกกก! ราคาเท่าไหร่ครับ?"
    print(f"Original: {sample}")
    print(f"Cleaned: {cleaner.clean(sample)}")