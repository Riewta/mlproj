"""Aspect detection baseline: which of taste / price / service / ambience a text talks about.

Keywords are matched on word tokens, not substrings: "ถูก" (cheap) must not fire on
"ถูกใจ" (like), and "ช้า" (slow) must not fire on "ช้าง" (elephant). The flip side: PyThaiNLP's
dictionary keeps some compounds as one token ("ราคาแพง", "เจ้าของร้าน"), so those are listed too.

Words that are only an aspect in context are matched as two-token phrases instead:
"เยอะ" alone is usually a crowd ("คนเยอะ"), but "ให้ เยอะ" is portion size (price);
"เร็ว" alone can be anything ("หมดเร็ว"), but "เสิร์ฟ เร็ว" is service.
"""
import re

from pythainlp.tokenize import sent_tokenize, word_tokenize

ASPECTS = ["taste", "price", "service", "ambience"]
ASPECT_NAMES_TH = {"taste": "อาหาร/รสชาติ", "price": "ราคา/ความคุ้มค่า", "service": "บริการ", "ambience": "บรรยากาศ/สถานที่"}

ASPECT_KEYWORDS = {
    "taste": {"อร่อย", "รสชาติ", "รส", "จืด", "เค็ม", "หวาน", "เผ็ด", "เปรี้ยว", "กลมกล่อม", "สด",
              "คาว", "เหม็น", "นุ่ม", "เหนียว", "กรอบ", "หอม", "เลี่ยน", "แข็ง", "เย็นชืด", "ชืด",
              "จืดชืด", "กลิ่นหอม", "กลิ่นคาว", "ความสด", "ถูกปาก", "รสจัด", "มีรส", "รสเด็ด",
              "ความหวาน", "ความนุ่ม", "เหม็นคาว", "เหม็นหืน", "รสเปรี้ยว", "รสเค็ม", "แข็งๆ", "สดๆ",
              "ผงชูรส", "ปรุงรส"},
    "price": {"ราคา", "แพง", "ถูก", "คุ้ม", "คุ้มค่า", "บาท", "ปริมาณ", "ส่วนลด", "โปร", "โปรโมชั่น",
              "ราคาแพง", "ขึ้นราคา", "ค่าบริการ", "สมราคา", "ความคุ้มค่า", "ลดราคา", "สนนราคา"},
    "service": {"บริการ", "พนักงาน", "เสิร์ฟ", "รอ", "รอนาน", "ช้า", "สุภาพ", "ยิ้มแย้ม",
                "ต้อนรับ", "เจ้าของ", "แม่ค้า", "ดูแล", "ใส่ใจ", "ล่าช้า", "ความล่าช้า", "ชักช้า",
                "เจ้าของร้าน", "พนักงานบริการ", "ให้บริการ", "เด็กเสิร์ฟ", "พนักงานเสิร์ฟ", "ยิ้มแย้มแจ่มใส", "รวดเร็ว"},
    "ambience": {"บรรยากาศ", "ตกแต่ง", "สะอาด", "สกปรก", "แอร์", "ร้อน", "ที่นั่ง", "วิว", "เพลง",
                 "เสียงดัง", "กว้าง", "แคบ", "มุม", "ถ่ายรูป", "ห้องน้ำ", "ความสะอาด", "กว้างขวาง", "คับแคบ",
                 "ติดแอร์", "ความร้อน", "ทำความสะอาด", "สะอาดสะอ้าน", "ยุง"},
}

_FAST = ["เร็ว", "ไว"]
ASPECT_PHRASES = {
    "price": {("ให้", "เยอะ"), ("ปริมาณ", "เยอะ"), ("ให้", "น้อย")},
    "service": {(a, b) for a in ["ได้", "มา", "เสิร์ฟ", "บริการ"] for b in _FAST},
}

# Common misspellings that the segmenter chops into fragments ("เสริฟ" -> "เส", "ริ", "ฟ")
_SPELLING = {"เสริ์ฟ": "เสิร์ฟ", "เสริฟ": "เสิร์ฟ", "รสชาด": "รสชาติ"}

_PARAGRAPHS = re.compile(r"\n+")


def split_sentences(text: str) -> list[str]:
    """Split on line breaks first (crfcut merges across paragraphs), then crfcut inside each."""
    out = []
    for para in _PARAGRAPHS.split(text):
        out += [s.strip() for s in sent_tokenize(para, engine="crfcut") if s.strip()]
    return out


def detect_aspects(text: str) -> dict[str, int]:
    text = text.lower()
    for wrong, right in _SPELLING.items():
        text = text.replace(wrong, right)
    tokens = [t for t in word_tokenize(text) if not t.isspace()]
    words, pairs = set(tokens), set(zip(tokens, tokens[1:]))
    return {a: int(bool(words & ASPECT_KEYWORDS[a] or pairs & ASPECT_PHRASES.get(a, set()))) for a in ASPECTS}
