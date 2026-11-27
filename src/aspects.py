"""Aspect detection baseline: which of taste / price / service / ambience a text talks about.

Keywords are matched on word tokens, not substrings: "ถูก" (cheap) must not fire on
"ถูกใจ" (like), and "ช้า" (slow) must not fire on "ช้าง" (elephant). The flip side: PyThaiNLP's
dictionary keeps some compounds as one token ("ราคาแพง"), so those are listed explicitly.
"""
import re

from pythainlp.tokenize import sent_tokenize, word_tokenize

ASPECTS = ["taste", "price", "service", "ambience"]
ASPECT_NAMES_TH = {"taste": "อาหาร/รสชาติ", "price": "ราคา/ความคุ้มค่า", "service": "บริการ", "ambience": "บรรยากาศ/สถานที่"}

ASPECT_KEYWORDS = {
    "taste": {"อร่อย", "รสชาติ", "รส", "จืด", "เค็ม", "หวาน", "เผ็ด", "เปรี้ยว", "กลมกล่อม", "สด",
              "คาว", "เหม็น", "นุ่ม", "เหนียว", "กรอบ", "หอม", "เลี่ยน", "แข็ง", "เย็นชืด", "ชืด",
              "จืดชืด", "กลิ่นหอม", "กลิ่นคาว", "ความสด"},
    "price": {"ราคา", "แพง", "ถูก", "คุ้ม", "คุ้มค่า", "บาท", "ปริมาณ", "เยอะ", "ส่วนลด", "โปร", "โปรโมชั่น",
              "ราคาแพง", "ขึ้นราคา", "ค่าบริการ"},
    "service": {"บริการ", "พนักงาน", "เสิร์ฟ", "เสริฟ", "รอ", "รอนาน", "ช้า", "สุภาพ", "ยิ้มแย้ม",
                "ต้อนรับ", "เจ้าของ", "แม่ค้า", "ดูแล", "ใส่ใจ", "ล่าช้า", "ความล่าช้า", "ชักช้า"},
    "ambience": {"บรรยากาศ", "ตกแต่ง", "สะอาด", "สกปรก", "แอร์", "ร้อน", "ที่นั่ง", "วิว", "เพลง",
                 "เสียงดัง", "กว้าง", "แคบ", "มุม", "ถ่ายรูป", "ห้องน้ำ", "ความสะอาด", "กว้างขวาง", "คับแคบ"},
}

_PARAGRAPHS = re.compile(r"\n+")


def split_sentences(text: str) -> list[str]:
    """Split on line breaks first (crfcut merges across paragraphs), then crfcut inside each."""
    out = []
    for para in _PARAGRAPHS.split(text):
        out += [s.strip() for s in sent_tokenize(para, engine="crfcut") if s.strip()]
    return out


def detect_aspects(text: str) -> dict[str, int]:
    tokens = set(word_tokenize(text.lower()))
    return {a: int(bool(tokens & kws)) for a, kws in ASPECT_KEYWORDS.items()}
