"""Keyword-rule baseline for aspect detection (food / rider / delay / price)."""

ASPECT_KEYWORDS = {
    "food": ["อร่อย", "รสชาติ", "รส", "อาหาร", "เมนู", "จืด", "เค็ม", "หวาน", "เผ็ด",
             "สด", "เหม็น", "คาว", "ปริมาณ", "น้อย", "เย็นชืด", "ชืด", "บูด"],
    "rider": ["ไรเดอร์", "rider", "คนส่ง", "พนักงานส่ง", "พี่ส่ง", "สุภาพ", "ไม่รับสาย", "โทร"],
    "delay": ["ช้า", "นาน", "รอ", "เร็ว", "ไว", "ตรงเวลา", "ชั่วโมง", "นาที", "ล่าช้า", "เย็นหมด"],
    "price": ["ราคา", "แพง", "ถูก", "คุ้ม", "บาท", "ค่าส่ง", "โปร", "ส่วนลด", "คูปอง"],
}
ASPECTS = list(ASPECT_KEYWORDS)


def detect_aspects(text: str) -> dict[str, int]:
    low = text.lower()
    return {a: int(any(k in low for k in kws)) for a, kws in ASPECT_KEYWORDS.items()}
