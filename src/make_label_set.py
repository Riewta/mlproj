"""Pick delivery-related reviews from data/train.csv into a CSV to hand-label.

Run:  python -m src.make_label_set [N]
Output: data/to_label.csv  (fill `sentiment` 1/0 and aspect columns 1/0 by hand,
then save as data/labeled.csv)
"""
import csv
import os
import random
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE_DIR, "data", "train.csv")
OUT = os.path.join(BASE_DIR, "data", "to_label.csv")

# "สั่งอาหาร" alone is mostly dine-in ordering, so it is excluded on purpose.
DELIVERY_KW = [
    "เดลิเวอรี", "เดลิเวอรี่", "เดริเวอรี", "delivery", "ไรเดอร์", "rider",
    "lineman", "line man", "ไลน์แมน", "grab", "แกร็บ", "foodpanda", "ฟู้ดแพนด้า",
    "ส่งอาหาร", "จัดส่ง", "ส่งถึงบ้าน", "สั่งผ่าน",
]
COLUMNS = ["review", "sentiment", "food", "rider", "delay", "price"]


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    with open(SRC, encoding="utf-8", newline="") as f:
        reviews = [r[0] for r in csv.reader(f) if r]

    hits = [r for r in reviews if any(k in r.lower() for k in DELIVERY_KW)]
    print(f"{len(hits)} delivery-related reviews found")

    random.Random(42).shuffle(hits)
    with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for r in hits[:n]:
            w.writerow([r, "", "", "", "", ""])
    print(f"Wrote {min(n, len(hits))} rows to {OUT}")


if __name__ == "__main__":
    main()
