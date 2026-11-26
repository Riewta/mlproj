import hashlib
import json
import os
import sqlite3
from collections import Counter
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB = os.path.join(BASE_DIR, "data", "reviews.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS reviews (
    id          INTEGER PRIMARY KEY,
    source      TEXT NOT NULL,
    dedupe_key  TEXT NOT NULL,
    text        TEXT NOT NULL,
    tokens      TEXT NOT NULL,
    sentiment   TEXT NOT NULL,
    confidence  REAL NOT NULL,
    aspects     TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    UNIQUE (source, dedupe_key)
);
CREATE INDEX IF NOT EXISTS idx_reviews_created ON reviews (created_at);
"""


def db_path() -> str:
    return os.getenv("DB_PATH", DEFAULT_DB)


def connect(path: str | None = None) -> sqlite3.Connection:
    path = path or db_path()
    if path != ":memory:":
        os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def text_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def insert_review(conn, *, source: str, dedupe_key: str, text: str, result: dict, created_at: str) -> bool:
    """Insert one analysed review. Returns False if it already exists."""
    cur = conn.execute(
        "INSERT OR IGNORE INTO reviews "
        "(source, dedupe_key, text, tokens, sentiment, confidence, aspects, created_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (
            source, dedupe_key, text,
            " ".join(result["tokens"]),
            result["sentiment"], result["confidence"],
            json.dumps(result["aspects"], ensure_ascii=False),
            created_at,
        ),
    )
    return cur.rowcount == 1


def daily_report(conn, days: int | None = None) -> list[dict]:
    """Per-day sentiment counts and the aspects mentioned in negative reviews."""
    rows = conn.execute(
        "SELECT substr(created_at, 1, 10) AS day, sentiment, aspects FROM reviews ORDER BY day"
    ).fetchall()

    days_map: dict[str, dict] = {}
    for r in rows:
        d = days_map.setdefault(r["day"], {"day": r["day"], "total": 0, "positive": 0,
                                           "negative": 0, "_neg_aspects": Counter()})
        d["total"] += 1
        d[r["sentiment"]] += 1
        if r["sentiment"] == "negative":
            d["_neg_aspects"].update(json.loads(r["aspects"]))

    out = []
    for d in days_map.values():
        d["negative_pct"] = round(100 * d["negative"] / d["total"], 1)
        d["top_complaints"] = dict(d.pop("_neg_aspects").most_common())
        out.append(d)
    return out[-days:] if days else out
