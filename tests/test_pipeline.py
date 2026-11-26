from src import db
from src.pipeline import ingest, normalize_ts


class FakePredictor:
    def predict(self, text):
        neg = "แย่" in text
        return {"sentiment": "negative" if neg else "positive", "confidence": 0.9, "label": int(not neg),
                "aspects": ["delay"] if neg else [], "tokens": text.split()}


def test_ingest_dedupe_and_daily(tmp_path):
    f = tmp_path / "r.csv"
    f.write_text(
        "review,created_at\n"
        "อร่อย ดี,2026-11-01\n"
        "แย่ ช้า,2026-11-01\n"
        "แย่ ช้า,2026-11-01\n"          # duplicate
        "ดีมาก,2026-11-02T10:00:00+07:00\n",
        encoding="utf-8",
    )
    conn = db.connect(":memory:")
    added, skipped = ingest(str(f), "test", None, conn=conn, predictor=FakePredictor())
    assert (added, skipped) == (3, 1)

    rep = db.daily_report(conn)
    assert [r["day"] for r in rep] == ["2026-11-01", "2026-11-02"]
    assert rep[0]["total"] == 2 and rep[0]["negative"] == 1
    assert rep[0]["negative_pct"] == 50.0
    assert rep[0]["top_complaints"] == {"delay": 1}
    assert db.daily_report(conn, days=1)[0]["day"] == "2026-11-02"


def test_headerless_csv_uses_ingest_time(tmp_path):
    f = tmp_path / "t.csv"
    f.write_text('"รีวิว แย่",1\n', encoding="utf-8")
    conn = db.connect(":memory:")
    ingest(str(f), "t", None, conn=conn, predictor=FakePredictor())
    assert conn.execute("SELECT created_at FROM reviews").fetchone()[0][:10] == db.utc_now()[:10]


def test_normalize_ts_converts_to_utc():
    assert normalize_ts("2026-11-02T01:00:00+07:00").startswith("2026-11-01T18:00:00")
