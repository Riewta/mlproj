"""Review pipeline: ingest -> clean/tokenize -> sentiment -> SQLite -> daily report.

  python -m src.pipeline ingest data/reviews.csv [--limit N] [--source NAME]
  python -m src.pipeline report [--days N]

CSV formats:
  with header : columns `review` and optional `created_at` (ISO date/datetime)
  headerless  : review text in column 0 (e.g. data/train.csv), no dates -> ingest time is used
"""
import argparse
import csv
import os
import sys
from datetime import datetime, timezone

from src import db
from src.predictor import Predictor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")


def read_csv(path: str):
    """Yield (review, created_at | None)."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = csv.reader(f)
        first = next(rows, None)
        if first is None:
            return
        if "review" in [c.strip().lower() for c in first]:
            header = [c.strip().lower() for c in first]
            ri = header.index("review")
            ci = header.index("created_at") if "created_at" in header else None
            for r in rows:
                if len(r) > ri and r[ri].strip():
                    yield r[ri].strip(), (r[ci].strip() or None) if ci is not None and len(r) > ci else None
        else:
            for r in [first, *rows]:
                if r and r[0].strip():
                    yield r[0].strip(), None


def normalize_ts(value: str | None) -> str:
    if not value:
        return db.utc_now()
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def ingest(path: str, source: str, limit: int | None, conn=None, predictor=None) -> tuple[int, int]:
    conn = conn or db.connect()
    predictor = predictor or Predictor(MODEL_DIR)
    added = skipped = 0
    for i, (text, ts) in enumerate(read_csv(path)):
        if limit and i >= limit:
            break
        result = predictor.predict(text)
        if db.insert_review(conn, source=source, dedupe_key=db.text_key(text), text=text,
                            result=result, created_at=normalize_ts(ts)):
            added += 1
        else:
            skipped += 1
        if (i + 1) % 1000 == 0:
            conn.commit()
            print(f"  {i + 1} processed...")
    conn.commit()
    return added, skipped


def print_report(rows: list[dict]) -> None:
    if not rows:
        print("No data. Run ingest first.")
        return
    print(f"{'day':10}  {'total':>5}  {'pos':>5}  {'neg':>5}  {'neg%':>5}  top complaints")
    for r in rows:
        comp = ", ".join(f"{a}={n}" for a, n in r["top_complaints"].items()) or "-"
        print(f"{r['day']:10}  {r['total']:5d}  {r['positive']:5d}  {r['negative']:5d}  "
              f"{r['negative_pct']:5.1f}  {comp}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("ingest")
    pi.add_argument("csv")
    pi.add_argument("--limit", type=int)
    pi.add_argument("--source", default=None, help="label stored with each row (default: file name)")
    pr = sub.add_parser("report")
    pr.add_argument("--days", type=int)
    args = p.parse_args()

    if args.cmd == "ingest":
        added, skipped = ingest(args.csv, args.source or os.path.basename(args.csv), args.limit)
        print(f"Added {added}, skipped {skipped} duplicates -> {db.db_path()}")
    else:
        print_report(db.daily_report(db.connect(), args.days))


if __name__ == "__main__":
    sys.exit(main())
