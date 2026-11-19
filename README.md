# mlproj — Thai Sentiment API

Thai review sentiment (positive/negative) with TF-IDF + Logistic Regression, served via FastAPI and a LINE bot webhook.

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env   # fill LINE_ACCESS_TOKEN, LINE_CHANNEL_SECRET
```

## Train
Put `data/train.csv` (`review,rating` rows; rating ≥4 positive, ≤2 negative, 3 dropped), then:
```bash
python -m src.train
```

## Evaluate on delivery reviews
```bash
python -m src.make_label_set 200   # -> data/to_label.csv (delivery-related Wongnai reviews)
# label by hand: sentiment (1/0) and food/rider/delay/price (1/0), save as data/labeled.csv
python -m src.evaluate             # sentiment report + aspect baseline F1
```
Aspects are a keyword-rule baseline (`src/aspects.py`), returned in `/analyze` as `aspects`.

## Pipeline (ingest → clean/tokenize → SQLite → daily report)
```bash
python -m src.pipeline ingest data/reviews.csv   # header: review[,created_at]; headerless: review in col 0
python -m src.pipeline report --days 14
```
Stored in `data/reviews.db` (override with `DB_PATH`). Duplicates are skipped, so re-running is safe.
LINE messages received on `/webhook` are stored too (text only, no user id). `GET /stats/daily` returns the same report.
Rows without `created_at` get the ingest time, so undated data (e.g. Wongnai) lands on a single day.

## Data source
`data/train.csv` = [Wongnai reviews](https://huggingface.co/datasets/Wongnai/wongnai_reviews) (restaurant reviews from wongnai.com, LGPL-3.0). Not committed (`data/` is gitignored). Check the license terms before showing or redistributing the raw text.

## Run
```bash
uvicorn src.main:app --reload
curl -X POST localhost:8000/analyze -H 'Content-Type: application/json' -d '{"text":"สินค้าดีมาก"}'
```

## Test
```bash
pytest
```
