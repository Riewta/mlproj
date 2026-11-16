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

## Run
```bash
uvicorn src.main:app --reload
curl -X POST localhost:8000/analyze -H 'Content-Type: application/json' -d '{"text":"สินค้าดีมาก"}'
```

## Test
```bash
pytest
```
