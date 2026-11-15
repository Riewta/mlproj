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

## Run
```bash
uvicorn src.main:app --reload
curl -X POST localhost:8000/analyze -H 'Content-Type: application/json' -d '{"text":"สินค้าดีมาก"}'
```

## Test
```bash
pytest
```
