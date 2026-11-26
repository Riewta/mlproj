# mlproj — Thai Review Sentiment Pipeline

Thai review sentiment (positive / negative) + aspect tags (food, rider, delay, price), stored in SQLite with a daily report. Served through FastAPI and a LINE bot webhook.

The model is trained from scratch (PyThaiNLP tokenization → TF-IDF → Logistic Regression). It is **not** an LLM call.

## Flow

```mermaid
flowchart LR
    A[CSV reviews] --> C
    B[LINE message<br/>POST /webhook] --> C
    C[Clean + tokenize<br/>PyThaiNLP newmm] --> D[TF-IDF +<br/>Logistic Regression]
    C --> E[Aspect keyword rules]
    D --> F[(SQLite<br/>data/reviews.db)]
    E --> F
    F --> G[Daily report<br/>CLI / GET /stats/daily]
    D --> H[LINE reply]
```

## Quick start
```bash
pip install -r requirements-notebooks.txt   # API deps + Hugging Face download (training needs it)
cp .env.example .env              # LINE_ACCESS_TOKEN, LINE_CHANNEL_SECRET (only needed for /webhook)
python -m src.train               # downloads Wongnai/Wisesight from Hugging Face, writes models/*.pkl
uvicorn src.main:app --reload
curl -X POST localhost:8000/analyze -H 'Content-Type: application/json' -d '{"text":"ส่งช้ามาก อาหารเย็นหมด"}'
```

Docker (mount the trained models; they are not baked into the image):
```bash
docker build -t mlproj .
docker run -p 8000:8000 --env-file .env -v "$PWD/models:/app/models" -v "$PWD/data:/app/data" mlproj
```

## Pipeline
```bash
python -m src.pipeline ingest data/reviews.csv   # header: review[,created_at]; headerless: review in col 0
python -m src.pipeline report --days 14
```
- Stored in `data/reviews.db` (override with `DB_PATH`). Duplicates are skipped, so re-running is safe.
- LINE messages received on `/webhook` are stored too (text only, no user id).
- `GET /stats/daily` returns the same report as the CLI.

### Known limitation: dates
The training/demo data ([Wongnai](https://huggingface.co/datasets/Wongnai/wongnai_reviews)) has **no timestamps**. Rows without `created_at` get the ingest time, so undated data lands on a single day and the daily report shows no trend. A real trend needs dated data: a CSV with a `created_at` column, or LINE messages collected over time.

## Results

Sentiment (positive / negative). Labels: Wongnai stars ≥4 → positive, ≤2 → negative, 3 dropped; Wisesight `pos` / `neg`, `neu` and `q` dropped.
Wongnai train: 25,039 reviews after dropping duplicates and train rows that also appear in test. Hyperparameters were chosen on a 10% stratified validation split. The test sets were only used for the final numbers.

| model | Wongnai test macro-F1 (95% CI) | Wongnai F1 negative | Wisesight macro-F1 (cross-domain) | latency p50 | size |
|---|---|---|---|---|---|
| always "positive" | 0.485 | 0.000 | 0.292 | – | – |
| **TF-IDF + LR** | **0.862** (0.839–0.884) | **0.739** | 0.473 | 0.6 ms | 1.2 MB |

Wongnai test = the published test split (4,296 reviews, 6% negative). Wisesight = its test split, pos/neg only (1,161 messages, 59% negative). Latency is per review on CPU and includes text cleaning. Full write-up: [notebooks/02_baseline.ipynb](notebooks/02_baseline.ipynb).

Read this carefully:
- The data is imbalanced (~94% positive in test). "Always positive" already scores 94% accuracy, so **macro-F1 and negative-class F1 are the numbers to look at**, not accuracy.
- The weak spot is recall on negatives: 31% of negative Wongnai reviews are predicted positive. They are mostly mixed reviews that open with praise.
- Wisesight (social media, many topics) shows how badly the model transfers outside restaurant reviews. 80% of Wisesight negatives are predicted positive.
- The earlier README reported macro-F1 0.83 on a random 80/20 split of the training file. That measured something different, so it can't be compared with the numbers above.
- Aspects (food / rider / delay / price) are a keyword-rule baseline in `src/aspects.py`. They have **not been measured** against labeled data yet.

### Measure on your own labels
```bash
python -m src.make_label_set 200   # -> data/to_label.csv (delivery-related Wongnai reviews, ~119 found)
# fill sentiment (1/0) and food/rider/delay/price (1/0) by hand, save as data/labeled.csv
python -m src.evaluate             # sentiment report + per-aspect F1
```

## Data source
- [Wongnai reviews](https://huggingface.co/datasets/Wongnai/wongnai_reviews): restaurant reviews from wongnai.com, LGPL-3.0.
- [Wisesight Sentiment](https://huggingface.co/datasets/pythainlp/wisesight_sentiment): Thai social-media messages, CC0.

Both are downloaded from Hugging Face by `src/data/load.py` and are not committed. `data/train.csv` (the Wongnai train split as CSV) is only used by `src.make_label_set`. Check the license terms before showing or redistributing the raw text.

## Test / CI
```bash
pytest
```
GitHub Actions (`.github/workflows/ci.yml`) runs pytest on every push and pull request. Tests do not need trained models.

## Layout
```
src/cleaner.py      Thai text cleaning + tokenization (keeps negation words)
src/train.py        train the served model (same splits/settings as notebook 02) + print metrics
src/predictor.py    model wrapper shared by API and pipeline
src/aspects.py      aspect keyword baseline
src/db.py           SQLite store + daily report
src/pipeline.py     ingest / report CLI
src/main.py         FastAPI app + LINE webhook
src/evaluate.py     evaluation on hand-labeled reviews
src/data/load.py    load Wongnai / Wisesight from Hugging Face (Parquet) + label mapping
src/data/splits.py  fixed train / val / test splits shared by every model
src/models/baseline.py  TF-IDF + Logistic Regression pipeline (used by train.py and notebooks)
src/metrics.py      macro-F1 / per-class F1 + bootstrap confidence interval
src/viz.py          Thai font setup for matplotlib
notebooks/01_eda.ipynb   EDA + the data decisions it led to (pip install -r requirements-notebooks.txt)
notebooks/02_baseline.ipynb  baseline re-evaluated on the published test set + Wisesight -> reports/metrics/baseline.json
reports/figures/    figures exported by the notebooks
```
