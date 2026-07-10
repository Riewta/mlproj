import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from cleaner import ThaiTextCleaner


def identity_tokenizer(text):
    return text.split()


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'train.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

os.makedirs(MODEL_DIR, exist_ok=True)

print(f"Project Root: {BASE_DIR}")
print(f"Loading data from: {DATA_PATH}")

try:
    data_list = []
    with open(DATA_PATH, 'r', encoding='utf-8-sig', errors='ignore') as f:
        content = f.read()

    lines = content.replace('\r\n', '\n').replace('\r', '\n').split('\n')
    print(f"Total lines found: {len(lines)}")

    for line in lines:
        if not line.strip():
            continue
        parts = line.rsplit(',', 1)
        if len(parts) == 2:
            review_raw = parts[0].strip().strip('"')
            rating_raw = parts[1].strip().strip('"')
            if rating_raw.isdigit():
                data_list.append({'review': review_raw, 'rating': int(rating_raw)})

    df = pd.DataFrame(data_list)

    if df.empty:
        print("Trying alternative split (rating first)...")
        data_list = []
        for line in lines:
            parts = line.split(',', 1)
            if len(parts) == 2:
                r_raw = parts[0].strip().strip('"')
                if r_raw.isdigit():
                    data_list.append({'rating': int(r_raw), 'review': parts[1].strip().strip('"')})
        df = pd.DataFrame(data_list)

    df['label'] = df['rating'].apply(lambda x: 1 if x >= 4 else 0)
    pos = df['label'].sum()
    neg = len(df) - pos
    print(f"Loaded {len(df)} rows | Positive: {pos} | Negative: {neg}")

except Exception as e:
    print(f"Error loading data: {e}")
    exit()

cleaner = ThaiTextCleaner()
print("\nCleaning Thai text (this may take a few minutes)...")
df['cleaned_review'] = df['review'].apply(lambda x: " ".join(cleaner.clean(x)))

X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_review'], df['label'],
    test_size=0.2, random_state=42, stratify=df['label']
)

print("Training model...")
tfidf = TfidfVectorizer(
    tokenizer=identity_tokenizer,
    lowercase=False,
    max_features=10000,
    ngram_range=(1, 2),
    sublinear_tf=True,
)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

model = LogisticRegression(
    max_iter=1000,
    C=5,
    class_weight='balanced',
)
model.fit(X_train_tfidf, y_train)

print("\nModel Evaluation:")
y_pred = model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

joblib.dump(model, os.path.join(MODEL_DIR, 'sentiment_model.pkl'))
joblib.dump(tfidf, os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
print(f"\nModels saved to: {MODEL_DIR}")
