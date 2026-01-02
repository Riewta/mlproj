import pandas as pd
import os
import joblib
import csv
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from cleaner import ThaiTextCleaner


# special function for recording models
def identity_tokenizer(text):
    return text.split()


# path setting
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'train.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

print(f"System Check")
print(f"Project Root: {BASE_DIR}")
print(f"Loading local data from: {DATA_PATH}")

# load data (specific data)
try:
    print("Reading raw file content")
    data_list = []

    # read the file as one paragraph
    with open(DATA_PATH, 'r', encoding='utf-8-sig', errors='ignore') as f:
        content = f.read()

    # separate line (รองรับทั้ง \r และ \n)
    lines = content.replace('\r\n', '\n').replace('\r', '\n').split('\n')
    print(f"Total lines found: {len(lines)}")

    for line in lines:
        if not line.strip(): continue

        # searching comma
        parts = line.rsplit(',', 1)

        if len(parts) == 2:
            review_raw = parts[0].strip().strip('"')  #delete "";
            rating_raw = parts[1].strip().strip('"')

            if rating_raw.isdigit():
                data_list.append({'review': review_raw, 'rating': int(rating_raw)})

    df = pd.DataFrame(data_list)

    if df.empty:
        # if the last cha of comma not a number move to the first one
        print("Trying alternative split...")
        for line in lines:
            parts = line.split(',', 1)
            if len(parts) == 2:
                r_raw = parts[0].strip().strip('"')
                if r_raw.isdigit():
                    data_list.append({'rating': int(r_raw), 'review': parts[1].strip().strip('"')})
        df = pd.DataFrame(data_list)

    # build Label
    df['label'] = df['rating'].apply(lambda x: 1 if x >= 4 else 0)

    print(f"Success! Loaded {len(df)} rows.")

except Exception as e:
    print(f"Error during loading: {e}")
    exit()

# Preprocessing (cut Th words)
cleaner = ThaiTextCleaner()
print("\n--- Processing ---")
print("Cleaning Thai text (ใช้เวลา 3-5 นาที ห้ามปิดโปรแกรม)")

# to cutting the word and merge back also separator by space
df['cleaned_review'] = df['review'].apply(lambda x: " ".join(cleaner.clean(x)))

# train models
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_review'], df['label'], test_size=0.2, random_state=42
)

print("Extracting features and Training AI...")
# เปลี่ยนมาใช้ฟังก์ชัน identity_tokenizer แทน lambda เพื่อให้บันทึกไฟล์ได้
tfidf = TfidfVectorizer(tokenizer=identity_tokenizer, lowercase=False, max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# summary of accuracy
print("\n Model Evaluation")
y_pred = model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))

# recording the results
joblib.dump(model, os.path.join(MODEL_DIR, 'sentiment_model.pkl'))
joblib.dump(tfidf, os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))

print(f"\n ภารกิจ สำเร็จ!")
print(f"โมเดลถูกบันทึกไว้ที่: {MODEL_DIR}")