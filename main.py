import pandas as pd
import re
import pickle
from urllib.parse import urlparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ===== Step 1: Load Dataset =====
data = pd.read_csv("dataset.csv")

# ===== Step 2: Manual Feature Extraction =====
def extract_manual_features(url):
    parsed = urlparse(url)
    
    url_length = len(url)
    hostname_length = len(parsed.hostname) if parsed.hostname else 0
    path_length = len(parsed.path)
    has_https = 1 if parsed.scheme == "https" else 0
    num_digits = len(re.findall(r'\d', url))
    num_special_chars = len(re.findall(r'[@!#$%^&*()?/\\]', url))
    num_subdomains = len(parsed.hostname.split(".")) - 2 if parsed.hostname else 0
    
    unsafe_keywords = [
        "porn", "sex", "xhamster", "xvideos", "strip", "nude", "xxx",
        "phish", "bank", "login", "verify", "secure", "update", "payment"
    ]
    unsafe_keywords_flag = 1 if any(k in url.lower() for k in unsafe_keywords) else 0

    return [
        url_length, hostname_length, path_length, has_https,
        num_digits, num_special_chars, num_subdomains, unsafe_keywords_flag
    ]

# ===== Step 3: Combine Manual + Vectorized Features =====
manual_features = [extract_manual_features(url) for url in data["url"]]

# Create vectorizer for URL text
vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 5))
vectorized_features = vectorizer.fit_transform(data["url"])

# Combine manual and vectorized features
import numpy as np
from scipy.sparse import hstack

X = hstack([np.array(manual_features), vectorized_features])
y = data["label"]

# ===== Step 4: Train/Test Split =====
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===== Step 5: Train Model =====
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ===== Step 6: Evaluate =====
y_pred = model.predict(X_test)
print(f"✅ Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ===== Step 7: Save Model & Vectorizer =====
with open("url_detector_model.pkl", "wb") as f:
    pickle.dump(model, f)

joblib.dump(vectorizer, "vectorizer.pkl")
print("✅ Model saved as url_detector_model.pkl")
print("✅ Vectorizer saved as vectorizer.pkl")

# ===== Step 8: Test Predictions =====
test_urls = [
    "https://www.google.com",
    "http://192.168.0.1/login",
    "http://secure-bank-login.com",
    "https://trustedwebsite.net",
    "https://xhamster.desi/",
    "https://www.pornhat.com/",
    "https://stripchatgirls.com/model123"
]

for url in test_urls:
    manual_feat = extract_manual_features(url)
    vectorized_feat = vectorizer.transform([url])
    combined_feat = hstack([np.array(manual_feat).reshape(1, -1), vectorized_feat])
    
    prediction = model.predict(combined_feat)[0]
    if prediction == "benign":
        print(f"{url} --> Benign safe ✅")
    else:
        print(f"{url} --> Malicious unsafe 🚨")
