import pickle
import joblib
import numpy as np
from flask import Flask, render_template, request
from urllib.parse import urlparse
import re
from scipy.sparse import hstack

# Load model & vectorizer
model = pickle.load(open("url_detector_model.pkl", "rb"))
vectorizer = joblib.load("vectorizer.pkl")

# Manual feature extraction (same as in main.py)
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

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    if request.method == "POST":
        url = request.form["url"]

        # Get manual features
        manual_feat = extract_manual_features(url)

        # Get vectorized features
        vectorized_feat = vectorizer.transform([url])

        # Combine them exactly like training
        combined_feat = hstack([np.array(manual_feat).reshape(1, -1), vectorized_feat])

        # Predict
        pred = model.predict(combined_feat)[0]
        if pred == "benign":
            prediction = f"{url} --> Benign safe ✅"
        else:
            prediction = f"{url} --> Malicious unsafe ❌🚨"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
