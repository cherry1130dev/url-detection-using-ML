# URL Detection Project

This project detects whether a URL is **benign (safe)** or **malicious (unsafe)** using a combination of manual feature extraction and character-level vectorization with a Random Forest classifier.

---

## Features

- Extracts manual features from URLs such as:
  - URL length
  - Hostname length
  - Path length
  - HTTPS presence
  - Number of digits
  - Special characters count
  - Number of subdomains
  - Presence of unsafe keywords (porn, sex, bank, login, etc.)
- Uses **character n-gram vectorization** (3-5 grams) for URLs.
- Combines manual and vectorized features for robust classification.
- Trained with a **Random Forest Classifier**.
- Supports prediction for new URLs.

---

## Requirements

Python 3.10 or above.  

**Install dependencies using:**

pip install -r requirements.txt

first run the "**python main.py**" (for fenerating modals using dataset.csv)

after the model is created then run "**python app.py**" (it creating a server in localhost)

then click the ip address link like "**Running on http://127.0.0.1:**"

for stoping server Press "**CTRL+C**" to quit

---

## **Developer & Contact Info**

Website: https://easitronics.netlify.app/
Developer: Charan Cherry
Email: charancherry1130@gmail.com
Phone: +91 7989604815
LinkedIn: https://www.linkedin.com/in/charan-teja-2a08b12b4

```bash
pip install -r requirements.txt
