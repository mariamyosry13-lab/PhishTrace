# 🎣 PhishTrace

Phishing detection project + simple campaign analysis

---

## 📁 Project Structure

```
PhishTrace/
├── data/
│   ├── raw/          # raw data (PhishTank, OpenPhish, Kaggle)
│   └── processed/    # cleaned data + features
├── frontend/
├── notebooks/        # experiments / quick analysis
├── src/
│   ├── data/         # collect + clean data
│   ├── features/     # extract features from URLs
│   ├── models/       # training + evaluation
│   ├── explainability/ # SHAP / LIME
│   ├── clustering/   # detect phishing campaigns
│   └── api/          # Flask API
├── tests/            # unit tests (basic)
├── models/           # saved models
└── reports/          # plots / outputs
```

---

## ⚙️ Quick Start

```bash
# create virtual env
python -m venv venv
source venv/bin/activate   # windows: venv\Scripts\activate

# install deps
pip install -r requirements.txt

# step 1: collect data
python src/data/collect.py

# step 2: extract features
python src/features/extract.py

# step 3: train model
python src/models/train.py

# run api
python src/api/app.py
```

---

## 🚀 Workflow

* collect data from different sources
* clean + prepare it
* extract features from URLs
* train model
* try explain predictions (SHAP / LIME)
* group similar attacks (campaign detection)
* expose everything through API

---

## 📌 Notes

* still improving feature quality
* model performance will be tuned later
* UI not final yet

---

## 📊 Status

* data collection → in progress
* rest → not done yet
