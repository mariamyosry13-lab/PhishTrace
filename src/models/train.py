import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, f1_score)
from xgboost import XGBClassifier
import joblib

# ── Paths ──────────────────────────────────────────────
FEATURES_CSV = "data/processed/phishtrace_features.csv"
MODELS_DIR   = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURE_COLS = [
    "url_length","num_dots","num_hyphens","num_underscores","num_slashes",
    "num_at","num_question","num_equals","num_percent","num_digits",
    "has_ip","has_https","has_suspicious_word","num_subdomains",
    "hostname_length","path_length","double_slash","has_at_in_url",
    "num_suspicious_words"
]

# ── Load data ───────────────────────────────────────────
print("Loading features...")
df = pd.read_csv(FEATURES_CSV)
X  = df[FEATURE_COLS].values
y  = df["label"].values

# ── Split ───────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# Save test indices for later use
np.save(os.path.join(MODELS_DIR, "test_indices.npy"),
        np.where(np.isin(np.arange(len(df)), 
                 df.index[len(X_train):]))[0])

# ── Scale ───────────────────────────────────────────────
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

# ── Models ──────────────────────────────────────────────
models = {
    "logistic_regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42),
    "random_forest": RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1),
    "xgboost": XGBClassifier(
        n_estimators=100, scale_pos_weight=(y_train==0).sum()/(y_train==1).sum(),
        random_state=42, eval_metric="logloss", verbosity=0),
}

results = {}

for name, model in models.items():
    print(f"\n{'='*40}")
    print(f"Training: {name}")

    # Cross-validation on train set
    cv_scores = cross_val_score(model, X_train, y_train,
                                cv=5, scoring="f1", n_jobs=-1)
    print(f"CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Train on full train set
    model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = model.predict(X_test)
    f1     = f1_score(y_test, y_pred)
    auc    = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    print(f"Test F1:  {f1:.4f}")
    print(f"Test AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred,
                                target_names=["Legit","Phishing"]))

    # Save model
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    joblib.dump(model, path)

    results[name] = {"cv_f1": cv_scores.mean(), "test_f1": f1, "auc": auc}

# ── Best model ──────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["test_f1"])
best_model = joblib.load(os.path.join(MODELS_DIR, f"{best_name}.pkl"))
joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.pkl"))

print(f"\n{'='*40}")
print(f"Best model: {best_name}")
print(f"  Test F1 : {results[best_name]['test_f1']:.4f}")
print(f"  Test AUC: {results[best_name]['auc']:.4f}")
print("All models saved to models/")