import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, roc_auc_score,
                             f1_score, precision_score, recall_score,
                             accuracy_score)
from xgboost import XGBClassifier
import joblib

# ── Paths ──────────────────────────────────────────────────────────────────
FEATURES_CSV = "data/processed/phishtrace_features.csv"
MODELS_DIR   = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURE_COLS = [
    "url_length", "num_dots", "num_hyphens", "num_underscores", "num_slashes",
    "num_at", "num_question", "num_equals", "num_percent",
    "num_digits_in_domain", "num_digits_in_path", "last_path_segment_is_integer",
    "has_ip", "has_https", "num_subdomains",
    "hostname_length", "path_length", "double_slash",
    "num_suspicious_words"
]

# ── Load data ──────────────────────────────────────────────────────────────
print("Loading features...")
df = pd.read_csv(FEATURES_CSV)

missing = [c for c in FEATURE_COLS if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

X = df[FEATURE_COLS].values
y = df["label"].values

phishing_count = int(y.sum())
legit_count    = int((y == 0).sum())
total          = len(y)
print(f"Dataset   : {total:,} samples")
print(f"Phishing  : {phishing_count:,} ({phishing_count/total*100:.1f}%)")
print(f"Legit     : {legit_count:,} ({legit_count/total*100:.1f}%)")

# ── Split ───────────────────────────────────────────────────────────────────
indices = np.arange(len(df))
train_idx, test_idx = train_test_split(
    indices, test_size=0.2, random_state=42, stratify=y
)
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")

np.save(os.path.join(MODELS_DIR, "test_indices.npy"), test_idx)

# ── Scale ───────────────────────────────────────────────────────────────────
scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
print("Scaler saved")

pos_weight = legit_count / max(phishing_count, 1)

# ── Models ──────────────────────────────────────────────────────────────────
models_dict = {
    "logistic_regression": LogisticRegression(
        max_iter=1000, class_weight="balanced",
        C=0.1, random_state=42
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_leaf=5,
        class_weight="balanced_subsample", random_state=42, n_jobs=1
    ),
    "xgboost": XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        subsample=0.7, colsample_bytree=0.8,
        scale_pos_weight=pos_weight, n_jobs=1,
        tree_method="hist",
        random_state=42, eval_metric="logloss", verbosity=0
    ),
}

# ✅ FIX: results بيخزن dicts مش strings
results = {}
cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, mdl in models_dict.items():
    print(f"\n{'='*50}")
    print(f"Training: {name}")

    cv_f1 = cross_val_score(mdl, X_train_s, y_train,
                            cv=cv, scoring="f1", n_jobs=1)
    print(f"  CV F1  : {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

    mdl.fit(X_train_s, y_train)

    y_pred = mdl.predict(X_test_s)
    y_prob = mdl.predict_proba(X_test_s)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_prob)
    fp   = int(((y_pred == 1) & (y_test == 0)).sum())
    fn   = int(((y_pred == 0) & (y_test == 1)).sum())
    fpr  = fp / max(int((y_test == 0).sum()), 1)

    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  AUC       : {auc:.4f}")
    print(f"  FPR       : {fpr:.4f}")
    print(f"  FP: {fp}  |  FN: {fn}")
    print(classification_report(y_test, y_pred,
                                target_names=["Legit", "Phishing"]))

    joblib.dump(mdl, os.path.join(MODELS_DIR, f"{name}.pkl"))

    results[name] = {
        "model"    : mdl,
        "cv_f1"    : float(cv_f1.mean()),
        "accuracy" : float(acc),
        "precision": float(prec),
        "recall"   : float(rec),
        "test_f1"  : float(f1),
        "auc"      : float(auc),
        "fpr"      : float(fpr),
        "fp"       : fp,
        "fn"       : fn,
    }

# ── Best model ──────────────────────────────────────────────────────────────
# ✅ FIX: score_model بتاخد dict صح دلوقتي
def score_model(name):
    r = results[name]
    return r["test_f1"] * (1 - r["fpr"])

best_name = max(results, key=score_model)

# احفظ الـ best model مباشرة من الـ results dict
joblib.dump(results[best_name]["model"],
            os.path.join(MODELS_DIR, "best_model.pkl"))

print(f"\n{'='*50}")
print(f"Best model : {best_name}")
print(f"  Test F1  : {results[best_name]['test_f1']:.4f}")
print(f"  AUC      : {results[best_name]['auc']:.4f}")
print(f"  FPR      : {results[best_name]['fpr']:.4f}")
print(f"  FP       : {results[best_name]['fp']}  (safe URLs wrongly flagged)")
print(f"  FN       : {results[best_name]['fn']}  (phishing missed)")

import json
best = results[best_name]
eval_results = {
    "model"    : best_name,
    "accuracy" : round(best["accuracy"],  4),
    "precision": round(best["precision"], 4),
    "recall"   : round(best["recall"],    4),
    "f1"       : round(best["test_f1"],   4),
    "auc"      : round(best["auc"],       4),
    "fpr"      : round(best["fpr"],       4),
    "fp"       : best["fp"],
    "fn"       : best["fn"],
}
eval_path = os.path.join(MODELS_DIR, "evaluation_results.json")
with open(eval_path, "w") as f:
    json.dump(eval_results, f, indent=2)
print(f"\nEvaluation results saved -> {eval_path}")
print("\nAll models saved")