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
    "num_at", "num_question", "num_equals", "num_percent", "num_digits",
    "has_ip", "has_https", "num_subdomains",
    "hostname_length", "path_length", "double_slash", "has_at_in_url",
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
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")

np.save(os.path.join(MODELS_DIR, "test_indices.npy"),
        np.where(np.isin(np.arange(len(df)),
                 df.index[len(X_train):]))[0])

# ── Scale ───────────────────────────────────────────────────────────────────
scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
print("Scaler saved ✅")

pos_weight = legit_count / max(phishing_count, 1)

# ── Models ──────────────────────────────────────────────────────────────────
models_dict = {
    "logistic_regression": LogisticRegression(
        max_iter=1000, class_weight="balanced",
        C=0.1, random_state=42
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_leaf=5,
        class_weight="balanced_subsample", random_state=42, n_jobs=-1
    ),
    "xgboost": XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
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
                            cv=cv, scoring="f1", n_jobs=-1)
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

    # ✅ FIX: بنخزن dict كامل مش string
    results[name] = {
        "model"  : mdl,
        "cv_f1"  : float(cv_f1.mean()),
        "test_f1": float(f1),
        "auc"    : float(auc),
        "fpr"    : float(fpr),
        "fp"     : fp,
        "fn"     : fn,
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
print("\nAll models saved ✅")