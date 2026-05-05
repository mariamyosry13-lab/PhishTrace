import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')

import joblib, numpy as np, pandas as pd
from config import FEATURE_COLS, THRESHOLD_DANGEROUS, THRESHOLD_SUSPICIOUS
from features.unified_extractor import extract_all

scaler = joblib.load('models/scaler.pkl')
model  = joblib.load('models/best_model.pkl')

urls = [
    ('amazoon.net typosquat',     'https://amazoon.net/deals'),
    ('login-portal.xyz',          'http://login-portal.xyz/account'),
    ('goog1e.com typosquat',      'http://goog1e.com/signin'),
    ('wikipedia Phishing article','https://en.wikipedia.org/wiki/Phishing'),
]

RULE_KEYS = [
    'is_typosquat','min_levenshtein','has_ip','brand_in_subdomain',
    'tld_suspicious','has_at_in_url','num_subdomains','hostname_entropy',
    'has_https','num_suspicious_words','brand_impersonation',
]
ML_KEYS = [
    'url_length','num_dots','num_hyphens','path_length',
    'num_digits_in_domain','num_digits_in_path','last_path_segment_is_integer',
    'double_slash','hostname_length',
]

for label, url in urls:
    feats = extract_all(url)
    X     = pd.DataFrame([feats])[FEATURE_COLS]
    Xs    = scaler.transform(X)
    rf    = float(model.predict_proba(Xs)[0][1])

    # ── simulate rule_based_boost ─────────────────────────────────────────────
    boost = 0.0
    fired = []
    lev = feats.get('min_levenshtein', 99)
    if feats.get('is_typosquat', 0):
        boost += 0.15; fired.append(f'typosquat(lev={lev}) +0.15')
    if feats.get('has_ip', 0):
        boost += 0.15; fired.append('has_ip +0.15')
    if feats.get('brand_in_subdomain', 0):
        boost += 0.15; fired.append('brand_in_subdomain +0.15')
    if feats.get('tld_suspicious', 0):
        boost += 0.10; fired.append('tld_suspicious +0.10')
    if feats.get('has_at_in_url', 0):
        boost += 0.10; fired.append('has_at_in_url +0.10')
    if feats.get('num_subdomains', 0) > 3:
        boost += 0.05; fired.append(f'subdomains({feats["num_subdomains"]}) +0.05')
    entropy = feats.get('hostname_entropy', 0)
    if entropy > 4.0:
        boost += 0.05; fired.append(f'entropy({entropy:.2f}) +0.05')

    raw_boost  = boost
    boost      = min(boost, 0.30)

    hard_evidence = (
        feats.get('has_ip', 0) or
        feats.get('brand_in_subdomain', 0) or
        feats.get('has_at_in_url', 0) or
        feats.get('num_subdomains', 0) > 2
    )

    THRESHOLD_DANGEROUS_LOCAL = THRESHOLD_DANGEROUS
    if rf >= THRESHOLD_DANGEROUS_LOCAL and not hard_evidence:
        rf    = THRESHOLD_DANGEROUS_LOCAL - 0.05   # cap in Suspicious band
        boost = 0.0
    elif rf >= THRESHOLD_DANGEROUS_LOCAL:
        boost = 0.0
    elif rf < 0.40:
        boost *= 0.30
    elif rf < 0.65:
        boost *= 0.60

    # is_clean dampening
    is_clean = (
        feats.get('has_https', 0) == 1 and
        feats.get('tld_suspicious', 0) == 0 and
        feats.get('num_suspicious_words', 0) == 0 and
        feats.get('has_ip', 0) == 0 and
        feats.get('is_typosquat', 0) == 0 and
        feats.get('brand_in_subdomain', 0) == 0 and
        feats.get('has_at_in_url', 0) == 0
    )
    raw_score = rf
    if is_clean:
        raw_score *= 0.40

    final = min(raw_score + boost, 1.0)
    verdict = 'Dangerous' if final >= THRESHOLD_DANGEROUS else 'Suspicious' if final >= THRESHOLD_SUSPICIOUS else 'Safe'

    print(f'\n{"="*60}')
    print(f'  {label}')
    print(f'  URL: {url}')
    print(f'  rf_score = {rf:.4f}  |  is_clean = {is_clean}')
    print(f'  rules fired: {fired}')
    print(f'  raw_boost={raw_boost:.2f}  dampened_boost={boost:.4f}')
    print(f'  final = {final:.4f}  ->  {verdict}')
    print(f'  ML features: ' + str({k: feats.get(k) for k in ML_KEYS}))
    print(f'  Rule features: ' + str({k: feats.get(k) for k in RULE_KEYS}))
