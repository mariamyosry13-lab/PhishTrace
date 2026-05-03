import warnings

_pipeline = None
_available = False


def load() -> None:
    """Load the HuggingFace BERT phishing classifier. Called once at startup."""
    global _pipeline, _available
    try:
        from transformers import pipeline as hf_pipeline
        # Prefer safetensors to avoid CVE-2025-32434 (torch.load restriction on torch < 2.6).
        # Falls back to the standard loader if safetensors are not available on the hub.
        try:
            _pipeline = hf_pipeline(
                "text-classification",
                model="ealvaradob/bert-finetuned-phishing",
                truncation=True,
                max_length=512,
                model_kwargs={"use_safetensors": True},
            )
        except Exception:
            _pipeline = hf_pipeline(
                "text-classification",
                model="ealvaradob/bert-finetuned-phishing",
                truncation=True,
                max_length=512,
            )
        _available = True
        print("BERT phishing classifier loaded.")
    except Exception as exc:
        warnings.warn(f"BERT classifier unavailable — falling back to RF only: {exc}")
        _available = False


def predict_proba(url: str) -> float | None:
    """Return phishing probability in [0, 1], or None if the model is unavailable."""
    if not _available or _pipeline is None:
        return None
    try:
        result = _pipeline(url[:512])[0]
        label = result["label"].lower()
        score = float(result["score"])
        # label_1 / phishing / malicious → score is P(phishing)
        # label_0 / benign / legitimate  → score is P(safe), so invert
        return score if label in ("label_1", "phishing", "malicious", "1") else 1.0 - score
    except Exception:
        return None


def is_available() -> bool:
    return _available
