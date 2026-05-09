"""Optional BERT phishing classifier (40% weight in the ensemble).  Returns None
when unavailable so callers always get a result.  Model: ealvaradob/bert-finetuned-phishing."""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_pipeline  = None
_available = False

MODEL_NAME = os.environ.get("BERT_MODEL", "ealvaradob/bert-finetuned-phishing")
_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "bert_cache"
_MAX_LEN   = 512          # BERT hard token limit


def is_available() -> bool:
    """Return True if the BERT model loaded successfully and can score URLs."""
    return _available


def load() -> None:
    """Try GPU (fp16), then CPU (fp32).  Never raises; sets _available=False on failure."""
    global _pipeline, _available

    try:
        import torch
        from transformers import pipeline as hf_pipeline
    except ImportError as exc:
        logger.warning(
            "BERT ensemble disabled — missing dependency (%s). "
            "Install: pip install 'transformers>=4.40.0' 'torch>=2.6.0'",
            exc,
        )
        return

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cuda_ok = torch.cuda.is_available()
    if cuda_ok:
        gpu_name = torch.cuda.get_device_name(0)
        logger.info("BERT: CUDA available — %s", gpu_name)

    attempts = [(0, "GPU", torch.float16)] if cuda_ok else []
    attempts.append((-1, "CPU", torch.float32))

    for device, label, dtype in attempts:
        try:
            pipe = hf_pipeline(
                "text-classification",
                model        = MODEL_NAME,
                tokenizer    = MODEL_NAME,
                device       = device,
                cache_dir    = str(_CACHE_DIR),
                model_kwargs = {"torch_dtype": dtype},
            )
            # Warm-up: one dummy inference to catch lazy-load errors early
            _ = pipe("http://test.com", truncation=True, max_length=_MAX_LEN)

            _pipeline  = pipe
            _available = True
            logger.info(
                "BERT classifier ready — model=%s  device=%s  dtype=%s",
                MODEL_NAME, label, str(dtype).split(".")[-1],
            )
            return

        except Exception as exc:
            if device == 0:
                logger.warning(
                    "BERT GPU load failed (%s) — retrying on CPU", exc
                )
            else:
                logger.error(
                    "BERT classifier could not be loaded on %s: %s. "
                    "Analysis will use the RF model alone (BERT weight dropped).",
                    label, exc,
                )


def predict_proba(url: str) -> float | None:
    """Return P(phishing) in [0, 1], or None if BERT is unavailable."""
    if not _available or _pipeline is None:
        return None

    try:
        result = _pipeline(
            url[:_MAX_LEN],
            truncation = True,
            max_length = _MAX_LEN,
        )[0]

        label = result["label"].upper()
        score = float(result["score"])

        # convert output label to P(phishing)
        is_phishing = "1" in label or "PHISH" in label
        return score if is_phishing else 1.0 - score

    except Exception as exc:
        logger.warning("BERT predict_proba error url=%r: %s", url[:60], exc)
        return None
