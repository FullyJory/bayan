"""Lab 1 starter: versioned bilingual preprocessing for Bayan."""

import re
import unicodedata

PREPROC_VERSION = "1.2.0"


def normalize(text: str) -> str:
    """Return deterministic Bayan normalisation while preserving task signal."""
    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Remove Tatweel
    text = text.replace("ـ", "")

    # Remove HTML remnants
    text = re.sub(r"<[^>]+>", " ", text)

    # Reduce repeated characters
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    # Clean repeated whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def mask_pii(text: str) -> str:
    """Mask supported phone numbers and Saudi national-ID-shaped values."""
    # Mask Saudi phone numbers
    text = re.sub(r"(?<!\d)(?:\+966|966|0)5\d{8}(?!\d)", "<PHONE>", text)

    # Mask Saudi national-ID-shaped values
    text = re.sub(r"(?<!\d)[12]\d{9}(?!\d)", "<NATIONAL_ID>", text)

    return text


def preprocess(text: str) -> str:
    """Apply the shared train/eval/serve preprocessing contract."""
    text = mask_pii(text)
    text = normalize(text)

    return text
