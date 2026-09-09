"""Lab 4: per-model Arabic normalisation profiles."""

import re
from dataclasses import dataclass

from camel_tools.disambig.mle import MLEDisambiguator
from camel_tools.tokenizers.morphological import MorphologicalTokenizer


@dataclass(frozen=True)
class ArabicProfile:
    name: str
    dediacritize: bool = False


_ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")

_MLE = None
_D3TOK = None


def normalize_arabic(text: str, profile: ArabicProfile) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    normalized = text

    if profile.dediacritize:
        normalized = _ARABIC_DIACRITICS.sub("", normalized)

    if profile.name == "bayan_ar_v1":
        normalized = normalized.replace("ـ", "")
        normalized = re.sub(r"[أإآ]", "ا", normalized)
        normalized = normalized.replace("ؤ", "و")
        normalized = normalized.replace("ئ", "ي")
        normalized = normalized.replace("ى", "ي")
        normalized = normalized.replace("ة", "ه")
    else:
        raise ValueError(f"Unknown Arabic normalization profile: {profile.name}")

    return normalized


def segment(text: str) -> list[str]:
    global _MLE, _D3TOK

    if _MLE is None:
        _MLE = MLEDisambiguator.pretrained()
        _D3TOK = MorphologicalTokenizer(_MLE, scheme="d3tok", split=True)

    words = text.split()
    segmented = _D3TOK.tokenize(words)

    return [token.replace("+", "") for token in segmented if token]
