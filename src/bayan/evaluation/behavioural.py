"""Lab 6: behavioural evaluation for the Bayan topic classifier."""

from pathlib import Path

import pandas as pd


DEFAULT_TEMPLATES = Path("data/eval/behavioural_templates.csv")


MFT_CASES = [
    {
        "test_id": "MFT-001",
        "lang": "en",
        "text": "There is a dangerous pothole in the road.",
        "expected": "roads",
    },
    {
        "test_id": "MFT-002",
        "lang": "ar",
        "text": "الطريق يحتاج إلى صيانة بسبب حفرة كبيرة.",
        "expected": "roads",
    },
    {
        "test_id": "MFT-003",
        "lang": "en",
        "text": "The street lights are not working.",
        "expected": "lighting",
    },
    {
        "test_id": "MFT-004",
        "lang": "ar",
        "text": "إنارة الشارع لا تعمل.",
        "expected": "lighting",
    },
    {
        "test_id": "MFT-005",
        "lang": "en",
        "text": "The waste bin has not been emptied.",
        "expected": "waste",
    },
    {
        "test_id": "MFT-006",
        "lang": "ar",
        "text": "حاوية النفايات ممتلئة ولم يتم تفريغها.",
        "expected": "waste",
    },
    {
        "test_id": "MFT-007",
        "lang": "en",
        "text": "There is a water leak in the neighbourhood.",
        "expected": "water",
    },
    {
        "test_id": "MFT-008",
        "lang": "ar",
        "text": "يوجد تسرب مياه في الحي.",
        "expected": "water",
    },
    {
        "test_id": "MFT-009",
        "lang": "en",
        "text": "I paid my bill but it still shows as unpaid.",
        "expected": "billing",
    },
    {
        "test_id": "MFT-010",
        "lang": "ar",
        "text": "دفعت الفاتورة لكنها ما زالت تظهر كغير مسددة.",
        "expected": "billing",
    },
    {
        "test_id": "MFT-011",
        "lang": "en",
        "text": "The mobile application crashes when I sign in.",
        "expected": "digital_services",
    },
    {
        "test_id": "MFT-012",
        "lang": "ar",
        "text": "التطبيق يتوقف عند تسجيل الدخول.",
        "expected": "digital_services",
    },
    {
        "test_id": "MFT-013",
        "lang": "en",
        "text": "My licence request is still under review.",
        "expected": "licensing",
    },
    {
        "test_id": "MFT-014",
        "lang": "ar",
        "text": "طلب الترخيص ما زال قيد المراجعة.",
        "expected": "licensing",
    },
    {
        "test_id": "MFT-015",
        "lang": "en",
        "text": "The children's playground in the park needs maintenance.",
        "expected": "parks",
    },
    {
        "test_id": "MFT-016",
        "lang": "ar",
        "text": "ألعاب الأطفال في الحديقة تحتاج إلى صيانة.",
        "expected": "parks",
    },
]


TERM_GROUPS = [
    ["Riyadh", "Jeddah", "Dammam"],
    ["الرياض", "جدة", "الدمام"],
    ["today", "tomorrow"],
    ["اليوم", "غداً"],
]


def _replacement_term(term, terms=None):
    """Choose a same-category alternative for invariance testing."""
    for group in TERM_GROUPS:
        if term in group:
            index = group.index(term)
            return group[(index + 1) % len(group)]
    return term


def run_behavioural_suite(
    predict_fn,
    templates_path=DEFAULT_TEMPLATES,
    mft_cases=None,
):
    """Run invariance, directional, and minimum-functionality tests.

    Parameters
    ----------
    predict_fn : callable
        Function accepting a list of strings and returning predicted
        topic labels in the same order.
    templates_path : str or pathlib.Path
        CSV containing the supplied behavioural test skeletons.
    mft_cases : list of dict, optional
        Custom minimum-functionality cases. Defaults to the Bayan MFT set.

    Returns
    -------
    pandas.DataFrame
        One row per behavioural test with pass/fail and applicability.
    """
    templates = pd.read_csv(templates_path)
    results = []

    terms = templates["term"].dropna().astype(str).unique().tolist()

    # Invariance:
    # replacing a location/time-like term should not change the predicted topic.
    invariance = templates[templates["test_type"] == "invariance"].copy()

    if not invariance.empty:
        original_texts = []
        perturbed_texts = []

        for _, row in invariance.iterrows():
            term = str(row["term"])
            replacement = _replacement_term(term, terms)

            original_texts.append(
                str(row["template"]).format(term=term)
            )
            perturbed_texts.append(
                str(row["template"]).format(term=replacement)
            )

        original_preds = list(predict_fn(original_texts))
        perturbed_preds = list(predict_fn(perturbed_texts))

        for (_, row), text, changed, pred_a, pred_b in zip(
            invariance.iterrows(),
            original_texts,
            perturbed_texts,
            original_preds,
            perturbed_preds,
        ):
            results.append(
                {
                    "test_id": row["test_id"],
                    "test_type": "invariance",
                    "lang": row["lang"],
                    "text": text,
                    "perturbed_text": changed,
                    "expected": "topic unchanged",
                    "prediction": pred_a,
                    "perturbed_prediction": pred_b,
                    "applicable": True,
                    "passed": bool(pred_a == pred_b),
                    "notes": "",
                }
            )

    # Directional:
    # supplied skeletons require sentiment behaviour, but Bayan's Lab 3A
    # artefact predicts topics only. Record this limitation explicitly.
    directional = templates[templates["test_type"] == "directional"]

    for _, row in directional.iterrows():
        text = str(row["template"]).format(term=str(row["term"]))

        results.append(
            {
                "test_id": row["test_id"],
                "test_type": "directional",
                "lang": row["lang"],
                "text": text,
                "perturbed_text": "",
                "expected": row["expected_relation"],
                "prediction": "",
                "perturbed_prediction": "",
                "applicable": False,
                "passed": pd.NA,
                "notes": (
                    "Not applicable: the Bayan Lab 3A artefact is a "
                    "topic classifier and has no sentiment output."
                ),
            }
        )

    # Minimum-functionality tests:
    # simple bilingual examples that should map to their intended topic.
    cases = MFT_CASES if mft_cases is None else mft_cases

    mft_texts = [case["text"] for case in cases]
    mft_preds = list(predict_fn(mft_texts))

    for case, prediction in zip(cases, mft_preds):
        results.append(
            {
                "test_id": case["test_id"],
                "test_type": "MFT",
                "lang": case["lang"],
                "text": case["text"],
                "perturbed_text": "",
                "expected": case["expected"],
                "prediction": prediction,
                "perturbed_prediction": "",
                "applicable": True,
                "passed": bool(prediction == case["expected"]),
                "notes": "",
            }
        )

    return pd.DataFrame(results)
