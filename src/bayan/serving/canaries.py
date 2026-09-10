"""Lab 7: startup canaries for Bayan serving."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


MODEL_DIR = Path("artifacts/topic_classifier")
ONNX_MODEL = Path("artifacts/onnx_classifier/classifier_int8.onnx")


def run_startup_canaries() -> None:
    assert MODEL_DIR.exists(), "Classifier artifact missing"
    assert ONNX_MODEL.exists(), "INT8 ONNX artifact missing"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    session = ort.InferenceSession(
        str(ONNX_MODEL),
        providers=["CPUExecutionProvider"],
    )

    inputs = tokenizer(
        "الخدمة ممتازة",
        return_tensors="np",
        truncation=True,
        max_length=128,
    )

    logits = session.run(
        None,
        {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        },
    )[0]

    assert logits.shape[0] == 1, "Unexpected classifier output shape"
    assert np.isfinite(logits).all(), "Classifier produced invalid values"
