"""Lab 7: Bayan FastAPI inference service."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer


MODEL_DIR = Path("artifacts/topic_classifier")
ONNX_MODEL = Path("artifacts/onnx_classifier/classifier_int8.onnx")

app = FastAPI(title="Bayan — Bilingual Citizen-Feedback Intelligence Service")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

session = ort.InferenceSession(
    str(ONNX_MODEL),
    providers=["CPUExecutionProvider"],
)

id2label = {
    int(k): v
    for k, v in tokenizer.init_kwargs.get("id2label", {}).items()
} if tokenizer.init_kwargs.get("id2label") else None


class ClassifyRequest(BaseModel):
    text: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "ONNX INT8",
    }


@app.post("/v1/classify")
def classify(payload: ClassifyRequest):
    text = payload.text.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    inputs = tokenizer(
        text,
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

    prediction = int(np.argmax(logits, axis=-1)[0])

    label = id2label.get(prediction, str(prediction)) if id2label else prediction

    return {
        "label": label,
        "label_id": prediction,
    }
