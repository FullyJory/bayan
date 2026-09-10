"""Lab 7: export Bayan classifier to ONNX and dynamic INT8."""

from pathlib import Path
import shutil

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from onnxruntime.quantization import quantize_dynamic, QuantType


MODEL_DIR = Path("artifacts/topic_classifier")
OUT_DIR = Path("artifacts/onnx_classifier")

FP32_ONNX = OUT_DIR / "classifier_fp32.onnx"
INT8_ONNX = OUT_DIR / "classifier_int8.onnx"
ROLLBACK_DIR = OUT_DIR / "rollback_fp32"


class ClassifierWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading classifier...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    model.to("cpu")

    print("Saving FP32 rollback artifact...")
    if ROLLBACK_DIR.exists():
        shutil.rmtree(ROLLBACK_DIR)
    shutil.copytree(MODEL_DIR, ROLLBACK_DIR)

    sample = tokenizer(
        "الخدمة ممتازة",
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )

    wrapper = ClassifierWrapper(model)

    print("Exporting ONNX FP32...")
    torch.onnx.export(
        wrapper,
        (sample["input_ids"], sample["attention_mask"]),
        FP32_ONNX,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True, dynamo=False,
    )

    print("Quantizing ONNX model to INT8...")
    quantize_dynamic(
        model_input=str(FP32_ONNX),
        model_output=str(INT8_ONNX),
        weight_type=QuantType.QInt8,
    )

    print("\nLab 7 export complete")
    print("FP32 ONNX:", FP32_ONNX)
    print("INT8 ONNX:", INT8_ONNX)
    print("Rollback:", ROLLBACK_DIR)


if __name__ == "__main__":
    main()
