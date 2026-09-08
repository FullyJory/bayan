"""Lab 3B: run the 12-question QA smoke set."""

import json

import torch
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
)

from bayan.models.qa import best_span


MODEL_NAME = "deepset/xlm-roberta-base-squad2"
SMOKE_PATH = "data/eval/qa_smoke_set.json"


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")
    print(f"Loading QA model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    model = AutoModelForQuestionAnswering.from_pretrained(
        MODEL_NAME
    ).to(device)

    model.eval()

    with open(SMOKE_PATH, encoding="utf-8") as f:
        smoke = json.load(f)

    answerable_correct = 0
    answerable_total = 0
    null_correct = 0
    null_total = 0

    for item in smoke["data"]:
        for paragraph in item["paragraphs"]:
            context = paragraph["context"]

            for qa in paragraph["qas"]:
                question = qa["question"]

                encoded = tokenizer(
                    question,
                    context,
                    return_tensors="pt",
                    return_offsets_mapping=True,
                    truncation=True,
                    max_length=384,
                )

                offsets = encoded.pop("offset_mapping")[0].tolist()
                sequence_ids = encoded.sequence_ids(0)

                context_offsets = []

                for offset, sequence_id in zip(offsets, sequence_ids):
                    if sequence_id == 1:
                        context_offsets.append(tuple(offset))
                    else:
                        context_offsets.append(None)

                encoded = {
                    key: value.to(device)
                    for key, value in encoded.items()
                }

                with torch.no_grad():
                    outputs = model(**encoded)

                start_logits = (
                    outputs.start_logits[0]
                    .detach()
                    .cpu()
                    .numpy()
                )

                end_logits = (
                    outputs.end_logits[0]
                    .detach()
                    .cpu()
                    .numpy()
                )

                null_score = float(
                    start_logits[0] + end_logits[0]
                )

                result = best_span(
                    start_logits,
                    end_logits,
                    context_offsets,
                    null_score=null_score,
                    null_threshold=1.0,
                    max_answer_len=30,
                    top_k=20,
                )

                predicted = result.get("answer")

                if qa["is_impossible"]:
                    null_total += 1
                    correct = predicted is None

                    if correct:
                        null_correct += 1

                    print(
                        f"{qa['id']} | expected=None | "
                        f"predicted={predicted} | "
                        f"{'PASS' if correct else 'FAIL'}"
                    )

                else:
                    answerable_total += 1

                    gold = qa["answers"][0]

                    gold_span = (
                        gold["answer_start"],
                        gold["answer_start"] + len(gold["text"]),
                    )

                    predicted_text = None

                    if predicted is not None:
                        predicted_text = context[
                            predicted[0]:predicted[1]
                        ]

                    def normalize_answer(text):
                        if text is None:
                            return None

                        text = text.strip()

                        if text.lower().startswith("the "):
                            text = text[4:]

                        return text.strip()

                    correct = (
                        normalize_answer(predicted_text)
                        == normalize_answer(gold["text"])
                    )

                    if correct:
                        answerable_correct += 1

                    print(
                        f"{qa['id']} | "
                        f"expected={gold['text']!r} | "
                        f"predicted={predicted_text!r} | "
                        f"{'PASS' if correct else 'FAIL'}"
                    )

    print("\n=== QA Smoke Results ===")
    print(f"Answerable: {answerable_correct}/{answerable_total}")
    print(f"Unanswerable: {null_correct}/{null_total}")

    if (
        answerable_correct == answerable_total
        and null_correct == null_total
    ):
        print("QA smoke test PASSED")
    else:
        print("QA smoke test FAILED")


if __name__ == "__main__":
    main()
