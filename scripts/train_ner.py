

"""Lab 3B: fine-tune token classification with correct alignment."""

import argparse
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict
from seqeval.metrics import f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from bayan.models.ner import align_labels


CHECKPOINT = "xlm-roberta-base"
DATA_PATH = "data/models/bayan_ner.conll"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="artifacts/ner",
        help="Where to save the trained NER artefact.",
    )
    return parser.parse_args()


def read_conll(path):
    sentences = []
    tokens = []
    labels = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            if not line.strip():
                if tokens:
                    sentences.append({
                        "tokens": tokens,
                        "ner_tags": labels,
                    })
                    tokens = []
                    labels = []
                continue

            token, label = line.rsplit(maxsplit=1)
            tokens.append(token)
            labels.append(label)

    if tokens:
        sentences.append({
            "tokens": tokens,
            "ner_tags": labels,
        })

    return sentences


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = read_conll(DATA_PATH)

    labels = sorted({
        label
        for example in examples
        for label in example["ner_tags"]
    })

    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    train_examples, temp_examples = train_test_split(
        examples,
        test_size=0.20,
        random_state=42,
    )

    valid_examples, test_examples = train_test_split(
        temp_examples,
        test_size=0.50,
        random_state=42,
    )

    dataset = DatasetDict({
        "train": Dataset.from_list(train_examples),
        "validation": Dataset.from_list(valid_examples),
        "test": Dataset.from_list(test_examples),
    })

    tokenizer = AutoTokenizer.from_pretrained(
        CHECKPOINT,
        use_fast=True,
    )

    def tokenize_and_align(example):
        encoded = tokenizer(
            example["tokens"],
            is_split_into_words=True,
            truncation=True,
        )

        word_ids = encoded.word_ids()

        numeric_labels = [
            label2id[label]
            for label in example["ner_tags"]
        ]

        encoded["labels"] = align_labels(
            word_ids,
            numeric_labels,
        )

        return encoded

    tokenized = dataset.map(
        tokenize_and_align,
        remove_columns=dataset["train"].column_names,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        CHECKPOINT,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer
    )

    def compute_metrics(eval_pred):
        logits, true_labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        true_predictions = []
        true_references = []

        for pred_row, label_row in zip(predictions, true_labels):
            pred_labels = []
            ref_labels = []

            for pred, label in zip(pred_row, label_row):
                if label == -100:
                    continue

                pred_labels.append(id2label[int(pred)])
                ref_labels.append(id2label[int(label)])

            true_predictions.append(pred_labels)
            true_references.append(ref_labels)

        return {
            "f1": f1_score(
                true_references,
                true_predictions,
            )
        }

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=25,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    results = trainer.evaluate(tokenized["test"])

    print("\n=== NER Test Results ===")
    print(f"Entity-level F1: {results['eval_f1']:.4f}")

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print(f"\nSaved model to: {output_dir}")




if __name__ == "__main__":
    main()
