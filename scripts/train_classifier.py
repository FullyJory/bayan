"""Lab 3A: fine-tune the Bayan topic classifier."""

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from bayan.models.data import build_topic_dataset


CHECKPOINT = "xlm-roberta-base"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="artifacts/topic_classifier",
        help="Where to save the trained classifier artefact.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_topic_dataset()

    labels = sorted(set(dataset["train"]["topic"]))

    label2id = {
        label: i
        for i, label in enumerate(labels)
    }

    id2label = {
        i: label
        for label, i in label2id.items()
    }

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

    def preprocess(batch):
        tokens = tokenizer(
            batch["text"],
            truncation=True,
            max_length=128,
        )

        tokens["labels"] = [
            label2id[label]
            for label in batch["topic"]
        ]

        return tokens

    tokenized = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )

    def compute_metrics(eval_pred):
        logits, true_labels = eval_pred

        predictions = np.argmax(
            logits,
            axis=-1,
        )

        macro_f1 = f1_score(
            true_labels,
            predictions,
            average="macro",
        )

        return {
            "macro_f1": macro_f1
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
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
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

    test_results = trainer.evaluate(
        tokenized["test"]
    )

    print("\n=== Transformer Test Results ===")
    print(
        f"Macro-F1: "
        f"{test_results['eval_macro_f1']:.4f}"
    )

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print(f"\nSaved model to: {output_dir}")


if __name__ == "__main__":
    main()
