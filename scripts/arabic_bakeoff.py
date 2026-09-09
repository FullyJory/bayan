"""Lab 4: compare Arabic-centric checkpoints on all/Gulf/MSA slices."""

from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

DATA_PATH = "data/raw/bayan_feedback.csv"

MODELS = {
    "CAMeLBERT-mix": "CAMeL-Lab/bert-base-arabic-camelbert-mix",
    "CAMeLBERT-DA": "CAMeL-Lab/bert-base-arabic-camelbert-da",
}

SEED = 42


def make_splits():
    df = pd.read_csv(DATA_PATH)
    df = df[df["lang"].eq("ar")].copy()

    topics = sorted(df["topic"].unique())
    label2id = {label: i for i, label in enumerate(topics)}
    id2label = {i: label for label, i in label2id.items()}

    df["label"] = df["topic"].map(label2id)
    df["strata"] = df["topic"] + "__" + df["dialect_region"]

    train_df, eval_df = train_test_split(
        df,
        test_size=0.20,
        random_state=SEED,
        stratify=df["strata"],
    )

    return train_df.reset_index(drop=True), eval_df.reset_index(drop=True), label2id, id2label


def evaluate_slice(logits, labels):
    preds = np.argmax(logits, axis=-1)
    return f1_score(labels, preds, average="macro")


def run_model(name, checkpoint, train_df, eval_df, label2id, id2label):
    print(f"\n=== {name} ===")
    print(f"Checkpoint: {checkpoint}")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=128,
        )

    train_ds = Dataset.from_pandas(
        train_df[["text", "label"]],
        preserve_index=False,
    ).map(tokenize, batched=True)

    eval_ds = Dataset.from_pandas(
        eval_df[["text", "label"]],
        preserve_index=False,
    ).map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
    )

    args = TrainingArguments(
        output_dir=f"artifacts/arabic_bakeoff/{name}",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=100,
        report_to="none",
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    trainer.train()

    output = trainer.predict(eval_ds)
    logits = output.predictions
    labels = output.label_ids

    all_f1 = evaluate_slice(logits, labels)

    gulf_mask = eval_df["dialect_region"].eq("Gulf").to_numpy()
    msa_mask = eval_df["dialect_region"].eq("MSA").to_numpy()

    gulf_f1 = evaluate_slice(logits[gulf_mask], labels[gulf_mask])
    msa_f1 = evaluate_slice(logits[msa_mask], labels[msa_mask])

    fertility = np.mean([
        len(tokenizer.tokenize(text)) / max(len(text.split()), 1)
        for text in eval_df["text"]
    ])

    result = {
        "checkpoint": name,
        "macro_f1_all": all_f1,
        "gulf_f1": gulf_f1,
        "msa_f1": msa_f1,
        "ar_fertility": fertility,
    }

    print(f"Macro-F1 all: {all_f1:.4f}")
    print(f"Gulf Macro-F1: {gulf_f1:.4f}")
    print(f"MSA Macro-F1: {msa_f1:.4f}")
    print(f"Arabic fertility: {fertility:.4f}")

    del trainer
    del model

    return result


def main():
    train_df, eval_df, label2id, id2label = make_splits()

    print("=== Arabic Bake-off Split ===")
    print(f"Train: {len(train_df)}")
    print(f"Eval: {len(eval_df)}")
    print(eval_df["dialect_region"].value_counts())

    results = []

    for name, checkpoint in MODELS.items():
        results.append(
            run_model(
                name,
                checkpoint,
                train_df,
                eval_df,
                label2id,
                id2label,
            )
        )

    results_df = pd.DataFrame(results)

    print("\n=== Final Arabic Bake-off ===")
    print(results_df.to_string(index=False))

    Path("artifacts/arabic_bakeoff").mkdir(parents=True, exist_ok=True)
    results_df.to_csv(
        "artifacts/arabic_bakeoff/results.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
