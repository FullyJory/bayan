"""Lab 3A: TF-IDF + LinearSVC baseline."""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


DATA = Path("data/raw/bayan_feedback.csv")


def main():
    df = pd.read_csv(DATA)

    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=2,
                max_features=50000,
            ),
        ),
        (
            "classifier",
            LinearSVC(),
        ),
    ])

    model.fit(
        train_df["text"],
        train_df["topic"],
    )

    predictions = model.predict(test_df["text"])

    macro_f1 = f1_score(
        test_df["topic"],
        predictions,
        average="macro",
    )

    print("=== TF-IDF + LinearSVC Baseline ===")
    print(f"Macro-F1: {macro_f1:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            test_df["topic"],
            predictions,
            digits=4,
        )
    )


if __name__ == "__main__":
    main()
