"""Lab 3A: dataset construction and split integrity."""

from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import GroupShuffleSplit


DATA = Path("data/raw/bayan_feedback.csv")


def build_topic_dataset(data_path=DATA, seed=42):
    """Build leakage-safe grouped train/validation/test splits."""

    df = pd.read_csv(data_path)

    # Split into 80% train and 20% temporary set
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=seed,
    )

    train_idx, temp_idx = next(
        splitter.split(
            df,
            groups=df["citizen_group_id"],
        )
    )

    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)

    # Split temporary set equally into validation and test
    splitter_2 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=seed,
    )

    valid_idx, test_idx = next(
        splitter_2.split(
            temp_df,
            groups=temp_df["citizen_group_id"],
        )
    )

    valid_df = temp_df.iloc[valid_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    return DatasetDict({
        "train": Dataset.from_pandas(
            train_df,
            preserve_index=False,
        ),
        "validation": Dataset.from_pandas(
            valid_df,
            preserve_index=False,
        ),
        "test": Dataset.from_pandas(
            test_df,
            preserve_index=False,
        ),
    })
    
