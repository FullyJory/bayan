"""Lab 6: sliced evaluation report."""

import pandas as pd


def sliced_report(df, *, min_n=20):
    """Return accuracy sliced by language, dialect, class, and length.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain: lang, dialect_region, length_bucket, y_true, y_pred.
    min_n : int
        Slices smaller than this are flagged as small.

    Returns
    -------
    pandas.DataFrame
        Columns: slice_type, slice_value, n, accuracy, small_slice.
    """
    required = {
        "lang",
        "dialect_region",
        "length_bucket",
        "y_true",
        "y_pred",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    rows = []

    slice_specs = {
        "language": "lang",
        "dialect": "dialect_region",
        "class": "y_true",
        "length": "length_bucket",
    }

    for slice_type, column in slice_specs.items():
        grouped = df.groupby(column, dropna=False)

        for value, group in grouped:
            n = len(group)
            accuracy = float((group["y_true"] == group["y_pred"]).mean())

            if pd.isna(value):
                value = "missing"

            rows.append(
                {
                    "slice_type": slice_type,
                    "slice_value": str(value),
                    "n": int(n),
                    "accuracy": accuracy,
                    "small_slice": bool(n < min_n),
                }
            )

    return pd.DataFrame(rows)
