"""Lab 4: audit dialect mix over the Arabic slice."""

import pandas as pd

DATA_PATH = "data/raw/bayan_feedback.csv"


def main():
    df = pd.read_csv(DATA_PATH)

    arabic = df[df["lang"].eq("ar")].copy()

    counts = arabic["dialect_region"].value_counts(dropna=False)
    percentages = arabic["dialect_region"].value_counts(
        normalize=True, dropna=False
    ) * 100

    print("=== Arabic Dialect Audit ===")
    print(f"Total Arabic rows: {len(arabic)}")
    print()

    for region, count in counts.items():
        print(f"{region}: {count} ({percentages.loc[region]:.1f}%)")


if __name__ == "__main__":
    main()
