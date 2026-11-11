import pandas as pd
from pathlib import Path


RAW_FILE = Path("data/raw/phishing_email.csv")
OUTPUT_FILE = Path("data/emails.csv")


def load_dataset():
    """Load the main phishing email dataset."""
    print("Loading dataset...")

    df = pd.read_csv(RAW_FILE)

    print(f"Original rows: {len(df):,}")
    print(f"Original columns: {df.columns.tolist()}")

    return df


def clean_dataset(df):
    """Clean and prepare the dataset for machine learning."""

    # Keep only the columns we need
    df = df[["text_combined", "label"]].copy()

    # Rename text column
    df.rename(
        columns={
            "text_combined": "body"
        },
        inplace=True
    )

    # Convert body to string
    df["body"] = df["body"].fillna("").astype(str)

    # Convert labels to integers
    df["label"] = pd.to_numeric(df["label"], errors="coerce")

    # Remove invalid labels
    df = df[df["label"].isin([0, 1])]

    # Remove empty emails
    df = df[df["body"].str.strip().str.len() > 0]

    # Remove duplicate emails
    before_duplicates = len(df)
    df = df.drop_duplicates(subset=["body"])
    duplicates_removed = before_duplicates - len(df)

    # Add an empty subject column.
    # The source dataset combines subject and body into one text field.
    df.insert(0, "subject", "")

    # Reorder columns
    df = df[["subject", "body", "label"]]

    print(f"Duplicates removed: {duplicates_removed:,}")
    print(f"Final rows: {len(df):,}")

    return df


def show_statistics(df):
    """Display dataset statistics."""

    print("\n" + "=" * 50)
    print("DATASET STATISTICS")
    print("=" * 50)

    print(f"Total emails: {len(df):,}")

    safe_count = (df["label"] == 0).sum()
    phishing_count = (df["label"] == 1).sum()

    print(f"Safe emails: {safe_count:,}")
    print(f"Phishing emails: {phishing_count:,}")

    print("\nClass distribution:")
    print(df["label"].value_counts())

    print("\nClass percentages:")
    print(
        (df["label"].value_counts(normalize=True) * 100)
        .round(2)
    )

    print("\nMissing values:")
    print(df.isnull().sum())


def save_dataset(df):
    """Save cleaned dataset."""

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    print(f"\nClean dataset saved to: {OUTPUT_FILE}")


def main():

    print("=" * 60)
    print("PHISHING EMAIL DATASET PREPARATION")
    print("=" * 60)

    if not RAW_FILE.exists():
        print(f"\nERROR: Dataset not found:")
        print(RAW_FILE)
        return

    df = load_dataset()

    df = clean_dataset(df)

    show_statistics(df)

    save_dataset(df)

    print("\nDataset preparation completed successfully!")


if __name__ == "__main__":
    main()