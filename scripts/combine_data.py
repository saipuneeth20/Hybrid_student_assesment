import pandas as pd

# =========================
# Input / Output paths
# =========================

INPUT_FILES = [
    "data/train.csv",
    "data/train_ms.csv",
    "data/train_si.csv"
]

OUTPUT_FILE = "data/train_combined.csv"


def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # ---- Required columns ----
    required_cols = {"question", "score"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {missing}")

    # ---- Handle answer column ----
    if "answer" not in df.columns:
        # Entire file has no answers → treat as empty answers
        df["answer"] = ""

    # Replace NaN answers with empty string
    df["answer"] = df["answer"].fillna("").astype(str)

    # ---- Handle context column (optional) ----
    if "context" not in df.columns:
        df["context"] = ""
    else:
        df["context"] = df["context"].fillna("").astype(str)

    # ---- Ensure score is numeric (keep 0, drop invalid) ----
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"])

    return df


def main():
    dfs = []

    for path in INPUT_FILES:
        print(f"Loading {path}")
        df = load_and_clean(path)
        print(f"  -> {len(df)} valid rows")
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    print("\nFinal combined dataset summary:")
    print(combined_df["score"].describe())
    print(f"Total rows: {len(combined_df)}")

    combined_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved combined dataset to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
