import pandas as pd
import os
import glob

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# ---- CONFIG ----
INPUT_FOLDER = "Demo"
EVAL_FILE = "Evaluation Summary.csv"
MISMATCH_FOLDER = "Evaluation"
os.makedirs(MISMATCH_FOLDER, exist_ok=True)


def normalize_code(val):
    """Convert codes to string without trailing .0 or NaN"""
    if pd.isna(val):
        return None
    try:
        return str(int(float(val)))
    except ValueError:
        return str(val)


def process_file(file_path):
    filename = os.path.splitext(os.path.basename(file_path))[0].replace("final_matches_", "")

    df = pd.read_csv(file_path)
    df = df[["t_NEW_CODES", "matched_itemcode", "rank"]]

    # Normalize both columns
    df["t_NEW_CODES"] = df["t_NEW_CODES"].apply(normalize_code)
    df["matched_itemcode"] = df["matched_itemcode"].apply(normalize_code)

    # Keep only valid numeric codes
    valid_mask = df["t_NEW_CODES"].fillna("").str.isnumeric()
    valid_df = df[valid_mask].copy()
    total_df = df.copy()

    if valid_df.empty:
        print(f"|WARNING| {file_path} has no valid rows, skipping...")
        return

    # Generate suggestion ranks (1..n per t_NEW_CODES)
    valid_df["suggestion_rank"] = valid_df.groupby("t_NEW_CODES").cumcount() + 1

    pivoted = valid_df.pivot_table(
        index="t_NEW_CODES",
        columns="suggestion_rank",
        values="matched_itemcode",
        aggfunc="first"
    )
    pivoted.columns = [f"rank_{c}" for c in pivoted.columns]
    pivoted = pivoted.reset_index()

    total = len(pivoted)

    # ---- Debug output ----
    # print("\n[DEBUG] Dtypes for", filename)
    # print(pivoted.dtypes)

    # print("\n[DEBUG] Sample pivoted rows for", filename)
    # print(pivoted.head(10))

    # ---- Accuracy calculations ----
    def accuracy_at_k(row, k):
        tcode = row["t_NEW_CODES"]
        ranks = [row.get(f"rank_{i}") for i in range(1, k + 1)]
        # Debug per-row
        # print(f"[DEBUG] tcode={tcode}, ranks={ranks}")
        return any(r is not None and r == tcode for r in ranks)

    top1 = pivoted.apply(lambda r: accuracy_at_k(r, 1), axis=1).mean()
    top2 = pivoted.apply(lambda r: accuracy_at_k(r, 2), axis=1).mean()
    top3 = pivoted.apply(lambda r: accuracy_at_k(r, 3), axis=1).mean()

    print(f"\n[DEBUG] Accuracies for {filename}: Top1={top1}, Top2={top2}, Top3={top3}")

    # ---- Save eval summary ----
    row = pd.DataFrame([[filename, round(top1, 4), round(top2, 4), round(top3, 4)]],
                       columns=["filename", "Top1", "Top2", "Top3"])
    if os.path.exists(EVAL_FILE):
        row.to_csv(EVAL_FILE, mode="a", header=False, index=False)
    else:
        row.to_csv(EVAL_FILE, index=False)

    # ---- Save mismatched rows ----
    def flag_group(group):
        tcode = group["t_NEW_CODES"].iloc[0]
        matches = group.loc[group["matched_itemcode"] == tcode, "rank"]
        if matches.empty:
            flag_val = 4
        else:
            flag_val = matches.min()
        group = group.copy()
        group["FLAG"] = flag_val
        return group

    mismatches = total_df.groupby("t_NEW_CODES", group_keys=False).apply(flag_group)
    mismatches = mismatches[mismatches["FLAG"] != 1]

    mismatch_file = os.path.join(MISMATCH_FOLDER, f"eval_{filename}.csv")
    mismatches.to_csv(mismatch_file, index=False)

    print(f"|INFO| Processed {file_path}")
    print(f"       Results appended to {EVAL_FILE}")
    print(f"       Mismatched rows saved in {mismatch_file}")


# ---- MAIN ----
if __name__ == "__main__":
    csv_files = glob.glob(os.path.join(INPUT_FOLDER, "final_matches_*.csv"))
    if not csv_files:
        print(f"|WARNING| No files found in {INPUT_FOLDER}")
    else:
        print(f"|INFO| Found {len(csv_files)} files to process")
        for file in csv_files:
            process_file(file)