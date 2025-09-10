import pandas as pd
import os
import glob
from datetime import datetime

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message="DataFrameGroupBy.apply")

# ---------------- CONFIG ----------------
input_folder = r"FinalMatches"  # <-- replace with your folder path
output_folder = os.path.join("TransactionFormatting", "FormattedOutput")
metrics_file = os.path.join("TransactionFormatting", "Evaluation_metrics.csv")

# ---------------- TRANSACTION COLUMNS ----------------
txn_cols = [
    "t_row_id",  # ✅ added here so it is carried forward
    "t_DATE", "t_PERIOD", "t_AUDITYPE", "t_STORECODE", "t_DLRCODE", "t_ITEMCODE",
    "t_NEW_CODES", "t_CATEGORY", "t_MANUFACTURE", "t_BRAND", "t_ITEMDESC", "t_MRP",
    "t_PACKSIZE", "t_PACKTYPE", "t_COMMENTS", "t_IMAGE", "t_CODE COMMENT", "t_FLAG"
]

# ---------------- PROCESSING ----------------
os.makedirs(output_folder, exist_ok=True)

for file in glob.glob(os.path.join(input_folder, "*.csv")):
    print(f"Processing: {os.path.basename(file)}")
    df = pd.read_csv(file)

    # Check required columns
    if "matched_itemcode" not in df.columns or "rank" not in df.columns:
        print(f"⚠️ Skipping {file} (required columns missing)")
        continue

    # Determine which transaction columns are actually present in this file
    present_cols = [col for col in txn_cols if col in df.columns]

    # Add helper index to preserve original order
    df["_order"] = df.index

    # Sort by rank so suggestions stay in correct order
    df = df.sort_values(["_order", "rank"])

    # Group and collect top 3 suggestions
    def collect_suggestions(x):
        codes = x["matched_itemcode"].astype(str).tolist()
        return pd.Series({
            "1": codes[0] if len(codes) > 0 else None,
            "2": codes[1] if len(codes) > 1 else None,
            "3": codes[2] if len(codes) > 2 else None,
            "rank": x["rank"].min()
        })

    formatted_df = (
        df.groupby(present_cols, dropna=False)
          .apply(collect_suggestions)
          .reset_index()
    )

    # Restore original order
    formatted_df = formatted_df.sort_values("rank").reset_index(drop=True)
    
    # ✅ Force clean up of codes (avoid floats like 99763433.0)
    for col in ["t_NEW_CODES", "1", "2", "3"]:
        if col in formatted_df.columns:
            formatted_df[col] = (
                formatted_df[col]
                .astype(str)
                .str.strip()
                .str.replace(r"\.0$", "", regex=True)  # remove trailing .0
            )

    # Save in FormattedOutput folder with same filename
    output_file = os.path.join(output_folder, os.path.basename(file))
    formatted_df.to_csv(output_file, index=False)

print(f"✅ All files processed. Results saved in: {output_folder}")


# ---------------- EVALUATION FUNCTION ----------------
def normalize_code(val):
    """Convert codes to clean comparable strings."""
    if pd.isna(val):
        return ""
    val = str(val).strip()
    if val.endswith(".0"):  # remove trailing .0
        val = val[:-2]
    return val


def evaluate_folder(output_folder=output_folder, metrics_file=metrics_file):
    """Evaluate all CSV files in the formatted output folder."""
    records = []

    for file in glob.glob(os.path.join(output_folder, "*.csv")):
        df = pd.read_csv(file)

        # Drop invalid rows (t_NEW_CODES or t_CATEGORY blank/NaN)
        total_before = len(df)
        eval_df = df.dropna(subset=["t_NEW_CODES", "t_CATEGORY"]).copy()
        eval_df = eval_df[(eval_df["t_NEW_CODES"].astype(str).str.strip() != "") &
                          (eval_df["t_CATEGORY"].astype(str).str.strip() != "")]
        total_after = len(eval_df)
        dropped = total_before - total_after

        print(f"✅ {os.path.basename(file)}: using {total_after} valid rows (dropped {dropped})")

        if total_after == 0:
            print(f"⚠️ Skipping {os.path.basename(file)} (no valid rows).")
            continue

        # ✅ Deduplicate by t_ITEMCODE (keep first occurrence only)
        if "t_ITEMCODE" in eval_df.columns:
            eval_df = eval_df.drop_duplicates(subset=["t_ITEMCODE"], keep="first")
            print(f"   ↪️ Reduced to {len(eval_df)} unique ITEMCODE rows for evaluation")

        total_rows = 0
        top1_hits = 0
        top2_hits = 0
        top3_hits = 0

        for _, row in eval_df.iterrows():
            true_code = normalize_code(row["t_NEW_CODES"])
            preds = [normalize_code(row.get(str(i), "")) for i in ["1", "2", "3"]]

            if not true_code:
                continue

            total_rows += 1
            if true_code == preds[0]:
                top1_hits += 1
            if true_code in preds[:2]:
                top2_hits += 1
            if true_code in preds[:3]:
                top3_hits += 1

        # Compute accuracies
        top1_acc = top1_hits / total_rows
        top2_acc = top2_hits / total_rows
        top3_acc = top3_hits / total_rows

        # Extra metrics (for Top-3)
        TP = top3_hits
        FN = total_rows - top3_hits
        FP = total_rows * 3 - TP
        TN = 0

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        type1_error = FP / (FP + TN) if (FP + TN) > 0 else 0
        type2_error = FN / (FN + TP) if (FN + TP) > 0 else 0

        records.append({
            "filename": os.path.basename(file),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Top1": round(top1_acc, 4),
            "Top1+Top2": round(top2_acc, 4),
            "Top1+Top2+Top3": round(top3_acc, 4),
            "Precision (Top-3)": round(precision, 4),
            "Recall (Top-3)": round(recall, 4),
            "Specificity (Top-3)": round(specificity, 4),
            "Type I Error": round(type1_error, 4),
            "Type II Error": round(type2_error, 4)
        })

    if not records:
        print("⚠️ No evaluation metrics recorded.")
        return

    df_metrics = pd.DataFrame(records)

    # Append to CSV (create if doesn't exist)
    if os.path.exists(metrics_file):
        df_metrics.to_csv(metrics_file, mode="a", header=False, index=False)
    else:
        df_metrics.to_csv(metrics_file, index=False)

    print(f"\n📊 Evaluation completed. Metrics saved to {metrics_file}")

evaluate_folder()