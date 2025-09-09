import pandas as pd
import os
import glob
import argparse
from datetime import datetime

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message="DataFrameGroupBy.apply")


# ---------------- ARGUMENT PARSER ----------------
parser = argparse.ArgumentParser(description="Format transactions and optionally evaluate results.")
parser.add_argument("--evaluate", action="store_true", help="Enable evaluation and save metrics to CSV")
args = parser.parse_args()
EVALUATE = args.evaluate

# ---------------- CONFIG ----------------
input_folder = r"FinalMatches"  # <-- replace with your folder path
output_folder = os.path.join("TransactionFormatting", "FormattedOutput")
metrics_file = os.path.join("TransactionFormatting", "Evaluation_metrics.csv")

# ---------------- TRANSACTION COLUMNS ----------------
txn_cols = [
    "t_DATE", "t_PERIOD", "t_AUDITYPE", "t_STORECODE", "t_DLRCODE", "t_ITEMCODE",
    "t_NEW_CODES", "t_CATEGORY", "t_MANUFACTURE", "t_BRAND", "t_ITEMDESC", "t_MRP",
    "t_PACKSIZE", "t_PACKTYPE", "t_COMMENTS", "t_IMAGE", "t_CODE COMMENT", "t_FLAG"
]

# ---------------- PROCESSING ----------------
all_outputs = []  # store (filename, DataFrame) for evaluation
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

    # Save in FormattedOutput folder with same filename
    output_file = os.path.join(output_folder, os.path.basename(file))
    formatted_df.to_csv(output_file, index=False)

    # If evaluation is enabled, make a clean copy *before grouping*
    if EVALUATE:
        total_before = len(df)
        eval_df = df.dropna(subset=["t_NEW_CODES", "t_CATEGORY"]).copy()
        eval_df = eval_df[(eval_df["t_NEW_CODES"].astype(str).str.strip() != "") &
                          (eval_df["t_CATEGORY"].astype(str).str.strip() != "")]
        total_after = len(eval_df)
        dropped = total_before - total_after
        print(f"✅ {os.path.basename(file)}: using {total_after} valid rows (dropped {dropped})")

        # Build grouped dataframe for eval_df only
        eval_formatted = (
            eval_df.groupby(present_cols, dropna=False)
                   .apply(collect_suggestions)
                   .reset_index()
                   .sort_values("rank")
                   .reset_index(drop=True)
        )
        all_outputs.append((os.path.basename(file), eval_formatted))
    else:
        all_outputs.append((os.path.basename(file), formatted_df))


print(f"✅ All files processed. Results saved in: {output_folder}")

# ---------------- EVALUATION ----------------
def evaluate_accuracy(outputs):
    records = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for filename, df in outputs:
        total_rows = 0
        top1_hits = 0
        top2_hits = 0
        top3_hits = 0

        for _, row in df.iterrows():
            true_code = str(row["t_NEW_CODES"]).strip()
            preds = [str(row.get(str(i), "")) for i in ["1", "2", "3"]]

            total_rows += 1
            if true_code == preds[0]:
                top1_hits += 1
            if true_code in preds[:2]:
                top2_hits += 1
            if true_code in preds[:3]:
                top3_hits += 1

        if total_rows == 0:
            continue

        # Compute accuracies
        top1_acc = top1_hits / total_rows
        top2_acc = top2_hits / total_rows
        top3_acc = top3_hits / total_rows

        # Extra metrics (for Top-3)
        TP = top3_hits
        FN = total_rows - top3_hits
        FP = total_rows * 3 - TP  # each row predicts 3 codes, only 1 can be true
        TN = 0  # in multi-class ranking, true negatives not well-defined

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        type1_error = FP / (FP + TN) if (FP + TN) > 0 else 0
        type2_error = FN / (FN + TP) if (FN + TP) > 0 else 0

        records.append({
            "filename": filename,
            "timestamp": timestamp,
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
        print("⚠️ No valid rows for evaluation.")
        return

    # Append to CSV (create if doesn't exist)
    df_metrics = pd.DataFrame(records)
    if os.path.exists(metrics_file):
        df_metrics.to_csv(metrics_file, mode="a", header=False, index=False)
    else:
        df_metrics.to_csv(metrics_file, index=False)

    print(f"\n📊 Evaluation completed. Metrics saved to {metrics_file}")


if EVALUATE:
    evaluate_accuracy(all_outputs)