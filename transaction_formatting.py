import pandas as pd
import os
import glob

# Input folder containing CSVs
input_folder = r"FinalMatches"   # <-- replace with your folder path
output_folder = os.path.join(input_folder, "FormattedOutput")

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Full list of transaction columns (some may be missing in certain CSVs)
txn_cols = [
    "t_DATE","t_PERIOD","t_AUDITYPE","t_STORECODE","t_DLRCODE","t_ITEMCODE",
    "t_NEW_CODES","t_CATEGORY","t_MANUFACTURE","t_BRAND","t_ITEMDESC","t_MRP",
    "t_PACKSIZE","t_PACKTYPE","t_COMMENTS","t_IMAGE","t_CODE COMMENT","t_FLAG"
]

# Process each CSV file in the input folder
for file in glob.glob(os.path.join(input_folder, "*.csv")):
    print(f"Processing: {os.path.basename(file)}")
    
    df = pd.read_csv(file)

    # Determine which transaction columns are actually present in this file
    present_cols = [col for col in txn_cols if col in df.columns]

    # Add helper index to preserve original order
    df["_order"] = df.index

    # Group and aggregate
    formatted_df = (
        df.groupby(present_cols, dropna=False)
          .agg({
              "_order": "min",
              "rank": "min",   # keep first rank; change if you want joined ranks
              "matched_itemcode": lambda x: " | ".join(x.astype(str).unique())
          })
          .reset_index()
    )

    # Restore original order
    formatted_df = formatted_df.sort_values("_order").drop(columns="_order")

    # Save in FormattedOutput folder with same filename
    output_file = os.path.join(output_folder, os.path.basename(file))
    formatted_df.to_csv(output_file, index=False)

print(f"✅ All files processed. Results saved in: {output_folder}")