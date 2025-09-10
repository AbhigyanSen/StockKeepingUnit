import os
import pandas as pd
import numpy as np

def detect_dtype(series: pd.Series) -> str:
    """Detect simplified datatype for a pandas series."""
    if pd.api.types.is_integer_dtype(series):
        return "int"
    elif pd.api.types.is_float_dtype(series):
        return "float"
    elif pd.api.types.is_bool_dtype(series):
        return "bool"
    elif pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    else:
        return "string"

def summarize_csv_folder(input_folder, output_csv):
    summary = {}
    all_columns = set()

    # Loop through all CSV files in the folder
    for file in os.listdir(input_folder):
        if file.endswith(".csv"):
            file_path = os.path.join(input_folder, file)
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                print(f"Skipping {file} due to error: {e}")
                continue

            col_types = {}
            for col in df.columns:
                col_types[col] = detect_dtype(df[col])
                all_columns.add(col)

            summary[file] = col_types

    # Create final DataFrame
    all_columns = sorted(all_columns)  # Keep columns ordered
    result = pd.DataFrame(columns=["file"] + all_columns)

    for file, col_types in summary.items():
        row = {"file": file}
        for col in all_columns:
            row[col] = col_types.get(col, "")  # blank if column not present
        result = pd.concat([result, pd.DataFrame([row])], ignore_index=True)

    # Save to CSV
    result.to_csv(output_csv, index=False)
    print(f"Summary saved to {output_csv}")

# Example usage:
summarize_csv_folder("TransactionFormatting\\FormattedOutput", "output_datatype.csv")