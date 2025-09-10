import os
import pandas as pd

# Define the target datatypes
dtype_map = {
    "CATEGORY": "str",
    "FLAG": "int",
    "ITEMCODE": "str",
    "MRP": "str",
    "NEW_CODES": "str",
    "PERIOD": "int",
    "STORECODE": "str"
}

def clean_string_value(val):
    """Convert floats ending with .0 to int-like strings and strip spaces."""
    if pd.isna(val):
        return ""
    val_str = str(val).strip()
    if val_str.endswith(".0"):
        val_str = val_str[:-2]  # remove the ".0"
    return val_str

def enforce_dtypes_in_folder(folder_path):
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            try:
                df = pd.read_csv(file_path, dtype=str)  # read everything as str first
            except Exception as e:
                print(f"Skipping {file} due to error: {e}")
                continue

            for col, target_type in dtype_map.items():
                if col not in df.columns:
                    continue

                if target_type == "str":
                    df[col] = df[col].apply(clean_string_value)
                elif target_type == "int":
                    # Convert safely to int (empty string if not valid)
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

            # Save back to the same file
            df.to_csv(file_path, index=False)
            print(f"Processed {file}")


# Example usage:
enforce_dtypes_in_folder("Demo")