import pandas as pd
from difflib import get_close_matches

# Load master file
master = pd.read_csv("master_cleaned.csv")

# --- INPUT SECTION ---
input_manufacturer = "ARTINAT CORDIALS INDUSTRIES"
input_brand = "CAMEL ASTO"
input_qty = "1500"
input_packtype = "PET"
input_itemdesc = "CAMEL ASTO-PET 1500 ML"
# ----------------------

# Step 1: Filter by MANUFACTURE
A = master[master['company'] == input_manufacturer]
if A.empty:
    print("MANUFACTURER NOT FOUND")
else:
    # Step 2: Filter by BRAND
    B = A[A['brand'] == input_brand]
    if B.empty:
        print("BRAND NOT AVAILABLE")
    else:
        # Step 3: Filter by QUANTITY
        input_qty = int(input_qty)              # Convert input_qty to integer for comparison   
        C = B[B['qty'] == input_qty]
        if C.empty:
            print("QUANTITY NOT AVAILABLE")
        else:
            # Step 4: Filter by PACKTYPE
            D = C[C['packaging'] == input_packtype]
            if D.empty:
                print("PACKTYPE NOT AVAILABLE")
            else:
                # Step 5: Fuzzy matching on ITEMDESC (case-insensitive)
                itemdesc_list = D['itemdesc'].dropna().astype(str).tolist()
                close_matches = get_close_matches(input_itemdesc, itemdesc_list, n=5, cutoff=0.5)

                if not close_matches:
                    print("NO MATCH")
                else:
                    matches = D[D['itemdesc'].isin(close_matches)]
                    matches.to_csv("matches.csv", index=False)
                    print("Match found and saved to matches.csv")
