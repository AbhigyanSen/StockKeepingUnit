import pandas as pd
from difflib import get_close_matches, SequenceMatcher

# Load master file
master = pd.read_csv("master_cleaned.csv")

# Load transaction file
transactions = pd.read_csv("transaction_cleaned.csv")           # <-- Update with your actual file name

# Prepare output list
results = []

# Iterate over each transaction row
for idx, row in transactions.iterrows():
    input_manufacturer = str(row['MANUFACTURE']).strip()
    input_brand = str(row['BRAND']).strip()
    input_itemdesc = str(row['ITEMDESC']).strip()
    input_qty = int(row['QTY'])
    input_unit = str(row['UNIT']).strip()
    input_packsize = str(row['PACKSIZE']).strip()
    input_packtype = str(row['PACKTYPE']).strip()

    # Step 1: Filter by MANUFACTURER
    A = master[master['company'] == input_manufacturer]
    if A.empty:
        match_flag = 1
        result = [input_manufacturer, input_brand, input_itemdesc, input_qty, input_unit, input_packsize, input_packtype,
                  match_flag, '', '', '', '', '', '', '', '']
        results.append(result)
        continue

    # Step 2: Filter by BRAND
    B = A[A['brand'] == input_brand]
    if B.empty:
        match_flag = 1
        result = [input_manufacturer, input_brand, input_itemdesc, input_qty, input_unit, input_packsize, input_packtype,
                  match_flag, '', '', '', '', '', '', '', '']
        results.append(result)
        continue

    # Step 3: Filter by QTY
    C = B[B['qty'] == input_qty]
    if C.empty:
        match_flag = 1
        result = [input_manufacturer, input_brand, input_itemdesc, input_qty, input_unit, input_packsize, input_packtype,
                  match_flag, '', '', '', '', '', '', '', '']
        results.append(result)
        continue

    # Step 4: Filter by PACKTYPE
    D = C[C['packaging'] == input_packtype]
    if D.empty:
        match_flag = 1
        result = [input_manufacturer, input_brand, input_itemdesc, input_qty, input_unit, input_packsize, input_packtype,
                  match_flag, '', '', '', '', '', '', '', '']
        results.append(result)
        continue

    # Step 5: Fuzzy match on ITEMDESC
    itemdesc_list = D['itemdesc'].dropna().astype(str).tolist()
    close_matches = get_close_matches(input_itemdesc, itemdesc_list, n=1, cutoff=0.5)

    if not close_matches:
        match_flag = 1
        result = [input_manufacturer, input_brand, input_itemdesc, input_qty, input_unit, input_packsize, input_packtype,
                  match_flag, '', '', '', '', '', '', '', '']
    else:
        best_match = close_matches[0]
        match_row = D[D['itemdesc'] == best_match].iloc[0]
        fuzzy_score = round(SequenceMatcher(None, input_itemdesc, best_match).ratio(), 2)

        match_flag = 0
        result = [input_manufacturer, input_brand, input_itemdesc, input_qty, input_unit, input_packsize, input_packtype,
                  match_flag,
                  match_row['itemdesc'], match_row['company'], match_row['brand'], match_row['packaging'],
                  match_row['qty'], match_row['uomdesc'], match_row['pack_size'], fuzzy_score]

    results.append(result)

# Save results to matches.csv
columns = ['MANUFACTURE', 'BRAND', 'ITEMDESC', 'QTY', 'UNIT', 'PACKSIZE', 'PACKTYPE',
           'MATCHED', 'MATCH_ITEMDESC', 'MATCH_COMPANY', 'MATCH_BRAND', 'MATCH_PACKAGING',
           'MATCH_QTY', 'MATCH_UNIT', 'MATCH_PACKSIZE', 'FUZZY_SCORE']

results_df = pd.DataFrame(results, columns=columns)
results_df.to_csv("matches.csv", index=False)

print("Matching complete. Results saved to matches.csv.")