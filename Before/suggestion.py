import pandas as pd
import requests
import re
from difflib import SequenceMatcher

# Load match result and master
matches = pd.read_csv("matches.csv").fillna("")
master = pd.read_csv("/home/dcsadmin/Documents/del_SKU/StockKeepingUnit/Data/DataCleaned/master_cleaned.csv").fillna("")

# Only focus on unmatched rows
unmatched = matches[matches['MATCHED'] == 1].copy()

def extract_product_name(text):
    prompt = (
        f"Extract only the product name from the following description. "
        f"Do not include quantity, unit, or pack type:\n\n{text}\n\nProduct name:"
    )
    try:
        response = requests.post("http://localhost:11434/api/generate", json={"model": "mistral", "prompt": prompt, "stream": False})
        if response.status_code == 200:
            return re.sub(r"\(.*?\)", "", response.json()["response"]).strip()
        return "Error"
    except Exception as e:
        return "Error"

def similar(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def get_best_fallback_suggestion(row, master_df):
    manu = row['MANUFACTURE']
    brand = row['BRAND']
    itemdesc = row['ITEMDESC']
    
    best_score = 0
    best_row = None
    
    for _, m_row in master_df.iterrows():
        manu_sim = similar(manu, m_row['company'])
        brand_sim = similar(brand, m_row['brand'])
        item_sim = similar(itemdesc, m_row['itemdesc'])
        
        score = 0.4 * manu_sim + 0.2 * brand_sim + 0.4 * item_sim
        
        if score > best_score:
            best_score = score
            best_row = m_row
            
    if best_row is not None and best_score >= 0.70:
        return best_row["nitemcode"]
    return ""

# Apply fallback suggestion
matches['Suggestion'] = ''
for idx, row in unmatched.iterrows():
    suggestion = get_best_fallback_suggestion(row, master)
    matches.loc[idx, 'Suggestion'] = suggestion

matches.to_csv("matches+suggestions.csv", index=False)
print("? Fallback suggestions updated for unmatched rows.")