import os
import numpy as np
import pandas as pd
import ollama
import faiss
import json
from tqdm import tqdm

# ----------------- CONFIG -----------------
MODEL_NAME = "nomic-embed-text"
MASTER_FILE = 'Data/DataCSV/master.csv'

# Ensure required directories exist
os.makedirs("temp", exist_ok=True)

# Output files
FAISS_INDEX_FILE = "./temp/master_index.faiss"
METADATA_FILE = "./temp/metadata.json"

# Master columns
m_columns = ['itemcode', 'catcode', 'company', 'mbrand', 'brand', 'sku',
             'packtype', 'base_pack', 'flavor', 'color', 'wght', 'uom', 'mrp']

# ----------------- LOAD MASTER -----------------
master = pd.read_csv(MASTER_FILE)
master = master[m_columns]

# ----------------- FUNCTION: EMBEDDING -----------------
def get_embedding(text):
    """Call local Ollama client to get embedding for a given text."""
    try:
        response = ollama.embeddings(model=MODEL_NAME, prompt=text)
        return np.array(response["embedding"], dtype=np.float32)
    except Exception as e:
        print(f"|ERROR| Error embedding text: {e}")
        return None

# ----------------- PREPARE MASTER EMBEDDINGS -----------------
print(f"|INFO| Generating embeddings for {len(master)} master rows...")
texts, itemcodes, metadata = [], [], {}

for _, row in tqdm(master.iterrows(), total=len(master), desc="Master Embeddings"):
    itemcode = str(row["itemcode"])
    itemcodes.append(itemcode)

    # Join all fields except itemcode
    row_text = " ".join(str(val) for col, val in row.items() if col != "itemcode" and pd.notna(val))
    texts.append(row_text)

    # Store metadata
    metadata[itemcode] = row.drop("itemcode").to_dict()

# Generate embeddings
embeddings = []
for text in tqdm(texts, desc="Fetching Embeddings"):
    emb = get_embedding(text)
    if emb is not None:
        embeddings.append(emb)
    else:
        # embeddings.append(np.zeros(1536, dtype=np.float32))  # fallback, adjust dim if model differs
        fallback_dim = embeddings[0].shape[0] if embeddings else 1
        embeddings.append(np.zeros(fallback_dim, dtype=np.float32))

embedding_dim = embeddings[0].shape[0]
embeddings_np = np.vstack([emb if emb.shape[0] == embedding_dim else np.zeros(embedding_dim, dtype=np.float32) 
                           for emb in embeddings]).astype('float32')
    

# # Convert embeddings to numpy array
# embeddings_np = np.vstack(embeddings).astype('float32')

# ----------------- BUILD FAISS CPU INDEX -----------------
embedding_dim = embeddings_np.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(embeddings_np)

# Save index and metadata
faiss.write_index(index, FAISS_INDEX_FILE)
with open(METADATA_FILE, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"|INFO| Saved {len(embeddings)} embeddings to {FAISS_INDEX_FILE} and metadata to {METADATA_FILE}")