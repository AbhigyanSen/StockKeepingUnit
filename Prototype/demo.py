import torch
import pandas as pd
from transformers import AutoModel, AutoTokenizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load Data
file_path = "/home/dcsadmin/Documents/DeleteSKU/data.csv"
df = pd.read_csv(file_path, encoding='windows-1252')

# Define device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Initialize tokenizer and model
model_id = "intfloat/e5-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id).to(device)
model.eval()

# Function to get embeddings from text using the model
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embeddings.cpu().numpy()

# Function to get embeddings from text using the model
### Continuing from def get_embedding

# Generate embeddings for both strategies: itemdesc only, and itemdesc + other fields
embeddings_desc = []
embeddings_full = []

for index, row in df.iterrows():
    # Match with itemdesc only
    text_desc = row['itemdesc']
    embedding_desc = get_embedding(text_desc)
    embeddings_desc.append(embedding_desc)

    # Match with itemdesc, itemcode, category, company, and brand
    text_full = f"{row['itemdesc']} {row['itemcode']} {row['category']} {row['company']} {row['brand']}"
    embedding_full = get_embedding(text_full)
    embeddings_full.append(embedding_full)

# Convert lists to numpy arrays
embeddings_desc = np.array(embeddings_desc)
embeddings_full = np.array(embeddings_full)

# Function to perform retrieval and output results with similarity scores
def retrieve(query, top_k=5):
    # Query matching with itemdesc only
    query_embedding_desc = get_embedding(query).reshape(1, -1)
    similarities_desc = cosine_similarity(query_embedding_desc, embeddings_desc)
    top_k_indices_desc = similarities_desc.argsort()[0][-top_k:][::-1]
    top_k_similarities_desc = similarities_desc[0][top_k_indices_desc]  # Get the top k similarities

    # Query matching with itemdesc + other fields
    query_embedding_full = get_embedding(query).reshape(1, -1)
    similarities_full = cosine_similarity(query_embedding_full, embeddings_full)
    top_k_indices_full = similarities_full.argsort()[0][-top_k:][::-1]
    top_k_similarities_full = similarities_full[0][top_k_indices_full]  # Get the top k similarities

    # Extract the results for both strategies
    result_desc = df.iloc[top_k_indices_desc][['itemdesc', 'itemcode', 'category', 'company', 'brand']]
    result_full = df.iloc[top_k_indices_full][['itemdesc', 'itemcode', 'category', 'company', 'brand']]

    # Add similarity scores to both results
    result_desc['score'] = top_k_similarities_desc
    result_full['score'] = top_k_similarities_full

    # Display results in the desired format
    print("\nTop Matches for itemdesc:")
    print(result_desc)

    print("\nTop Matches for itemdesc + itemcode + category + company + brand:")
    print(result_full)

    return result_desc, result_full

# Example query
query = "WRISHAV"
print(f"Query: {query}")
retrieve(query)
print("\n")


# Example query
query = "WRISHAV"
print(f"Query: {query}")
retrieve(query)
print("\n")