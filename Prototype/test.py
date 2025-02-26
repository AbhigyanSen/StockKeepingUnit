import torch
import pandas as pd
from transformers import AutoModel, AutoTokenizer
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from IPython.display import display

# Define device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Set up Pinecone environment
pc = Pinecone(api_key="pcsk_4GT8aD_FPuF7yJHHbz2h8Tpn9GRrAXjTULo69KzemEafbBwyUawYMFz3hYXpSFkTtqkrdL")  # Replace with your API key

index_name = "product-embeddings-test"

# Check if the index exists, if not create it
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,  # Make sure the dimension matches your model output
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"  # Or your preferred region
        )
    )

# Connect to Pinecone index
vector_index = pc.Index(index_name)

# Initialize tokenizer and model
model_id = "intfloat/e5-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id).to(device)
model.eval()

# Load Data
file_path = "/home/dcsadmin/Documents/DeleteSKU/data.csv"
df = pd.read_csv(file_path, encoding='windows-1252')  # Or your CSV file
df = df.fillna('') # Handle missing values. Very Important.
df.head()

# Function to get embeddings from text using the model
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embeddings.cpu().numpy()

def store_embeddings_in_pinecone():
    vectors = []
    ids = []
    metadata_list = []  # List to store metadata dictionaries

    for index, row in df.iterrows():  # Correct way to iterate and access row
        text = row['itemdesc']
        embedding = get_embedding(text)
        vectors.append(embedding)
        ids.append(f"desc_{row['itemcode']}")
        metadata_list.append(row.to_dict())  # Store metadata for each row

    vectors_data = [{"id": id, "values": vec.tolist(), "metadata": metadata} for id, vec, metadata in zip(ids, vectors, metadata_list)]
    vector_index.upsert(vectors=vectors_data, namespace="desc")

# Store embeddings in Pinecone
store_embeddings_in_pinecone()

def retrieve(query, top_k=5):
    query_embedding = get_embedding(query).reshape(1, -1)

    query_results = vector_index.query(vector=query_embedding.tolist()[0], top_k=top_k, include_metadata=True, namespace="desc")

    def process_results(results):
        matches = results['matches']
        data = []  # List to store dictionaries for DataFrame creation

        if matches:  # Check if there are any matches
            for match in matches:
                metadata = match.get('metadata', {})  # Safely get metadata, handle missing
                data.append({
                    'id': match['id'],
                    'score': match['score'],
                    **metadata  # Unpack metadata into the dictionary
                })

            df_results = pd.DataFrame(data)  # Create DataFrame DIRECTLY from the list of dictionaries
            return df_results
        else:
            return pd.DataFrame()  # Return empty DataFrame if no matches

    result = process_results(query_results)

    print("\nTop Matches for itemdesc:")
    if not result.empty:  # Check if the DataFrame is empty before displaying
        display(result[['itemdesc', 'itemcode', 'category', 'company', 'brand', 'score']].head())
    else:
        print("No matches found.")

    return result

# Example query
query = "WRISHAV"
print(f"Query: {query}")
retrieve(query)
print("\n")