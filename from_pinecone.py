import torch
import pandas as pd
from transformers import AutoModel, AutoTokenizer
import numpy as np
from pinecone import Pinecone

# Define device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Set up Pinecone environment
pinecone_api = "pcsk_4GT8aD_FPuF7yJHHbz2h8Tpn9GRrAXjTULo69KzemEafbBwyUawYMFz3hYXpSFkTtqkrdL"
pc = Pinecone(api_key=pinecone_api)  # Replace with your API key

index_name = "product-embeddings-test"

# Connect to Pinecone index
vector_index = pc.Index(index_name)

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

# Function to retrieve data from Pinecone based on query
def retrieve_from_pinecone(query, top_k=5):
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

            df_results = pd.DataFrame(data)  # Create DataFrame directly from the list of dictionaries
            return df_results
        else:
            return pd.DataFrame()  # Return empty DataFrame if no matches

    result = process_results(query_results)

    print("\nTop Matches for itemdesc:")
    if not result.empty:  # Check if the DataFrame is empty before displaying
        print(result[['itemdesc', 'itemcode', 'category', 'company', 'brand', 'score']].head())
    else:
        print("No matches found.")

    return result

# Example query
# query = "WRISHAV"
query = input("Enter Your Query: ")
# print("\n")
# print(f"Query: {query}")
retrieve_from_pinecone(query)
print("\n")