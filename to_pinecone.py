import torch
import pandas as pd
from transformers import AutoModel, AutoTokenizer
import numpy as np
from pinecone import Pinecone, ServerlessSpec

# Define device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Set up Pinecone environment
pinecone_api = "pcsk_4GT8aD_FPuF7yJHHbz2h8Tpn9GRrAXjTULo69KzemEafbBwyUawYMFz3hYXpSFkTtqkrdL"
pc = Pinecone(api_key=pinecone_api)  # Replace with your API key

index_name = "product-embeddings-test"

# Check if the index exists, if not create it
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,  # Ensure the dimension matches your model output (E5 model output size)
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
file_path = "/home/dcsadmin/Documents/DeleteSKU/master_data"  # Your file path
df = pd.read_csv(file_path, encoding='windows-1252')  # Or your CSV file
df = df.fillna('')  # Handle missing values. Very Important.
print("Data loaded.")

# Function to get embeddings from text using the model
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embeddings.cpu().numpy()

def store_embeddings_in_pinecone(chunk_size=100):
    vectors = []
    ids = []
    metadata_list = []  # List to store metadata dictionaries

    # Iterate through the rows and collect embeddings
    for index, row in df.iterrows():
        text = row['itemdesc']
        embedding = get_embedding(text)
        vectors.append(embedding)
        ids.append(f"desc_{row['itemcode']}")
        metadata_list.append(row.to_dict())  # Store metadata for each row

        # When we reach the chunk size, upload to Pinecone
        if len(vectors) >= chunk_size:
            vectors_data = [{"id": id, "values": vec.tolist(), "metadata": metadata} 
                            for id, vec, metadata in zip(ids, vectors, metadata_list)]
            vector_index.upsert(vectors=vectors_data, namespace="desc")
            print(f"Uploaded {len(vectors_data)} embeddings to Pinecone.")
            
            # Reset the lists to start a new chunk
            vectors = []
            ids = []
            metadata_list = []

    # Upload any remaining embeddings that didn't fill the last chunk
    if vectors:
        vectors_data = [{"id": id, "values": vec.tolist(), "metadata": metadata} 
                        for id, vec, metadata in zip(ids, vectors, metadata_list)]
        vector_index.upsert(vectors=vectors_data, namespace="desc")
        print(f"Uploaded {len(vectors_data)} embeddings to Pinecone.")

# Store embeddings in Pinecone in chunks
store_embeddings_in_pinecone(chunk_size=100)
