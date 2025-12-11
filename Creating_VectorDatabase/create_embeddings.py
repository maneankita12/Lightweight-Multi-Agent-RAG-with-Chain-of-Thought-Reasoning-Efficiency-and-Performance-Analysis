import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os
import gc

BASE_PATH = '/Users/ankitamane/Documents/NEU/SEM1/NLP'
PROCESSED_PATH = f'{BASE_PATH}/processed_data'
DB_PATH = f'{BASE_PATH}/final_rag_database'

os.makedirs(DB_PATH, exist_ok=True)


# Load documents 
with open(f'{PROCESSED_PATH}/all_documents.json', 'r') as f:
    documents = json.load(f)

total_docs = len(documents)

#Initialize Model

# Disable multiprocessing to avoid segfault
import torch
torch.set_num_threads(1)

model = SentenceTransformer('all-MiniLM-L6-v2')

# Force single-threaded
model.encode(["test"], show_progress_bar=False)


# Create embeddings in small chunks

CHUNK_SIZE = 10000  # Much smaller chunks
BATCH_SIZE = 32     # Smaller batch size

num_chunks = (total_docs + CHUNK_SIZE - 1) // CHUNK_SIZE

print(f"\nProcessing {total_docs:,} documents in {num_chunks} chunks")
print(f"Chunk size: {CHUNK_SIZE:,}")
print(f"Batch size: {BATCH_SIZE}")

# Save embeddings incrementally to disk
embedding_files = []

for chunk_idx in range(num_chunks):
    start_idx = chunk_idx * CHUNK_SIZE
    end_idx = min((chunk_idx + 1) * CHUNK_SIZE, total_docs)
    
    print(f"\nChunk {chunk_idx + 1}/{num_chunks}: Docs {start_idx:,}-{end_idx:,}")
    
    # Extract texts
    chunk_texts = [documents[i]['text'] for i in range(start_idx, end_idx)]
    
    # Encode
    chunk_embeddings = model.encode(
        chunk_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device='cpu'
    )
    
    # Save chunk to disk immediately
    chunk_file = f'{DB_PATH}/embeddings_chunk_{chunk_idx}.npy'
    np.save(chunk_file, chunk_embeddings)
    embedding_files.append(chunk_file)
    
    print(f" Saved chunk {chunk_idx + 1}: {chunk_embeddings.shape}")
    
    # Free memory aggressively
    del chunk_texts
    del chunk_embeddings
    gc.collect()

# Combine Chunks

all_embeddings = []

for chunk_file in tqdm(embedding_files, desc="Loading chunks"):
    chunk = np.load(chunk_file)
    all_embeddings.append(chunk)
    del chunk
    gc.collect()

embeddings = np.vstack(all_embeddings)

# Clean up chunk files
for chunk_file in embedding_files:
    os.remove(chunk_file)
    
del all_embeddings
gc.collect()

# Build FAISS Index

print("\nBuilding FAISS index...")

embedding_dim = embeddings.shape[1]
index = faiss.IndexFlatIP(embedding_dim)
index.add(embeddings.astype('float32'))

print(f"Index built: {index.ntotal:,} vectors")

# save 

print("\Saving final files...")

faiss.write_index(index, f'{DB_PATH}/faiss_index.bin')

with open(f'{DB_PATH}/documents.pkl', 'wb') as f:
    pickle.dump(documents, f)
print("Documents saved")

np.save(f'{DB_PATH}/embeddings.npy', embeddings)
print("Embeddings saved")

config = {
    'embedding_model': 'all-MiniLM-L6-v2',
    'embedding_dim': embedding_dim,
    'num_documents': total_docs
}

with open(f'{DB_PATH}/config.json', 'w') as f:
    json.dump(config, f, indent=2)
print(" Config")
print(f"Location: {DB_PATH}")