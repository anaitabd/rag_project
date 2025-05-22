import faiss
import os
import numpy as np
import pickle

VECTOR_STORE_PATH = "ragbot/vector_store/faiss.index"
METADATA_PATH = "ragbot/vector_store/chunks.pkl"

index = None
texts = []

def save_index():
    faiss.write_index(index, VECTOR_STORE_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(texts, f)

def load_index():
    global index, texts
    if os.path.exists(VECTOR_STORE_PATH):
        index = faiss.read_index(VECTOR_STORE_PATH)
        with open(METADATA_PATH, "rb") as f:
            texts = pickle.load(f)

def build_index(embeddings, content_chunks):
    global index, texts
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    texts = content_chunks
    save_index()

def query_index(query_embedding, k=3):
    D, I = index.search(np.array([query_embedding]), k)
    return [texts[i] for i in I[0]]