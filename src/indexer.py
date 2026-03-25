import numpy as np
import faiss
import os
import json
from src.config import INDEX_FILE, DB_FILE

def build_faiss_index(embeddings):
    vecs = np.vstack(embeddings).astype("float32")
    index = faiss.IndexFlatL2(vecs.shape[1])
    index.add(vecs)
    faiss.write_index(index, INDEX_FILE)
    print(f"FAISS index saved to {INDEX_FILE}")

def load_faiss_index():
    if not os.path.exists(INDEX_FILE):
        raise FileNotFoundError("FAISS index not found. Please build it first.")
    index = faiss.read_index(INDEX_FILE)
    print(f"FAISS index loaded from {INDEX_FILE}")
    return index

def get_image_id_mapping():
    with open(DB_FILE, "r") as f:
        db = json.load(f)
    id_to_filename = {
        idx: data["filename"]
        for idx, (img_id, data) in enumerate(db.items())
        if data.get("embedding") is not None
    }
    return id_to_filename
