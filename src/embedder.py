import os
import torch
import clip
from PIL import Image
from tqdm import tqdm

from src.config import IMAGE_FOLDER, DATA_DIR
from src.image_db import load_image_db, save_image_db

PREPROCESSED_DIR = os.path.join(DATA_DIR, "preprocessed")
os.makedirs(PREPROCESSED_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

def preprocess_and_cache(image_id, filename):
    cache_path = os.path.join(PREPROCESSED_DIR, f"{image_id}.pt")
    if os.path.exists(cache_path):
        return torch.load(cache_path)

    path = os.path.join(IMAGE_FOLDER, filename)
    try:
        img_tensor = preprocess(Image.open(path))
        torch.save(img_tensor, cache_path)
        return img_tensor
    except Exception as e:
        print(f"Failed to preprocess {filename}: {e}")
        return None

def embed_all_images(batch_size=32):
    db = load_image_db()
    embeddings = []
    ids = []

    image_ids, tensors = [], []

    for image_id, entry in tqdm(db.items(), desc="Loading Preprocessed Tensors"):
        if entry["embedding"] is not None:
            continue

        tensor = preprocess_and_cache(image_id, entry["filename"])
        if tensor is not None:
            tensors.append(tensor)
            image_ids.append(image_id)

    for i in tqdm(range(0, len(tensors), batch_size), desc="Embedding"):
        batch = torch.stack(tensors[i:i+batch_size]).to(device)
        with torch.no_grad():
            vecs = model.encode_image(batch)
            vecs /= vecs.norm(dim=-1, keepdim=True)

        vecs_np = vecs.cpu().numpy()

        for j, vec in enumerate(vecs_np):
            image_id = image_ids[i + j]
            db[image_id]["embedding"] = vec.tolist()
            embeddings.append(vec)
            ids.append(image_id)

    save_image_db(db)
    print(f"Embedded {len(embeddings)} images using cached tensors.")
    return embeddings, ids
