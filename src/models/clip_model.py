from __future__ import annotations

import clip
import torch

import numpy as np
from functools import lru_cache
from pathlib import Path
from PIL import Image


MODEL_NAME = 'ViT-B/32'


@lru_cache(maxsize=1)
def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def get_clip_model():
    device = get_device()
    model, preprocess = clip.load(MODEL_NAME, device=device)
    model.eval()
    return model, preprocess


def _normalize_embedding(embedding: torch.Tensor) -> torch.Tensor:
    return embedding / embedding.norm(dim=-1, keepdim=True)


def encode_text(query: str) -> np.ndarray:
    model, _ = get_clip_model()
    device = get_device()
    
    tokens = clip.tokenize([query]).to(device)
    
    with torch.no_grad():
        embedding = model.encode_text(tokens)
        embedding = _normalize_embedding(embedding)

    return embedding.squeeze(0).cpu().numpy().astype(np.float32)


def encode_image(image_path: str | Path) -> np.ndarray:
    model, preprocess = get_clip_model()
    device = get_device()
    
    image_path = Path(image_path)
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        embedding = model.encode_image(image_tensor)
        embedding = _normalize_embedding(embedding)

    return embedding.squeeze(0).cpu().numpy().astype(np.float32)