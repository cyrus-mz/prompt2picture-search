from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Sequence

import clip
import numpy as np
import torch
from PIL import Image


MODEL_NAME = 'ViT-B/32'


@lru_cache(maxsize=1)
def get_device() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@lru_cache(maxsize=1)
def get_clip_model():
    device = get_device()
    model, preprocess = clip.load(MODEL_NAME, device=device)
    model.eval()
    return model, preprocess


def _normalize_embedding(embedding: torch.Tensor) -> torch.Tensor:
    return embedding / embedding.norm(dim=-1, keepdim=True)


def encode_text(query: str) -> np.ndarray:
    embeddings = encode_texts_batch([query])
    return embeddings[0]

 
def encode_texts_batch(queries: Sequence[str]) -> np.ndarray:
    if not queries:
        raise ValueError('queries cannot be empty')

    cleaned_queries = []
    for query in queries:
        if not isinstance(query, str):
            raise TypeError('each query must be a string')
        query = query.strip()
        if not query:
            raise ValueError('queries cannot contain empty strings')
        cleaned_queries.append(query)

    model, _ = get_clip_model()
    device = get_device()

    tokens = clip.tokenize(cleaned_queries).to(device)

    with torch.no_grad():
        embeddings = model.encode_text(tokens)
        embeddings = _normalize_embedding(embeddings)

    return embeddings.cpu().numpy().astype(np.float32)


def encode_image(image_path: str | Path) -> np.ndarray:
    embeddings = encode_images_batch([image_path])
    return embeddings[0]


def encode_images_batch(image_paths: Sequence[str | Path]) -> np.ndarray:
    if not image_paths:
        raise ValueError('image_paths cannot be empty')

    model, preprocess = get_clip_model()
    device = get_device()

    image_tensors = []

    for image_path in image_paths:
        image_path = Path(image_path)
        image = Image.open(image_path).convert('RGB')
        image_tensor = preprocess(image)
        image_tensors.append(image_tensor)

    batch_tensor = torch.stack(image_tensors, dim=0).to(device)

    with torch.no_grad():
        embeddings = model.encode_image(batch_tensor)
        embeddings = _normalize_embedding(embeddings)

    return embeddings.cpu().numpy().astype(np.float32)