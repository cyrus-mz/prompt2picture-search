from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from src.config import DB_FILE, INDEX_FILE


def create_faiss_index(dimension: int):
    if dimension <= 0:
        raise ValueError(f'dimension must be greater than 0, got {dimension}')
    return faiss.IndexFlatL2(dimension)


def save_faiss_index(index) -> None:
    index_path = Path(INDEX_FILE)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))


def load_faiss_index():
    index_path = Path(INDEX_FILE)

    if not index_path.exists():
        raise FileNotFoundError(
            f'FAISS index not found at {index_path}. Please build it first.'
        )

    return faiss.read_index(str(index_path))


def load_or_create_faiss_index(dimension: int):
    index_path = Path(INDEX_FILE)
    if index_path.exists():
        return faiss.read_index(str(index_path))
    return create_faiss_index(dimension)


def add_embeddings(index, embeddings: np.ndarray) -> None:
    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.ndim != 2:
        raise ValueError(
            f'Expected embeddings to have shape (batch_size, dim), got {embeddings.shape}'
        )

    index.add(embeddings)


def search_index(query_vector: np.ndarray, top_k: int = 5):
    index = load_faiss_index()

    if query_vector.ndim != 1:
        raise ValueError(
            f'Expected query_vector to have shape (dim,), got {query_vector.shape}'
        )

    query_vector = query_vector.reshape(1, -1).astype(np.float32)
    distances, indices = index.search(query_vector, top_k)

    return distances[0], indices[0]


def load_index_records() -> list[dict]:
    db_path = Path(DB_FILE)

    if not db_path.exists():
        raise FileNotFoundError(
            f'Image database not found at {db_path}. Please build it first.'
        )

    with db_path.open('r', encoding='utf-8') as f:
        db = json.load(f)

    records = []

    for image_id, entry in db.items():
        if entry.get('embedding') is None:
            continue

        source_path = entry.get('source_path')
        if not source_path:
            continue

        records.append(
            {
                'image_id': image_id,
                'filename': entry['filename'],
                'path': source_path,
                'source_path': source_path,
            }
        )

    return records


def get_record_by_faiss_index(faiss_index: int) -> dict:
    records = load_index_records()

    if faiss_index < 0 or faiss_index >= len(records):
        raise IndexError(
            f'FAISS result index {faiss_index} is out of range for {len(records)} records.'
        )

    return records[faiss_index]


def delete_faiss_index() -> None:
    index_path = Path(INDEX_FILE)
    if index_path.exists():
        index_path.unlink()