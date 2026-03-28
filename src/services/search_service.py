from __future__ import annotations

from typing import List, Dict, Any

from src.models.clip_model import encode_text
from src.storage.faiss_store import search_index, get_record_by_faiss_index


def l2_distance_to_cosine_similarity(distance: float) -> float:
    """
    Convert squared L2 distance between normalized vectors into cosine similarity.

    For normalized vectors:
        squared_l2 = 2 - 2 * cosine_similarity
        cosine_similarity = 1 - squared_l2 / 2
    """
    similarity = 1.0 - (distance / 2.0)

    if similarity > 1.0:
        similarity = 1.0
    elif similarity < -1.0:
        similarity = -1.0

    return similarity


def search_by_text(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    if not isinstance(query, str):
        raise TypeError("query must be a string")

    query = query.strip()
    if not query:
        raise ValueError("query cannot be empty")

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    query_vector = encode_text(query)
    distances, indices = search_index(query_vector, top_k=top_k)

    results = []

    for rank, (distance, faiss_index) in enumerate(zip(distances, indices), start=1):
        faiss_index = int(faiss_index)

        if faiss_index < 0:
            continue

        record = get_record_by_faiss_index(faiss_index)
        similarity_score = l2_distance_to_cosine_similarity(float(distance))

        results.append(
            {
                "rank": rank,
                "image_id": record["image_id"],
                "filename": record["filename"],
                "path": record["path"],
                "raw_distance": float(distance),
                "similarity_score": similarity_score,
            }
        )

    return results