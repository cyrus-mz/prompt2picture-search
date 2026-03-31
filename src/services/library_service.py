from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import numpy as np

from src.image_db import load_image_db, save_image_db
from src.models.clip_model import encode_image, encode_images_batch
from src.storage.faiss_store import (
    add_embeddings,
    create_faiss_index,
    delete_faiss_index,
    save_faiss_index,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_source_path(raw_path: str | Path) -> Path:
    return Path(raw_path).expanduser().resolve()


def _classify_record_path(record: dict) -> Tuple[str, dict]:
    image_id = record.get("image_id")
    filename = record.get("filename", "")
    source_path = record.get("source_path")

    if not source_path:
        return "invalid", {
            "image_id": image_id,
            "filename": filename,
            "source_path": None,
            "reason": "Missing source_path in database record",
        }

    normalized_path = _normalize_source_path(source_path)
    item = {
        "image_id": image_id,
        "filename": normalized_path.name if normalized_path.name else filename,
        "source_path": str(normalized_path),
    }

    if not normalized_path.exists():
        return "missing", {
            **item,
            "reason": "File does not exist",
        }

    if not normalized_path.is_file():
        return "invalid", {
            **item,
            "reason": "Path exists but is not a file",
        }

    return "available", item


def _encode_batch_with_fallback(batch_paths: list[Path]):
    try:
        embeddings = encode_images_batch(batch_paths)
        return embeddings, []
    except Exception:
        embeddings = []
        failures = []

        for image_path in batch_paths:
            try:
                embedding = encode_image(image_path)
                embeddings.append(embedding)
            except Exception as image_error:
                failures.append(
                    {
                        "path": str(image_path),
                        "reason": f"Embedding failed: {image_error}",
                    }
                )

        if embeddings:
            return np.vstack(embeddings).astype(np.float32), failures

        return np.empty((0, 512), dtype=np.float32), failures


def scan_indexed_library() -> dict:
    db = load_image_db()

    report = {
        "available": [],
        "missing": [],
        "invalid": [],
    }

    for image_id, record in db.items():
        record_with_id = {"image_id": image_id, **record}
        status, item = _classify_record_path(record_with_id)
        report[status].append(item)

    return report


def rebuild_index_from_db(batch_size: int = 32) -> dict:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    db = load_image_db()
    db_items = list(db.items())

    new_db = {}
    index = None

    report = {
        "reindexed": [],
        "missing": [],
        "failed": [],
    }

    for start in range(0, len(db_items), batch_size):
        batch_items = db_items[start:start + batch_size]

        valid_items = []

        for image_id, record in batch_items:
            record_with_id = {"image_id": image_id, **record}
            status, item = _classify_record_path(record_with_id)

            updated_record = dict(record)
            updated_record["filename"] = item.get("filename", updated_record.get("filename", ""))
            updated_record["source_path"] = item.get("source_path")
            updated_record["embedding"] = None

            if status == "available":
                valid_items.append(
                    {
                        "image_id": image_id,
                        "record": updated_record,
                        "source_path": Path(item["source_path"]),
                    }
                )
                continue

            updated_record["last_scan_status"] = status
            new_db[image_id] = updated_record

            if status == "missing":
                report["missing"].append(item)
            else:
                report["failed"].append(item)

        if not valid_items:
            continue

        batch_paths = [item["source_path"] for item in valid_items]
        embeddings, failures = _encode_batch_with_fallback(batch_paths)
        failed_paths = {failure["path"] for failure in failures}

        for failure in failures:
            matching_item = next(
                (item for item in valid_items if str(item["source_path"]) == failure["path"]),
                None,
            )

            if matching_item is None:
                continue

            image_id = matching_item["image_id"]
            failed_record = dict(matching_item["record"])
            failed_record["embedding"] = None
            failed_record["last_scan_status"] = "embedding_failed"
            new_db[image_id] = failed_record

            report["failed"].append(
                {
                    "image_id": image_id,
                    "filename": failed_record["filename"],
                    "source_path": failed_record["source_path"],
                    "reason": failure["reason"],
                }
            )

        successful_items = [
            item for item in valid_items
            if str(item["source_path"]) not in failed_paths
        ]

        if not successful_items:
            continue

        if index is None:
            index = create_faiss_index(embeddings.shape[1])

        add_embeddings(index, embeddings)

        for item, embedding in zip(successful_items, embeddings):
            image_id = item["image_id"]
            updated_record = dict(item["record"])
            updated_record["embedding"] = embedding.tolist()
            updated_record["last_scan_status"] = "available"
            updated_record["last_indexed_at"] = _utc_now_iso()

            new_db[image_id] = updated_record

            report["reindexed"].append(
                {
                    "image_id": image_id,
                    "filename": updated_record["filename"],
                    "source_path": updated_record["source_path"],
                    "last_indexed_at": updated_record["last_indexed_at"],
                }
            )

    save_image_db(new_db)

    if index is None:
        delete_faiss_index()
    else:
        save_faiss_index(index)

    return report