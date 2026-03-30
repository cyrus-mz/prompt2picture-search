from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from src.image_db import load_image_db, save_image_db
from src.models.clip_model import encode_image, encode_images_batch
from src.storage.faiss_store import (
    add_embeddings,
    load_or_create_faiss_index,
    save_faiss_index,
)


SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _compute_file_hash(file_path: Path) -> str:
    hasher = hashlib.sha256()

    with file_path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            hasher.update(chunk)

    return hasher.hexdigest()


def _validate_source_path(source_path: Path) -> None:
    if not source_path.exists():
        raise FileNotFoundError(f'Source image not found: {source_path}')

    if not source_path.is_file():
        raise ValueError(f'Source path is not a file: {source_path}')

    if source_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(
            f'Unsupported image extension "{source_path.suffix}" for file: {source_path}'
        )


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
                        'path': str(image_path),
                        'reason': f'Embedding failed: {image_error}',
                    }
                )

        if embeddings:
            return np.vstack(embeddings).astype(np.float32), failures

        return np.empty((0, 512), dtype=np.float32), failures


def ingest_images(image_paths: Sequence[str | Path], batch_size: int = 32) -> dict:
    if not image_paths:
        raise ValueError('image_paths cannot be empty')

    if batch_size <= 0:
        raise ValueError('batch_size must be greater than 0')

    db = load_image_db()
    existing_hashes = {
        entry.get('file_hash')
        for entry in db.values()
        if entry.get('file_hash')
    }

    report = {
        'ingested': [],
        'skipped': [],
        'failed': [],
    }

    pending_records = []

    for raw_path in image_paths:
        source_path = Path(raw_path).expanduser().resolve()

        try:
            _validate_source_path(source_path)
            file_hash = _compute_file_hash(source_path)

            if file_hash in existing_hashes:
                report['skipped'].append(
                    {
                        'source_path': str(source_path),
                        'reason': 'Duplicate file detected by SHA-256 hash',
                    }
                )
                continue

            image_id = uuid.uuid4().hex[:8]

            record = {
                'filename': source_path.name,
                'source_path': str(source_path),
                'caption': '',
                'tags': [],
                'embedding': None,
                'file_hash': file_hash,
                'ingested_at': _utc_now_iso(),
            }

            db[image_id] = record

            pending_records.append(
                {
                    'image_id': image_id,
                    'source_path': source_path,
                }
            )
            existing_hashes.add(file_hash)

        except Exception as e:
            report['failed'].append(
                {
                    'source_path': str(source_path),
                    'reason': str(e),
                }
            )

    if not pending_records:
        save_image_db(db)
        return report

    index = None

    for start in range(0, len(pending_records), batch_size):
        batch_items = pending_records[start:start + batch_size]
        batch_paths = [item['source_path'] for item in batch_items]

        embeddings, failures = _encode_batch_with_fallback(batch_paths)
        failed_paths = {failure['path'] for failure in failures}

        for failure in failures:
            matching_item = next(
                (item for item in batch_items if str(item['source_path']) == failure['path']),
                None,
            )

            if matching_item is not None:
                image_id = matching_item['image_id']
                db.pop(image_id, None)

                report['failed'].append(
                    {
                        'source_path': str(matching_item['source_path']),
                        'reason': failure['reason'],
                    }
                )

        successful_items = [
            item for item in batch_items
            if str(item['source_path']) not in failed_paths
        ]

        if not successful_items:
            continue

        if index is None:
            index = load_or_create_faiss_index(embeddings.shape[1])

        add_embeddings(index, embeddings)

        for item, embedding in zip(successful_items, embeddings):
            image_id = item['image_id']
            db[image_id]['embedding'] = embedding.tolist()

            report['ingested'].append(
                {
                    'image_id': image_id,
                    'filename': db[image_id]['filename'],
                    'source_path': db[image_id]['source_path'],
                    'ingested_at': db[image_id]['ingested_at'],
                }
            )

    save_image_db(db)

    if index is not None:
        save_faiss_index(index)

    return report