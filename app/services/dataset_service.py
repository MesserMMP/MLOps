"""Business logic for dataset management.

Routes and gRPC handlers delegate all heavy lifting here so they stay thin.
"""

import io
import os
from typing import List

from fastapi import HTTPException, UploadFile

from app.datasets.registry import DatasetRegistry, DatasetMeta
from app.datasets.service import save_uploaded_dataset
from app.datasets.dvc_config import DATA_DIR, get_dvc_repo
from app.utils.logging_config import setup_logging

logger = setup_logging()


def list_datasets() -> List[DatasetMeta]:
    """Return all datasets currently known to the registry."""
    logger.info("Listing datasets")
    return DatasetRegistry.list_all()


def upload_dataset(file: UploadFile) -> DatasetMeta:
    """Persist an uploaded file and register it.  Returns dataset metadata."""
    logger.info("Uploading dataset: %s", file.filename)
    dataset_name = save_uploaded_dataset(file)
    meta = DatasetRegistry.get(dataset_name)
    if meta is None:
        raise HTTPException(status_code=500, detail="Dataset was not registered after upload")
    return meta


def upload_dataset_from_bytes(filename: str, content: bytes) -> DatasetMeta:
    """Save raw bytes as a dataset file and register it.

    This variant is used by the gRPC server which receives the file contents
    as a byte payload rather than a multipart form upload.
    """
    logger.info("Uploading dataset from bytes: %s (%d bytes)", filename, len(content))

    safe_name = filename.replace(" ", "_")
    target_path = DATA_DIR / safe_name

    with target_path.open("wb") as f:
        f.write(content)

    ext = os.path.splitext(filename)[1].lower().lstrip(".")
    fmt = ext if ext in {"csv", "json"} else "csv"

    DatasetRegistry.register(
        name=safe_name,
        path=str(target_path),
        description=f"uploaded {fmt} file via gRPC (DVC-tracked)",
    )

    try:
        repo = get_dvc_repo()
        rel_path = os.path.relpath(target_path, repo.root_dir)
        logger.info("DVC add %s", rel_path)
        repo.add(rel_path)
        logger.info("DVC push")
        repo.push()
    except Exception as exc:
        logger.exception("Failed to run DVC for dataset %s: %s", safe_name, exc)

    meta = DatasetRegistry.get(safe_name)
    if meta is None:
        raise RuntimeError("Dataset was not registered after upload")
    return meta


def delete_dataset(name: str) -> bool:
    """Remove a dataset from the registry.

    Raises 404 if the dataset is not found.
    """
    logger.info("Deleting dataset: %s", name)
    if not DatasetRegistry.remove(name):
        raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found")
    return True
