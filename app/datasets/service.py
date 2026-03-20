import os
from pathlib import Path
from typing import Literal, List

import pandas as pd
from fastapi import HTTPException, UploadFile

from app.utils.logging_config import setup_logging
from app.datasets.registry import DatasetRegistry
from app.datasets.dvc_config import get_dvc_repo, DATA_DIR, ensure_data_dir

logger = setup_logging()

ensure_data_dir()


def save_uploaded_dataset(
    file: UploadFile,
    dataset_name: str | None = None,
    format_hint: Literal["csv", "json", "auto"] = "auto",
) -> str:
    """Save an uploaded CSV/JSON file to disk, register it and track with DVC.

    Returns the canonical dataset name (sanitised filename).
    """
    if dataset_name is None:
        dataset_name = file.filename

    safe_name = dataset_name.replace(" ", "_")
    target_path = DATA_DIR / safe_name

    logger.info("Saving uploaded dataset: %s -> %s", file.filename, target_path)

    with target_path.open("wb") as f:
        content = file.file.read()
        f.write(content)

    if format_hint == "auto":
        ext = os.path.splitext(file.filename)[1].lower()
        if ext in {".csv", ".json"}:
            format_hint = ext.lstrip(".")
        else:
            format_hint = "csv"

    DatasetRegistry.register(
        name=safe_name,
        path=str(target_path),
        description=f"uploaded {format_hint} file (DVC-tracked)",
    )

    # DVC add + push
    try:
        repo = get_dvc_repo()
        rel_path = os.path.relpath(target_path, repo.root_dir)
        logger.info("DVC add %s", rel_path)
        repo.add(rel_path)
        logger.info("DVC push")
        repo.push()
    except Exception as e:
        logger.exception("Failed to run DVC for dataset %s: %s", safe_name, e)

    return safe_name


def load_xy_from_csv(
    dataset_name: str,
    target_column: str,
    feature_columns: List[str] | None,
):
    """Load a CSV dataset and split it into feature matrix X and target vector y.

    Parameters
    ----------
    dataset_name:
        Name as stored in :class:`DatasetRegistry`.
    target_column:
        Column to use as the prediction target.
    feature_columns:
        Explicit list of feature columns.  When *None* all numeric columns
        (excluding obvious index columns) are used.
    """
    meta = DatasetRegistry.get(dataset_name)
    if meta is None:
        logger.error("Dataset not found: %s", dataset_name)
        raise HTTPException(status_code=404, detail="Dataset not found")

    logger.info("Loading dataset from %s", meta.path)
    df = pd.read_csv(meta.path)

    if target_column not in df.columns:
        logger.error("Target column %s not in dataset columns", target_column)
        raise HTTPException(status_code=400, detail="target_column not in dataset")

    index_like_cols = {"id", "ID", "index", "Index"}
    index_like_cols |= {c for c in df.columns if c.lower().startswith("unnamed:")}

    if feature_columns is None:
        numeric_cols = [
            c
            for c in df.columns
            if c != target_column
            and c not in index_like_cols
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        if not numeric_cols:
            raise HTTPException(
                status_code=400,
                detail="No numeric feature columns found (after dropping index columns)",
            )
        used_features = numeric_cols
    else:
        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            logger.error("Feature columns not in dataset: %s", missing)
            raise HTTPException(
                status_code=400,
                detail=f"Feature columns not in dataset: {missing}",
            )
        used_features = feature_columns

    X = df[used_features].to_numpy()
    y = df[target_column].to_numpy()
    logger.info(
        "Loaded dataset %s: X.shape=%s, y.shape=%s, target=%s, features=%s",
        dataset_name,
        X.shape,
        y.shape,
        target_column,
        used_features,
    )
    return X, y, used_features
