"""Business logic for model training, retraining, prediction and management.

Routes and gRPC handlers delegate all heavy lifting here so they stay thin.
"""

import os
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from fastapi import HTTPException

from app.models.registry import ModelRegistry
from app.datasets.service import load_xy_from_csv
from app.utils.clearml_wrapper import (
    init_task,
    list_published_models,
    load_model_from_clearml,
    get_task_metadata,
)
from app.utils.logging_config import setup_logging

logger = setup_logging()


def train(
    model_class: str,
    dataset_name: str,
    target_column: str,
    hyperparams: Dict[str, Any],
    feature_columns: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Train a new model and register it in ClearML.

    Returns a dict with ``model_id`` and ``model_class``.
    """
    if not ModelRegistry.has(model_class):
        logger.error("Unknown model_class: %s", model_class)
        raise HTTPException(status_code=400, detail="Unknown model_class")

    model_cfg = ModelRegistry.get(model_class)
    model_cls = model_cfg["class"]

    task_name = f"Train {model_class} on {dataset_name}"
    task = init_task(task_name=task_name)

    try:
        params = {**model_cfg["default_params"], **hyperparams}
        task.connect(params, name="Hyperparameters")

        if dataset_name == "synthetic":
            logger.info("Generating synthetic data for training")
            X = np.random.randn(100, 4)
            y = np.random.randint(0, 2, size=100)
            used_features = ["feat_0", "feat_1", "feat_2", "feat_3"]
            metadata = {
                "dataset_name": "synthetic",
                "model_class": model_class,
                "target_column": "synthetic_target",
                "feature_columns": ",".join(used_features),
            }
        else:
            X, y, used_features = load_xy_from_csv(
                dataset_name=dataset_name,
                target_column=target_column,
                feature_columns=feature_columns,
            )
            metadata = {
                "dataset_name": dataset_name,
                "model_class": model_class,
                "target_column": target_column,
                "feature_columns": ",".join(used_features),
            }

        task.connect(metadata, name="Metadata")
        logger.info("Training %s with params=%s", model_class, params)

        model = model_cls(**params)
        model.fit(X, y)

        local_model_path = f"model_{task.id}.pkl"
        joblib.dump(model, local_model_path)
        task.update_output_model(model_path=local_model_path, model_name=task_name)
        logger.info("Model trained and uploaded to ClearML: task_id=%s", task.id)

        if os.path.exists(local_model_path):
            os.remove(local_model_path)

        return {"model_id": task.id, "model_class": model_class}

    except HTTPException:
        task.close()
        raise
    except Exception as exc:
        logger.error("Training failed: %s", exc)
        task.close()
        raise HTTPException(status_code=500, detail=f"Training failed: {exc}") from exc
    finally:
        task.close()


def retrain(
    model_id: str,
    hyperparams: Dict[str, Any],
) -> Dict[str, str]:
    """Re-train an existing model (by ClearML task ID) with optional new hyperparameters.

    Returns a dict with ``model_id`` and ``model_class``.
    """
    try:
        old_meta = get_task_metadata(model_id)
    except ValueError:
        logger.error("Model task not found: %s", model_id)
        raise HTTPException(status_code=404, detail="Model task not found in ClearML")

    model_class_name = old_meta["model_class"]
    if not model_class_name or not ModelRegistry.has(model_class_name):
        raise HTTPException(
            status_code=500,
            detail=f"Unknown or missing model class in metadata: {model_class_name}",
        )

    model_cfg = ModelRegistry.get(model_class_name)
    model_cls = model_cfg["class"]

    dataset_name = old_meta["dataset_name"]
    target_column = old_meta["target_column"]
    feature_columns = old_meta["feature_columns"]

    task_name = f"Retrain {model_class_name} on {dataset_name}"
    task = init_task(task_name=task_name)

    try:
        final_params = {**model_cfg["default_params"], **old_meta["params"], **hyperparams}
        task.connect(final_params, name="Hyperparameters")

        if dataset_name == "synthetic":
            logger.info("Generating synthetic data for retrain")
            X = np.random.randn(100, 4)
            y = np.random.randint(0, 2, size=100)
            used_features = feature_columns or ["feat_0", "feat_1", "feat_2", "feat_3"]
        else:
            X, y, used_features = load_xy_from_csv(
                dataset_name=dataset_name,
                target_column=target_column,
                feature_columns=feature_columns,
            )

        metadata = {
            "dataset_name": dataset_name,
            "model_class": model_class_name,
            "target_column": target_column,
            "feature_columns": ",".join(used_features) if used_features else "",
        }
        task.connect(metadata, name="Metadata")

        logger.info("Retraining %s with params=%s", model_class_name, final_params)
        model = model_cls(**final_params)
        model.fit(X, y)

        local_model_path = f"model_{task.id}.pkl"
        joblib.dump(model, local_model_path)
        task.update_output_model(model_path=local_model_path, model_name=task_name)

        if os.path.exists(local_model_path):
            os.remove(local_model_path)

        logger.info("Model retrained: new_task_id=%s", task.id)
        return {"model_id": task.id, "model_class": model_class_name}

    except HTTPException:
        task.close()
        raise
    except Exception as exc:
        task.close()
        raise HTTPException(status_code=500, detail=f"Retraining failed: {exc}") from exc
    finally:
        task.close()


def predict(model_id: str, data: List[List[float]]) -> List[Any]:
    """Run inference with a stored model and return predictions."""
    logger.info("Prediction requested for model_id=%s", model_id)
    try:
        model = load_model_from_clearml(model_id)
    except ValueError as exc:
        logger.error("Model load failed: %s", exc)
        raise HTTPException(status_code=404, detail="Model not found or could not be loaded") from exc
    except Exception as exc:
        logger.error("Prediction error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    X = np.asarray(data)
    try:
        return model.predict(X).tolist()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Inference failed: {exc}") from exc


def list_models() -> List[Dict[str, Any]]:
    """Return all available trained models from ClearML."""
    logger.info("Listing trained models from ClearML")
    try:
        return list_published_models()
    except Exception as exc:
        logger.error("Failed to list models: %s", exc)
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch models from ClearML: {exc}"
        ) from exc


def delete_model(model_id: str) -> Dict[str, str]:
    """Archive a ClearML task (soft-delete)."""
    from clearml import Task

    logger.info("Archive requested for task_id=%s", model_id)
    try:
        task = Task.get_task(task_id=model_id)
        if task:
            task.set_archived(True)
            return {"detail": "model archived"}
        raise HTTPException(status_code=404, detail="Model not found")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {exc}") from exc
