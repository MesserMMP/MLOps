import os
import joblib
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File

from app.core.logging_config import setup_logging
from app.models.registry import MODEL_CLASSES
from app.datasets.registry import get_datasets_as_dicts, _DATASETS
from app.datasets.service import save_uploaded_dataset, load_xy_from_csv

from app.core.clearml_wrapper import (
    init_task,
    list_published_models,
    load_model_from_clearml,
    get_task_metadata
)

from app.schemas.api import (
    TrainRequest,
    TrainResponse,
    RetrainRequest,
    PredictRequest,
    PredictResponse,
    StatusResponse,
    ModelClassesResponse,
    ModelClassInfo,
    DatasetListResponse,
    DatasetInfo,
    DatasetUploadResponse,
)

logger = setup_logging()
router = APIRouter()


@router.get("/health", response_model=StatusResponse)
def health_check():
    logger.info("Health check requested")
    return StatusResponse(status="ok", detail="Service is running")


@router.get("/models/classes", response_model=ModelClassesResponse)
def list_model_classes():
    logger.info("Listing model classes")
    classes: List[ModelClassInfo] = [
        ModelClassInfo(name=name, default_params=cfg["default_params"])
        for name, cfg in MODEL_CLASSES.items()
    ]
    return ModelClassesResponse(classes=classes)


@router.post("/models/train", response_model=TrainResponse)
def train_model(req: TrainRequest):
    logger.info(
        "Train requested: model_class=%s, dataset=%s, target=%s",
        req.model_class,
        req.dataset_name,
        req.target_column,
    )

    if req.model_class not in MODEL_CLASSES:
        logger.error("Unknown model_class: %s", req.model_class)
        raise HTTPException(status_code=400, detail="Unknown model_class")

    model_cfg = MODEL_CLASSES[req.model_class]
    model_cls = model_cfg["class"]

    task_name = f"Train {req.model_class} on {req.dataset_name}"
    task = init_task(task_name=task_name)

    try:
        # Логируем параметры
        params = {**model_cfg["default_params"], **req.hyperparams}
        task.connect(params, name="Hyperparameters")

        # --- ИЗМЕНЕНИЕ: Поддержка синтетики ---
        if req.dataset_name == "synthetic":
            logger.info("Generating synthetic data for training")
            # Генерируем случайные данные (100 семплов, 4 фичи)
            X = np.random.randn(100, 4)
            y = np.random.randint(0, 2, size=100)
            used_features = ["feat_0", "feat_1", "feat_2", "feat_3"]

            metadata = {
                "dataset_name": "synthetic",
                "model_class": req.model_class,
                "target_column": "synthetic_target",
                "feature_columns": ",".join(used_features)
            }
        else:
            # Обычный режим: загрузка с диска
            X, y, used_features = load_xy_from_csv(
                dataset_name=req.dataset_name,
                target_column=req.target_column,
                feature_columns=req.feature_columns,
            )
            metadata = {
                "dataset_name": req.dataset_name,
                "model_class": req.model_class,
                "target_column": req.target_column,
                "feature_columns": ",".join(used_features)
            }
        # --------------------------------------

        task.connect(metadata, name="Metadata")

        logger.info("Training %s with params=%s", req.model_class, params)
        model = model_cls(**params)
        model.fit(X, y)

        local_model_path = f"model_{task.id}.pkl"
        joblib.dump(model, local_model_path)

        task.update_output_model(model_path=local_model_path, model_name=task_name)

        logger.info("Model trained and uploaded to ClearML: task_id=%s", task.id)

        if os.path.exists(local_model_path):
            os.remove(local_model_path)

        return TrainResponse(model_id=task.id, model_class=req.model_class)

    except Exception as e:
        logger.error(f"Training failed: {e}")
        task.close()
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")
    finally:
        task.close()


@router.post("/models/retrain", response_model=TrainResponse)
def retrain_model(req: RetrainRequest):
    logger.info("Retrain requested for model_id=%s", req.model_id)

    try:
        old_meta = get_task_metadata(req.model_id)
    except ValueError:
        logger.error("Model task not found: %s", req.model_id)
        raise HTTPException(status_code=404, detail="Model task not found in ClearML")

    model_class_name = old_meta["model_class"]
    if not model_class_name or model_class_name not in MODEL_CLASSES:
        raise HTTPException(status_code=500, detail=f"Unknown or missing model class in metadata: {model_class_name}")

    model_cfg = MODEL_CLASSES[model_class_name]
    model_cls = model_cfg["class"]

    dataset_name = old_meta["dataset_name"]
    target_column = old_meta["target_column"]
    feature_columns = old_meta["feature_columns"]

    task_name = f"Retrain {model_class_name} on {dataset_name}"
    task = init_task(task_name=task_name)

    try:
        base_params = {**model_cfg["default_params"], **old_meta["params"]}
        final_params = {**base_params, **req.hyperparams}

        task.connect(final_params, name="Hyperparameters")

        # --- ИЗМЕНЕНИЕ: Поддержка синтетики в retrain ---
        if dataset_name == "synthetic":
            logger.info("Generating synthetic data for retrain")
            X = np.random.randn(100, 4)
            y = np.random.randint(0, 2, size=100)
            used_features = feature_columns if feature_columns else ["feat_0", "feat_1", "feat_2", "feat_3"]
        else:
            X, y, used_features = load_xy_from_csv(
                dataset_name=dataset_name,
                target_column=target_column,
                feature_columns=feature_columns,
            )
        # -----------------------------------------------

        metadata = {
            "dataset_name": dataset_name,
            "model_class": model_class_name,
            "target_column": target_column,
            "feature_columns": ",".join(used_features) if used_features else ""
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
        return TrainResponse(model_id=task.id, model_class=model_class_name)

    except Exception as e:
        task.close()
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")
    finally:
        task.close()


@router.get("/models/list")
def list_trained_models():
    logger.info("Listing trained models from ClearML")
    try:
        return list_published_models()
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch models from ClearML: {e}")


@router.post("/models/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    logger.info("Prediction requested for model_id=%s", req.model_id)
    try:
        model = load_model_from_clearml(req.model_id)
    except ValueError as e:
        logger.error(f"Model load failed: {e}")
        raise HTTPException(status_code=404, detail="Model not found or could not be loaded")
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    X = np.asarray(req.data)
    try:
        preds = model.predict(X)
        return PredictResponse(predictions=preds.tolist())
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference failed: {e}")


@router.delete("/models/{model_id}")
def delete_model(model_id: str):
    logger.info("Archive requested for task_id=%s", model_id)
    from clearml import Task
    try:
        task = Task.get_task(task_id=model_id)
        if task:
            task.set_archived(True)
            return {"detail": "model archived"}
        else:
            raise HTTPException(status_code=404, detail="Model not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {e}")


@router.get("/datasets", response_model=DatasetListResponse)
def list_datasets():
    logger.info("Listing datasets")
    meta = get_datasets_as_dicts()
    datasets = [
        DatasetInfo(name=m["name"], description=m["description"])
        for m in meta
    ]
    return DatasetListResponse(datasets=datasets)


@router.post("/datasets/upload", response_model=DatasetUploadResponse)
def upload_dataset(file: UploadFile = File(...)):
    logger.info("Uploading dataset: %s", file.filename)
    dataset_name = save_uploaded_dataset(file)

    meta = _DATASETS[dataset_name]
    return DatasetUploadResponse(name=meta.name, path=meta.path)
