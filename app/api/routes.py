from uuid import uuid4
from typing import List

import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File

from app.core.logging_config import setup_logging
from app.models.registry import MODEL_CLASSES
from app.models.storage import TRAINED_MODELS
from app.datasets.registry import get_datasets_as_dicts, _DATASETS
from app.datasets.service import save_uploaded_dataset, load_xy_from_csv
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
    """
    Эндпоинт для проверки статуса сервиса.
    """
    logger.info("Health check requested")
    return StatusResponse(status="ok", detail="Service is running")


@router.get("/models/classes", response_model=ModelClassesResponse)
def list_model_classes():
    """
    Возвращает список доступных для обучения классов моделей
    и их дефолтные гиперпараметры.
    """
    logger.info("Listing model classes")
    classes: List[ModelClassInfo] = [
        ModelClassInfo(name=name, default_params=cfg["default_params"])
        for name, cfg in MODEL_CLASSES.items()
    ]
    return ModelClassesResponse(classes=classes)


@router.post("/models/train", response_model=TrainResponse)
def train_model(req: TrainRequest):
    """
    Обучает новую ML-модель на реальном CSV-датасете.
    Делит данные на X и y по target_column/feature_columns.
    """
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

    # Реальные данные из CSV
    X, y, used_features = load_xy_from_csv(
        dataset_name=req.dataset_name,
        target_column=req.target_column,
        feature_columns=req.feature_columns,
    )

    params = {**model_cfg["default_params"], **req.hyperparams}
    logger.info("Training %s with params=%s", req.model_class, params)

    model = model_cls(**params)
    model.fit(X, y)

    model_id = str(uuid4())
    TRAINED_MODELS[model_id] = {
        "model": model,
        "class": req.model_class,
        "dataset_name": req.dataset_name,
        "params": params,
        "target_column": req.target_column,
        "feature_columns": used_features,
    }

    logger.info("Model trained and stored: id=%s", model_id)
    return TrainResponse(model_id=model_id, model_class=req.model_class)


@router.post("/models/retrain", response_model=TrainResponse)
def retrain_model(req: RetrainRequest):
    """
    Переобучает существующую модель на том же датасете.
    Создаёт новую модель с новым model_id, не затирая старую.
    """
    logger.info("Retrain requested for model_id=%s", req.model_id)

    if req.model_id not in TRAINED_MODELS:
        logger.error("Model not found: %s", req.model_id)
        raise HTTPException(status_code=404, detail="Model not found")

    old_meta = TRAINED_MODELS[req.model_id]
    model_class_name = old_meta["class"]

    if model_class_name not in MODEL_CLASSES:
        logger.error("Unknown model_class in stored model: %s", model_class_name)
        raise HTTPException(status_code=500, detail="Invalid stored model_class")

    model_cfg = MODEL_CLASSES[model_class_name]
    model_cls = model_cfg["class"]

    dataset_name = old_meta["dataset_name"]
    target_column = old_meta["target_column"]
    feature_columns = old_meta["feature_columns"]

    X, y, used_features = load_xy_from_csv(
        dataset_name=dataset_name,
        target_column=target_column,
        feature_columns=feature_columns,
    )

    base_params = {**model_cfg["default_params"], **old_meta["params"]}
    params = {**base_params, **req.hyperparams}
    logger.info("Retraining %s with params=%s", model_class_name, params)

    model = model_cls(**params)
    model.fit(X, y)

    new_id = str(uuid4())
    TRAINED_MODELS[new_id] = {
        "model": model,
        "class": model_class_name,
        "dataset_name": dataset_name,
        "params": params,
        "target_column": target_column,
        "feature_columns": used_features,
    }

    logger.info("Model retrained and stored: id=%s (from %s)", new_id, req.model_id)
    return TrainResponse(model_id=new_id, model_class=model_class_name)


@router.get("/models/list")
def list_trained_models():
    """
    Возвращает краткую информацию о всех обученных моделях,
    доступных для инференса.
    """
    logger.info("Listing trained models")
    return [
        {
            "model_id": mid,
            "model_class": meta["class"],
            "dataset_name": meta["dataset_name"],
            "params": meta["params"],
            "target_column": meta["target_column"],
            "feature_columns": meta["feature_columns"],
        }
        for mid, meta in TRAINED_MODELS.items()
    ]


@router.post("/models/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Возвращает предсказания конкретной модели по переданным данным.
    """
    logger.info("Prediction requested for model_id=%s", req.model_id)

    if req.model_id not in TRAINED_MODELS:
        logger.error("Model not found: %s", req.model_id)
        raise HTTPException(status_code=404, detail="Model not found")

    model = TRAINED_MODELS[req.model_id]["model"]
    X = np.asarray(req.data)
    preds = model.predict(X)
    return PredictResponse(predictions=preds.tolist())


@router.delete("/models/{model_id}")
def delete_model(model_id: str):
    """
    Удаляет обученную модель из хранилища.
    """
    logger.info("Delete requested for model_id=%s", model_id)

    if model_id not in TRAINED_MODELS:
        logger.error("Model not found: %s", model_id)
        raise HTTPException(status_code=404, detail="Model not found")

    del TRAINED_MODELS[model_id]
    logger.info("Model deleted: %s", model_id)
    return {"detail": "deleted"}


@router.get("/datasets", response_model=DatasetListResponse)
def list_datasets():
    """
    Возвращает список зарегистрированных датасетов.
    """
    logger.info("Listing datasets")
    meta = get_datasets_as_dicts()
    datasets = [
        DatasetInfo(name=m["name"], description=m["description"])
        for m in meta
    ]
    return DatasetListResponse(datasets=datasets)


@router.post("/datasets/upload", response_model=DatasetUploadResponse)
def upload_dataset(file: UploadFile = File(...)):
    """
    Загружает новый датасет (csv/json), сохраняет на диск и
    регистрирует его в локальном реестре датасетов.
    """
    logger.info("Uploading dataset: %s", file.filename)
    dataset_name = save_uploaded_dataset(file)

    meta = _DATASETS[dataset_name]
    return DatasetUploadResponse(name=meta.name, path=meta.path)
