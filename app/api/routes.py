from uuid import uuid4
from typing import List

import numpy as np
from fastapi import APIRouter, HTTPException

from app.logging_config import setup_logging
from app.models.registry import MODEL_CLASSES
from app.models.storage import TRAINED_MODELS
from app.schemas.api import (
    TrainRequest,
    TrainResponse,
    PredictRequest,
    PredictResponse,
    StatusResponse,
    ModelClassesResponse,
    ModelClassInfo,
    DatasetListResponse,
    DatasetInfo,
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
    logger.info("Train requested: %s", req.model_class)

    if req.model_class not in MODEL_CLASSES:
        logger.error("Unknown model_class: %s", req.model_class)
        raise HTTPException(status_code=400, detail="Unknown model_class")

    model_cfg = MODEL_CLASSES[req.model_class]
    model_cls = model_cfg["class"]

    # Заглушка: бинарная классификация с синтетическими данными
    X = np.random.randn(100, 4)
    y = np.random.randint(0, 2, size=100)

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
    }

    logger.info("Model trained and stored: id=%s", model_id)
    return TrainResponse(model_id=model_id, model_class=req.model_class)


@router.get("/models/list")
def list_trained_models():
    logger.info("Listing trained models")
    return [
        {
            "model_id": mid,
            "model_class": meta["class"],
            "dataset_name": meta["dataset_name"],
            "params": meta["params"],
        }
        for mid, meta in TRAINED_MODELS.items()
    ]


@router.post("/models/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
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
    logger.info("Delete requested for model_id=%s", model_id)

    if model_id not in TRAINED_MODELS:
        logger.error("Model not found: %s", model_id)
        raise HTTPException(status_code=404, detail="Model not found")

    del TRAINED_MODELS[model_id]
    logger.info("Model deleted: %s", model_id)
    return {"detail": "deleted"}


@router.get("/datasets", response_model=DatasetListResponse)
def list_datasets():
    logger.info("Listing datasets (stub)")
    datasets = [
        DatasetInfo(name="demo_dataset", description="Stub dataset"),
    ]
    return DatasetListResponse(datasets=datasets)
