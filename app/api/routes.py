from typing import List

from fastapi import APIRouter, HTTPException, UploadFile, File

from app.models.registry import ModelRegistry
from app.services import model_service, dataset_service
from app.utils.logging_config import setup_logging
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
    """Return service liveness status."""
    logger.info("Health check requested")
    return StatusResponse(status="ok", detail="Service is running")


@router.get("/models/classes", response_model=ModelClassesResponse)
def list_model_classes():
    """Return all available model classes with their default hyperparameters."""
    logger.info("Listing model classes")
    classes: List[ModelClassInfo] = [
        ModelClassInfo(name=name, default_params=cfg["default_params"])
        for name, cfg in ModelRegistry.all_classes().items()
    ]
    return ModelClassesResponse(classes=classes)


@router.post("/models/train", response_model=TrainResponse)
def train_model(req: TrainRequest):
    """Train a new model and register it in ClearML."""
    logger.info(
        "Train requested: model_class=%s, dataset=%s, target=%s",
        req.model_class,
        req.dataset_name,
        req.target_column,
    )
    result = model_service.train(
        model_class=req.model_class,
        dataset_name=req.dataset_name,
        target_column=req.target_column,
        hyperparams=req.hyperparams,
        feature_columns=req.feature_columns,
    )
    return TrainResponse(**result)


@router.post("/models/retrain", response_model=TrainResponse)
def retrain_model(req: RetrainRequest):
    """Re-train an existing model with optional new hyperparameters."""
    logger.info("Retrain requested for model_id=%s", req.model_id)
    result = model_service.retrain(model_id=req.model_id, hyperparams=req.hyperparams)
    return TrainResponse(**result)


@router.get("/models/list")
def list_trained_models():
    """Return all trained models from ClearML."""
    return model_service.list_models()


@router.post("/models/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Run inference with a stored model."""
    predictions = model_service.predict(model_id=req.model_id, data=req.data)
    return PredictResponse(predictions=predictions)


@router.delete("/models/{model_id}")
def delete_model(model_id: str):
    """Archive (soft-delete) a model in ClearML."""
    logger.info("Archive requested for task_id=%s", model_id)
    return model_service.delete_model(model_id)


@router.get("/datasets", response_model=DatasetListResponse)
def list_datasets():
    """Return all registered datasets."""
    datasets = [
        DatasetInfo(name=m.name, description=m.description)
        for m in dataset_service.list_datasets()
    ]
    return DatasetListResponse(datasets=datasets)


@router.post("/datasets/upload", response_model=DatasetUploadResponse)
def upload_dataset(file: UploadFile = File(...)):
    """Upload a CSV or JSON dataset and register it (DVC-tracked)."""
    logger.info("Uploading dataset: %s", file.filename)
    meta = dataset_service.upload_dataset(file)
    return DatasetUploadResponse(name=meta.name, path=meta.path)


@router.delete("/datasets/{name}")
def delete_dataset(name: str):
    """Remove a dataset from the registry."""
    logger.info("Delete dataset requested: %s", name)
    dataset_service.delete_dataset(name)
    return {"detail": f"Dataset '{name}' removed from registry"}

