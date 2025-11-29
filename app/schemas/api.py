from typing import Dict, Any, List
from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    dataset_name: str = Field(..., description="Имя датасета (пока заглушка)")
    model_class: str = Field(..., description="Класс модели: logreg или rf")
    hyperparams: Dict[str, Any] = Field(
        default_factory=dict,
        description="Гиперпараметры конкретной модели",
    )


class TrainResponse(BaseModel):
    model_id: str
    model_class: str


class PredictRequest(BaseModel):
    model_id: str
    data: List[List[float]]


class PredictResponse(BaseModel):
    predictions: List[float]


class StatusResponse(BaseModel):
    status: str
    detail: str


class ModelClassInfo(BaseModel):
    name: str
    default_params: Dict[str, Any]


class ModelClassesResponse(BaseModel):
    classes: List[ModelClassInfo]


class DatasetInfo(BaseModel):
    name: str
    description: str = ""


class DatasetListResponse(BaseModel):
    datasets: List[DatasetInfo]
