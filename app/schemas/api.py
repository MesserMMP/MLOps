from typing import Any, Dict, List
from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    dataset_name: str = Field(
        ...,
        description="Имя датасета (совпадает с name из реестра /datasets)",
    )
    model_class: str = Field(
        ...,
        description="Класс модели: logreg или rf",
    )
    hyperparams: Dict[str, Any] = Field(
        default_factory=dict,
        description="Гиперпараметры конкретной модели",
    )
    target_column: str = Field(
        ...,
        description="Имя целевой колонки в CSV",
    )
    feature_columns: List[str] | None = Field(
        default=None,
        description="Явный список фич; если None — все числовые колонки, кроме target",
    )


class TrainResponse(BaseModel):
    model_id: str
    model_class: str


class RetrainRequest(BaseModel):
    model_id: str = Field(..., description="ID существующей модели")
    hyperparams: Dict[str, Any] = Field(
        default_factory=dict,
        description="Новые гиперпараметры; переопределяют старые",
    )


class PredictRequest(BaseModel):
    model_id: str
    data: List[List[float]]


class PredictResponse(BaseModel):
    predictions: List[Any]


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

class DatasetUploadResponse(BaseModel):
    name: str
    path: str
