from typing import Dict, Any, List, Optional
import joblib
import os

from clearml import Task, Model

from app.core.logging_config import setup_logging

logger = setup_logging()

PROJECT_NAME = "MLOps_Homework"


def init_task(task_name: str, task_type: str = "training") -> Task:
    """
    Инициализирует задачу в ClearML.
    """
    # auto_connect_frameworks=False отключает авто-магию sklearn
    task = Task.init(
        project_name=PROJECT_NAME,
        task_name=task_name,
        task_type=task_type,
        output_uri="s3://mlops-datasets/models",
        auto_connect_frameworks={"sklearn": False}
    )
    return task


def list_published_models() -> List[Dict[str, Any]]:
    """
    Получает список моделей из ClearML.
    """
    tasks = Task.get_tasks(
        project_name=PROJECT_NAME,
        task_filter={
            "status": ["completed", "published"],
            "type": ["training"]
        }
    )

    models_data = []
    for t in tasks:
        if not t.models or not t.models.get('output'):
            continue

        model = list(t.models['output'])[0]

        # Получаем все параметры
        all_params = t.get_parameters_as_dict()

        # Достаем наши секции
        hyperparams = all_params.get("Hyperparameters", {})
        metadata = all_params.get("Metadata", {})

        models_data.append({
            "model_id": t.id,
            "model_class": metadata.get("model_class", "unknown"),
            "dataset_name": metadata.get("dataset_name", "unknown"),
            "params": hyperparams,
            "target_column": metadata.get("target_column"),
            "feature_columns": metadata.get("feature_columns", "").split(",") if metadata.get(
                "feature_columns") else None,
            "clearml_model_id": model.id,
            "model_url": model.url
        })
    return models_data


def load_model_from_clearml(task_id: str):
    """
    Скачивает и загружает модель по ID задачи.
    """
    task = Task.get_task(task_id=task_id)
    if not task:
        raise ValueError(f"Task {task_id} not found")

    if not task.models or not task.models.get('output'):
        raise ValueError(f"Task {task_id} has no output models")

    model_obj = list(task.models['output'])[0]
    local_path = model_obj.get_local_copy()

    if not local_path:
        raise ValueError("Failed to download model file from ClearML/S3")

    return joblib.load(local_path)


def get_task_metadata(task_id: str) -> Dict[str, Any]:
    """
    Получает метаданные задачи для переобучения.
    """
    task = Task.get_task(task_id=task_id)
    if not task:
        raise ValueError(f"Task {task_id} not found")

    all_params = task.get_parameters_as_dict()
    hyperparams = all_params.get("Hyperparameters", {})
    metadata = all_params.get("Metadata", {})

    return {
        "model_class": metadata.get("model_class"),
        "dataset_name": metadata.get("dataset_name"),
        "target_column": metadata.get("target_column"),
        "feature_columns": metadata.get("feature_columns", "").split(",") if metadata.get("feature_columns") else None,
        "params": hyperparams
    }
