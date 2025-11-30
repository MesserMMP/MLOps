from typing import Any, Dict, List

import requests
from config import API_URL


def _get(path: str) -> Dict[str, Any] | List[Dict[str, Any]]:
    resp = requests.get(f"{API_URL}{path}")
    resp.raise_for_status()
    return resp.json()


def _post_json(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(f"{API_URL}{path}", json=payload)
    resp.raise_for_status()
    return resp.json()


def _post_file(path: str, file) -> Dict[str, Any]:
    files = {"file": (file.name, file.getvalue())}
    resp = requests.post(f"{API_URL}{path}", files=files)
    resp.raise_for_status()
    return resp.json()


def fetch_health() -> Dict[str, Any]:
    return _get("/health")


def fetch_model_classes() -> List[Dict[str, Any]]:
    data = _get("/models/classes")
    return data["classes"]


def fetch_datasets() -> List[Dict[str, Any]]:
    data = _get("/datasets")
    return data["datasets"]


def fetch_models() -> List[Dict[str, Any]]:
    return _get("/models/list")


def upload_dataset(file) -> Dict[str, Any]:
    return _post_file("/datasets/upload", file)


def train_model(
    dataset_name: str,
    model_class: str,
    hyperparams: Dict[str, Any],
    target_column: str,
    feature_columns: List[str] | None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "dataset_name": dataset_name,
        "model_class": model_class,
        "hyperparams": hyperparams,
        "target_column": target_column,
        "feature_columns": feature_columns,
    }
    return _post_json("/models/train", payload)


def retrain_model(model_id: str, hyperparams: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "model_id": model_id,
        "hyperparams": hyperparams,
    }
    return _post_json("/models/retrain", payload)


def predict(model_id: str, data: List[List[float]]) -> Dict[str, Any]:
    payload = {
        "model_id": model_id,
        "data": data,
    }
    return _post_json("/models/predict", payload)
