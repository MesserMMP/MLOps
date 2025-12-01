from typing import Dict, List
from dataclasses import dataclass, asdict


@dataclass
class DatasetMeta:
    name: str
    path: str
    description: str = ""


# Пока in-memory; позже можно заменить на файл/BД или DVC‑метаданные
_DATASETS: Dict[str, DatasetMeta] = {}


def register_dataset(name: str, path: str, description: str = "") -> DatasetMeta:
    meta = DatasetMeta(name=name, path=path, description=description)
    _DATASETS[name] = meta
    return meta


def list_datasets() -> List[DatasetMeta]:
    return list(_DATASETS.values())


def get_datasets_as_dicts() -> List[dict]:
    return [asdict(m) for m in _DATASETS.values()]


def get_dataset(name: str) -> DatasetMeta | None:
    return _DATASETS.get(name)
