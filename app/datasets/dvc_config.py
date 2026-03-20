# app/datasets/dvc_config.py
from pathlib import Path

from dvc.repo import Repo

from app.utils.logging_config import setup_logging

logger = setup_logging()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


def get_dvc_repo() -> Repo:
    """
    Возвращает объект DVC Repo для корня проекта.
    Предполагается, что `dvc init` уже выполнен.
    """
    return Repo(str(PROJECT_ROOT))


def ensure_data_dir():
    DATA_DIR.mkdir(exist_ok=True)
