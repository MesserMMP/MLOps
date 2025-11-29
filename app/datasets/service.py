import os
from pathlib import Path
from typing import Literal

from fastapi import UploadFile

from app.core.logging_config import setup_logging
from app.datasets.registry import register_dataset

logger = setup_logging()

DATA_DIR = Path("data")  # базовая директория для сырых датасетов
DATA_DIR.mkdir(exist_ok=True)


def save_uploaded_dataset(
    file: UploadFile,
    dataset_name: str | None = None,
    format_hint: Literal["csv", "json", "auto"] = "auto",
) -> str:
    """
    Сохраняет загруженный файл (csv/json) на диск и регистрирует его в реестре.
    Возвращает имя датасета.
    """
    if dataset_name is None:
        dataset_name = file.filename

    safe_name = dataset_name.replace(" ", "_")
    target_path = DATA_DIR / safe_name

    logger.info("Saving uploaded dataset: %s -> %s", file.filename, target_path)

    with target_path.open("wb") as f:
        content = file.file.read()
        f.write(content)

    # Небольшая эвристика по формату — пригодится, когда будешь делать DVC
    if format_hint == "auto":
        ext = os.path.splitext(file.filename)[1].lower()
        if ext in {".csv", ".json"}:
            format_hint = ext.lstrip(".")
        else:
            format_hint = "csv"

    register_dataset(name=safe_name, path=str(target_path), description=f"uploaded {format_hint} file")

    return safe_name
