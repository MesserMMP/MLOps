"""Backward-compatibility shim.  Logic has moved to :mod:`app.utils.clearml_wrapper`."""
from app.utils.clearml_wrapper import (  # noqa: F401
    init_task,
    list_published_models,
    load_model_from_clearml,
    get_task_metadata,
    PROJECT_NAME,
)
