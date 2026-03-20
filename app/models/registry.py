from typing import Any, Dict, Optional

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


class ModelRegistry:
    """Static registry of available model classes and their default hyperparameters.

    Extending the catalogue is a matter of adding an entry to ``_classes``.
    """

    _classes: Dict[str, Dict[str, Any]] = {
        "logreg": {
            "class": LogisticRegression,
            "default_params": {"max_iter": 100},
        },
        "rf": {
            "class": RandomForestClassifier,
            "default_params": {"n_estimators": 100, "max_depth": None},
        },
    }

    @classmethod
    def has(cls, name: str) -> bool:
        """Return True if *name* is a known model class."""
        return name in cls._classes

    @classmethod
    def get(cls, name: str) -> Optional[Dict[str, Any]]:
        """Return the config dict for *name*, or *None* if unknown."""
        return cls._classes.get(name)

    @classmethod
    def all_classes(cls) -> Dict[str, Dict[str, Any]]:
        """Return a copy of the entire registry."""
        return dict(cls._classes)
