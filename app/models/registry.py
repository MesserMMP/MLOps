from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


MODEL_CLASSES = {
    "logreg": {
        "class": LogisticRegression,
        "default_params": {"max_iter": 100},
    },
    "rf": {
        "class": RandomForestClassifier,
        "default_params": {"n_estimators": 100, "max_depth": None},
    },
}
