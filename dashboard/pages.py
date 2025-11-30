import json
from typing import Any, Dict, List

import streamlit as st

from api_client import (
    fetch_health,
    fetch_datasets,
    fetch_model_classes,
    fetch_models,
    upload_dataset,
    train_model,
    retrain_model,
    predict,
)
from layout import page_header


# ---------- Status ----------

def page_status():
    page_header("Service status", "Check backend health.")
    try:
        health = fetch_health()
        st.success(f"{health['status']} — {health['detail']}")
    except Exception as e:
        st.error(f"Service is not available: {e}")


# ---------- Datasets ----------

def page_datasets():
    page_header("Datasets", "Manage uploaded datasets used for training.")

    col_left, col_right = st.columns([1.1, 1])

    with col_left:
        st.markdown("#### Existing datasets")
        try:
            datasets = fetch_datasets()
            if datasets:
                st.dataframe(datasets, use_container_width=True, hide_index=True)
            else:
                st.info("No datasets registered yet.")
        except Exception as e:
            st.error(f"Failed to fetch datasets: {e}")

    with col_right:
        st.markdown("#### Upload new dataset")
        st.caption("Supported formats: CSV, JSON")

        uploaded_file = st.file_uploader(
            "Drag & drop file here or browse",
            type=["csv", "json"],
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            if st.button("Upload", use_container_width=True):
                try:
                    result = upload_dataset(uploaded_file)
                    st.success(f"Uploaded: {result['name']}")
                    st.caption(result.get("path", ""))
                except Exception as e:
                    st.error(f"Upload failed: {e}")


# ---------- Training ----------

def _get_default_params(
    classes: list[Dict[str, Any]], selected_class: str
) -> Dict[str, Any]:
    for c in classes:
        if c["name"] == selected_class:
            return c.get("default_params", {}) or {}
    return {}


def _model_labels(models: List[Dict[str, Any]]) -> List[str]:
    return [f"{m['model_id']} ({m.get('model_class', 'unknown')})" for m in models]


def page_training():
    page_header("Training", "Configure, train and retrain models on CSV datasets.")

    # модельные классы
    try:
        classes = fetch_model_classes()
    except Exception as e:
        st.error(f"Failed to fetch model classes: {e}")
        return

    if not classes:
        st.info("No model classes available.")
        return

    # датасеты
    try:
        datasets = fetch_datasets()
    except Exception as e:
        st.error(f"Failed to fetch datasets: {e}")
        return

    if not datasets:
        st.info("Upload at least one dataset first on the Datasets page.")
        return

    class_names = [c["name"] for c in classes]
    dataset_names = [d["name"] for d in datasets]

    col_top_left, col_top_right = st.columns(2)
    with col_top_left:
        selected_class = st.selectbox("Model class", class_names)
    with col_top_right:
        selected_dataset = st.selectbox("Dataset", dataset_names)

    st.subheader("Target and features")
    col_target, col_features = st.columns(2)
    with col_target:
        target_column = st.text_input("Target column name", value="diagnosis")
    with col_features:
        features_raw = st.text_input(
            "Feature columns (comma-separated, empty = auto detect numeric)",
            value="",
        )
        feature_columns: List[str] | None
        if features_raw.strip():
            feature_columns = [c.strip() for c in features_raw.split(",") if c.strip()]
        else:
            feature_columns = None

    default_params = _get_default_params(classes, selected_class)
    st.subheader("Hyperparameters")
    st.caption("Edit JSON if needed; defaults are provided for the selected model class.")

    default_json = json.dumps(default_params, indent=2)
    hyperparams_str = st.text_area(
        "Hyperparameters JSON",
        value=default_json,
        height=200,
        label_visibility="collapsed",
    )

    # выбор модели для retrain
    st.markdown("#### Optional: retrain existing model")
    try:
        models = fetch_models()
    except Exception as e:
        st.error(f"Failed to fetch models: {e}")
        models = []

    model_id_for_retrain: str | None = None
    if models:
        labels = _model_labels(models)
        selected_label = st.selectbox(
            "Existing model to retrain (optional)",
            ["— none —"] + labels,
        )
        if selected_label != "— none —":
            idx = labels.index(selected_label)
            model_id_for_retrain = models[idx]["model_id"]
    else:
        st.caption("No trained models yet for retraining.")

    col_buttons_left, col_buttons_right = st.columns(2)

    # Train new
    with col_buttons_left:
        if st.button("Train new model", use_container_width=True):
            if not target_column.strip():
                st.error("Target column name is required.")
                return

            try:
                hyperparams = json.loads(hyperparams_str) if hyperparams_str.strip() else {}
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")
                return

            with st.spinner("Training..."):
                try:
                    resp = train_model(
                        dataset_name=selected_dataset,
                        model_class=selected_class,
                        hyperparams=hyperparams,
                        target_column=target_column.strip(),
                        feature_columns=feature_columns,
                    )
                    st.success(
                        f"Trained model `{resp['model_id']}` ({resp['model_class']})"
                    )
                except Exception as e:
                    st.error(f"Training failed: {e}")

    # Retrain
    with col_buttons_right:
        if st.button("Retrain selected model", use_container_width=True):
            if not model_id_for_retrain:
                st.error("Select an existing model to retrain.")
                return

            try:
                hyperparams = json.loads(hyperparams_str) if hyperparams_str.strip() else {}
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")
                return

            with st.spinner("Retraining..."):
                try:
                    resp = retrain_model(model_id_for_retrain, hyperparams)
                    st.success(
                        f"Retrained model `{resp['model_id']}` "
                        f"({resp['model_class']}) from `{model_id_for_retrain}`"
                    )
                except Exception as e:
                    st.error(f"Retraining failed: {e}")


# ---------- Inference ----------

def page_inference():
    page_header("Inference", "Run predictions using trained models.")

    try:
        models = fetch_models()
    except Exception as e:
        st.error(f"Failed to fetch models: {e}")
        return

    if not models:
        st.info("No trained models yet. Train a model first on the Training page.")
        return

    labels = _model_labels(models)
    selected_label = st.selectbox("Model", labels)
    idx = labels.index(selected_label)
    model_id = models[idx]["model_id"]

    st.subheader("Input data")
    st.caption("JSON list of rows, e.g. [[0.1, 0.2, 0.3, 0.4]]")

    default_data = "[[0.1, 0.2, 0.3, 0.4]]"
    data_str = st.text_area(
        "Data JSON",
        value=default_data,
        height=200,
        label_visibility="collapsed",
    )

    if st.button("Predict", use_container_width=True):
        try:
            data = json.loads(data_str)
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
            return

        if not isinstance(data, list):
            st.error("Data must be a list of rows.")
            return

        with st.spinner("Predicting..."):
            try:
                resp = predict(model_id, data)
                st.subheader("Predictions")
                st.write(resp["predictions"])
            except Exception as e:
                st.error(f"Prediction failed: {e}")
