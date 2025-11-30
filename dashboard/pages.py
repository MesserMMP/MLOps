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
    predict,
)
from layout import page_header


# --- Status ---

def page_status():
    page_header("Service status", "Check backend health.")
    try:
        health = fetch_health()
        st.success(f"{health['status']} — {health['detail']}")
    except Exception as e:
        st.error(f"Service is not available: {e}")


# --- Datasets ---

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


# --- Training ---

def _get_default_params(
    classes: list[Dict[str, Any]], selected_class: str
) -> Dict[str, Any]:
    for c in classes:
        if c["name"] == selected_class:
            return c.get("default_params", {}) or {}
    return {}


def page_training():
    page_header("Training", "Configure and train models.")

    try:
        classes = fetch_model_classes()
    except Exception as e:
        st.error(f"Failed to fetch model classes: {e}")
        return

    if not classes:
        st.info("No model classes available.")
        return

    class_names = [c["name"] for c in classes]
    selected_class = st.selectbox("Model class", class_names)

    try:
        datasets = fetch_datasets()
    except Exception as e:
        st.error(f"Failed to fetch datasets: {e}")
        return

    if not datasets:
        st.info("Upload at least one dataset first on the Datasets page.")
        return

    dataset_names = [d["name"] for d in datasets]
    selected_dataset = st.selectbox("Dataset", dataset_names)

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

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("**Selected model**")
        st.write(f"`{selected_class}`")
    with col_right:
        st.markdown("**Selected dataset**")
        st.write(f"`{selected_dataset}`")

    if st.button("Train model", use_container_width=True):
        try:
            hyperparams = json.loads(hyperparams_str) if hyperparams_str.strip() else {}
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
            return

        with st.spinner("Training..."):
            try:
                resp = train_model(selected_dataset, selected_class, hyperparams)
                st.success(
                    f"Trained model `{resp['model_id']}` ({resp['model_class']})"
                )
            except Exception as e:
                st.error(f"Training failed: {e}")


# --- Inference ---

def _model_labels(models: List[Dict[str, Any]]) -> List[str]:
    return [f"{m['model_id']} ({m.get('model_class', 'unknown')})" for m in models]


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
