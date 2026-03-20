"""gRPC server implementation for the ML service.

All business logic is delegated to :mod:`app.services.model_service` and
:mod:`app.services.dataset_service` to keep this module thin.
"""

import json
from concurrent import futures

import grpc
import numpy as np

from app.utils.logging_config import setup_logging
from app.models.registry import ModelRegistry
from app.services import model_service, dataset_service
from app.grpc import ml_service_pb2 as pb2
from app.grpc import ml_service_pb2_grpc as pb2_grpc

logger = setup_logging()


class MLServiceServicer(pb2_grpc.MLServiceServicer):
    """Implements all RPCs defined in *ml_service.proto*."""

    # ── Model RPCs ────────────────────────────────────────────────────────────

    def ListModelClasses(self, request, context):
        """Return all available model classes."""
        logger.info("gRPC ListModelClasses called")
        classes = []
        for name, cfg in ModelRegistry.all_classes().items():
            classes.append(
                pb2.ModelClassInfo(
                    name=name,
                    default_params_json=json.dumps(cfg["default_params"]),
                )
            )
        return pb2.ListModelClassesResponse(classes=classes)

    def TrainModel(self, request, context):
        """Train a new model."""
        logger.info("gRPC TrainModel called: %s (ds: %s)", request.model_class, request.dataset_name)

        if not ModelRegistry.has(request.model_class):
            context.set_details("Unknown model_class")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return pb2.TrainModelResponse()

        try:
            hyperparams = json.loads(request.hyperparams_json) if request.hyperparams_json else {}
        except json.JSONDecodeError:
            context.set_details("Invalid hyperparams_json")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return pb2.TrainModelResponse()

        feature_cols = (
            [x.strip() for x in request.feature_columns_csv.split(",") if x.strip()]
            if request.feature_columns_csv
            else None
        )
        target_col = request.target_column or "synthetic_target"

        try:
            result = model_service.train(
                model_class=request.model_class,
                dataset_name=request.dataset_name,
                target_column=target_col,
                hyperparams=hyperparams,
                feature_columns=feature_cols,
            )
        except Exception as exc:
            logger.error("gRPC Train Error: %s", exc)
            context.set_details(f"Train failed: {exc}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return pb2.TrainModelResponse()

        return pb2.TrainModelResponse(model_id=result["model_id"], model_class=result["model_class"])

    def RetrainModel(self, request, context):
        """Re-train an existing model."""
        logger.info("gRPC RetrainModel called: %s", request.model_id)

        if not request.model_id:
            context.set_details("model_id is required")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return pb2.TrainModelResponse()

        try:
            hyperparams = json.loads(request.hyperparams_json) if request.hyperparams_json else {}
        except json.JSONDecodeError:
            context.set_details("Invalid hyperparams_json")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return pb2.TrainModelResponse()

        try:
            result = model_service.retrain(model_id=request.model_id, hyperparams=hyperparams)
        except Exception as exc:
            logger.error("gRPC Retrain Error: %s", exc)
            context.set_details(f"Retrain failed: {exc}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return pb2.TrainModelResponse()

        return pb2.TrainModelResponse(model_id=result["model_id"], model_class=result["model_class"])

    def Predict(self, request, context):
        """Run inference with a stored model."""
        logger.info("gRPC Predict called: model_id=%s", request.model_id)

        if request.n_features <= 0:
            context.set_details("n_features must be > 0")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return pb2.PredictResponse()

        data = np.array(request.data, dtype=float)
        if data.size % request.n_features != 0:
            context.set_details("data size is not divisible by n_features")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return pb2.PredictResponse()

        n_samples = data.size // request.n_features
        X = data.reshape(n_samples, request.n_features).tolist()

        try:
            raw_preds = model_service.predict(model_id=request.model_id, data=X)
            try:
                preds = [float(x) for x in raw_preds]
            except (TypeError, ValueError):
                logger.warning("Predictions are not numeric; returning 0.0 for gRPC demo")
                preds = [0.0] * len(raw_preds)
            return pb2.PredictResponse(predictions=preds)
        except Exception as exc:
            logger.error("gRPC Predict Error: %s", exc)
            context.set_details(f"Predict failed: {exc}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return pb2.PredictResponse()

    # ── Dataset RPCs ──────────────────────────────────────────────────────────

    def ListDatasets(self, request, context):
        """Return all registered datasets."""
        logger.info("gRPC ListDatasets called")
        datasets = [
            pb2.DatasetInfo(name=m.name, path=m.path, description=m.description)
            for m in dataset_service.list_datasets()
        ]
        return pb2.ListDatasetsResponse(datasets=datasets)

    def UploadDataset(self, request, context):
        """Upload and register a new dataset."""
        logger.info("gRPC UploadDataset called: filename=%s", request.filename)
        if not request.filename or not request.content:
            context.set_details("filename and content are required")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return pb2.UploadDatasetResponse()

        try:
            meta = dataset_service.upload_dataset_from_bytes(
                filename=request.filename,
                content=request.content,
            )
        except Exception as exc:
            logger.error("gRPC UploadDataset Error: %s", exc)
            context.set_details(f"Upload failed: {exc}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return pb2.UploadDatasetResponse()

        return pb2.UploadDatasetResponse(name=meta.name, path=meta.path)

    def DeleteDataset(self, request, context):
        """Remove a dataset from the registry."""
        logger.info("gRPC DeleteDataset called: name=%s", request.name)
        if not request.name:
            context.set_details("name is required")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return pb2.DeleteDatasetResponse()

        try:
            dataset_service.delete_dataset(request.name)
        except Exception as exc:
            logger.error("gRPC DeleteDataset Error: %s", exc)
            context.set_details(f"Delete failed: {exc}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return pb2.DeleteDatasetResponse()

        return pb2.DeleteDatasetResponse(detail=f"Dataset '{request.name}' removed from registry")


def serve(port: int = 50051) -> None:
    """Start the gRPC server and block until termination."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_MLServiceServicer_to_server(MLServiceServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    logger.info("Starting gRPC server on port %d", port)
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()



class MLServiceServicer(pb2_grpc.MLServiceServicer):
    """Implements all RPCs defined in *ml_service.proto*."""

    # ── Model RPCs ────────────────────────────────────────────────────────────

    def ListModelClasses(self, request, context):
        """Return all available model classes."""
        logger.info("gRPC ListModelClasses called")
        classes = []
        for name, cfg in ModelRegistry.all_classes().items():
            classes.append(
                pb2.ModelClassInfo(
                    name=name,
                    default_params_json=json.dumps(cfg["default_params"]),
                )
            )
        return pb2.ListModelClassesResponse(classes=classes)

    def TrainModel(self, request, context):
        """Train a new model."""
        logger.info("gRPC TrainModel called: %s (ds: %s)", request.model_class, request.dataset_name)

        if not ModelRegistry.has(request.model_class):
            context.set_details("Unknown model_class")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return pb2.TrainModelResponse()

        try:
            hyperparams = json.loads(request.hyperparams_json) if request.hyperparams_json else {}
        except json.JSONDecodeError:
            context.set_details("Invalid hyperparams_json")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return pb2.TrainModelResponse()

        feature_cols = (
            [x.strip() for x in request.feature_columns_csv.split(",") if x.strip()]
            if request.feature_columns_csv
            else None
        )
        target_col = request.target_column or "synthetic_target"

        try:
            result = model_service.train(
                model_class=request.model_class,
                dataset_name=request.dataset_name,
                target_column=target_col,
                hyperparams=hyperparams,
                feature_columns=feature_cols,
            )
        except Exception as exc:
            logger.error("gRPC Train Error: %s", exc)
            context.set_details(f"Train failed: {exc}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return pb2.TrainModelResponse()

        return pb2.TrainModelResponse(model_id=result["model_id"], model_class=result["model_class"])

    def RetrainModel(self, request, context):
        """Re-train an existing model."""
        logger.info("gRPC RetrainModel called: %s", request.model_id)

        if not request.model_id:
            context.set_details("model_id is required")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return pb2.TrainModelResponse()

        try:
            hyperparams = json.loads(request.hyperparams_json) if request.hyperparams_json else {}
        except json.JSONDecodeError:
            context.set_details("Invalid hyperparams_json")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return pb2.TrainModelResponse()

        try:
            result = model_service.retrain(model_id=request.model_id, hyperparams=hyperparams)
        except Exception as exc:
            logger.error("gRPC Retrain Error: %s", exc)
            context.set_details(f"Retrain failed: {exc}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return pb2.TrainModelResponse()

        return pb2.TrainModelResponse(model_id=result["model_id"], model_class=result["model_class"])

    def Predict(self, request, context):
        """Run inference with a stored model."""
        logger.info("gRPC Predict called: model_id=%s", request.model_id)

        if request.n_features <= 0:
            context.set_details("n_features must be > 0")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return pb2.PredictResponse()

        data = np.array(request.data, dtype=float)
        if data.size % request.n_features != 0:
            context.set_details("data size is not divisible by n_features")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return pb2.PredictResponse()

        n_samples = data.size // request.n_features
        X = data.reshape(n_samples, request.n_features).tolist()

        try:
            raw_preds = model_service.predict(model_id=request.model_id, data=X)
            try:
                preds = [float(x) for x in raw_preds]
            except (TypeError, ValueError):
                logger.warning("Predictions are not numeric; returning 0.0 for gRPC demo")
                preds = [0.0] * len(raw_preds)
            return pb2.PredictResponse(predictions=preds)
        except Exception as exc:
            logger.error("gRPC Predict Error: %s", exc)
            context.set_details(f"Predict failed: {exc}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return pb2.PredictResponse()

    # ── Dataset RPCs ──────────────────────────────────────────────────────────

    def ListDatasets(self, request, context):
        """Return all registered datasets."""
        logger.info("gRPC ListDatasets called")
        datasets = [
            pb2.DatasetInfo(name=m.name, path=m.path, description=m.description)
            for m in dataset_service.list_datasets()
        ]
        return pb2.ListDatasetsResponse(datasets=datasets)

    def UploadDataset(self, request, context):
        """Upload and register a new dataset."""
        logger.info("gRPC UploadDataset called: filename=%s", request.filename)
        if not request.filename or not request.content:
            context.set_details("filename and content are required")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return pb2.UploadDatasetResponse()

        fake_upload = _InMemoryUploadFile(filename=request.filename, content=request.content)
        try:
            meta = dataset_service.upload_dataset(fake_upload)  # type: ignore[arg-type]
        except Exception as exc:
            logger.error("gRPC UploadDataset Error: %s", exc)
            context.set_details(f"Upload failed: {exc}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return pb2.UploadDatasetResponse()

        return pb2.UploadDatasetResponse(name=meta.name, path=meta.path)

    def DeleteDataset(self, request, context):
        """Remove a dataset from the registry."""
        logger.info("gRPC DeleteDataset called: name=%s", request.name)
        if not request.name:
            context.set_details("name is required")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return pb2.DeleteDatasetResponse()

        try:
            dataset_service.delete_dataset(request.name)
        except Exception as exc:
            logger.error("gRPC DeleteDataset Error: %s", exc)
            context.set_details(f"Delete failed: {exc}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return pb2.DeleteDatasetResponse()

        return pb2.DeleteDatasetResponse(detail=f"Dataset '{request.name}' removed from registry")


def serve(port: int = 50051) -> None:
    """Start the gRPC server and block until termination."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_MLServiceServicer_to_server(MLServiceServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    logger.info("Starting gRPC server on port %d", port)
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
