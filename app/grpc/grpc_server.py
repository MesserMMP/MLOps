import json
from concurrent import futures

import grpc
import numpy as np

from app.core.logging_config import setup_logging
from app.models.registry import MODEL_CLASSES
from app.schemas.api import TrainRequest, RetrainRequest
from app.grpc import ml_service_pb2 as pb2
from app.grpc import ml_service_pb2_grpc as pb2_grpc

logger = setup_logging()


class MLServiceServicer(pb2_grpc.MLServiceServicer):
    def ListModelClasses(self, request, context):
        logger.info("gRPC ListModelClasses called")
        classes = []
        for name, cfg in MODEL_CLASSES.items():
            default_params_json = json.dumps(cfg["default_params"])
            classes.append(
                pb2.ModelClassInfo(
                    name=name,
                    default_params_json=default_params_json,
                )
            )
        return pb2.ListModelClassesResponse(classes=classes)

    def TrainModel(self, request, context):
        logger.info("gRPC TrainModel called: %s (ds: %s)", request.model_class, request.dataset_name)

        if request.model_class not in MODEL_CLASSES:
            context.set_details("Unknown model_class")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return pb2.TrainModelResponse()

        try:
            hyperparams = json.loads(request.hyperparams_json) if request.hyperparams_json else {}
        except json.JSONDecodeError:
            context.set_details("Invalid hyperparams_json")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return pb2.TrainModelResponse()

        feature_cols = None
        if request.feature_columns_csv:
            feature_cols = [x.strip() for x in request.feature_columns_csv.split(",") if x.strip()]

        # --- ЗАГЛУШКА ДЛЯ СИНТЕТИКИ ---
        target_col = request.target_column
        if request.dataset_name == "synthetic" and not target_col:
            target_col = "synthetic_target"  # Чтобы Pydantic не ругался
        # -----------------------------

        from app.api.routes import train_model as rest_train_model

        try:
            rest_req = TrainRequest(
                dataset_name=request.dataset_name,
                model_class=request.model_class,
                hyperparams=hyperparams,
                target_column=target_col,
                feature_columns=feature_cols
            )
            resp = rest_train_model(rest_req)
        except Exception as e:
            logger.error(f"gRPC Train Error: {e}")
            context.set_details(f"Train failed: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return pb2.TrainModelResponse()

        return pb2.TrainModelResponse(
            model_id=resp.model_id,
            model_class=resp.model_class,
        )

    def RetrainModel(self, request, context):
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

        from app.api.routes import retrain_model as rest_retrain_model

        try:
            rest_req = RetrainRequest(model_id=request.model_id, hyperparams=hyperparams)
            resp = rest_retrain_model(rest_req)
        except Exception as e:
            logger.error(f"gRPC Retrain Error: {e}")
            context.set_details(f"Retrain failed: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return pb2.TrainModelResponse()

        return pb2.TrainModelResponse(
            model_id=resp.model_id,
            model_class=resp.model_class,
        )

    def Predict(self, request, context):
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
        X = data.reshape(n_samples, request.n_features)

        from app.api.routes import predict as rest_predict
        from app.schemas.api import PredictRequest

        try:
            rest_req = PredictRequest(model_id=request.model_id, data=X.tolist())
            rest_resp = rest_predict(rest_req)

            # Кастим во float для совместимости с proto
            try:
                preds = [float(x) for x in rest_resp.predictions]
            except ValueError:
                logger.warning("Predictions are not numbers, returning 0.0 for gRPC demo")
                preds = [0.0] * len(rest_resp.predictions)

            return pb2.PredictResponse(predictions=preds)

        except Exception as e:
            logger.error(f"gRPC Predict Error: {e}")
            context.set_details(f"Predict failed: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return pb2.PredictResponse()


def serve(port: int = 50051) -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_MLServiceServicer_to_server(MLServiceServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    logger.info("Starting gRPC server on port %d", port)
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
