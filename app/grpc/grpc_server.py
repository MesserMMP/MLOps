import json
from concurrent import futures

import grpc
import numpy as np

from app.logging_config import setup_logging
from app.models.registry import MODEL_CLASSES
from app.models.storage import TRAINED_MODELS
from app.schemas.api import TrainRequest
from app.grpc import ml_service_pb2_grpc as pb2_grpc, ml_service_pb2 as pb2

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
        logger.info("gRPC TrainModel called: %s", request.model_class)

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

        # Переиспользуем TrainRequest логику
        train_req = TrainRequest(
            dataset_name=request.dataset_name,
            model_class=request.model_class,
            hyperparams=hyperparams,
        )

        # Сюда можно вынести общую функцию train_model_core(...)
        from app.api.routes import train_model as rest_train_model

        resp = rest_train_model(train_req)
        return pb2.TrainModelResponse(
            model_id=resp.model_id,
            model_class=resp.model_class,
        )

    def Predict(self, request, context):
        logger.info("gRPC Predict called: model_id=%s", request.model_id)

        if request.model_id not in TRAINED_MODELS:
            context.set_details("Model not found")
            context.set_code(grpc.StatusCode.NOT_FOUND)
            return pb2.PredictResponse()

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

        model = TRAINED_MODELS[request.model_id]["model"]
        preds = model.predict(X)

        return pb2.PredictResponse(predictions=[float(p) for p in preds])


def serve(port: int = 50051) -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_MLServiceServicer_to_server(MLServiceServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    logger.info("Starting gRPC server on port %d", port)
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
