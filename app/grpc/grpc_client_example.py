import json
import grpc

from app.grpc import ml_service_pb2_grpc as pb2_grpc, ml_service_pb2 as pb2


def main():
    channel = grpc.insecure_channel("localhost:50051")
    stub = pb2_grpc.MLServiceStub(channel)

    # 1. Список классов моделей
    resp_classes = stub.ListModelClasses(pb2.ListModelClassesRequest())
    print("Available model classes:")
    for cls in resp_classes.classes:
        print(cls.name, cls.default_params_json)

    # 2. Обучение
    train_req = pb2.TrainModelRequest(
        dataset_name="demo_dataset",
        model_class="logreg",
        hyperparams_json=json.dumps({"max_iter": 200}),
    )
    train_resp = stub.TrainModel(train_req)
    print("Trained model:", train_resp.model_id, train_resp.model_class)

    # 3. Инференс
    data = [0.1, 0.2, 0.3, 0.4]  # один объект, 4 фичи
    pred_req = pb2.PredictRequest(
        model_id=train_resp.model_id,
        data=data,
        n_features=4,
    )
    pred_resp = stub.Predict(pred_req)
    print("Predictions:", pred_resp.predictions)


if __name__ == "__main__":
    main()
