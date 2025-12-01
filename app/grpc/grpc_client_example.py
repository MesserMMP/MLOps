import json
import grpc
import sys

# Если импорт падает, убедись что файл сгенерирован корректно
from app.grpc import ml_service_pb2 as pb2
from app.grpc import ml_service_pb2_grpc as pb2_grpc


def main():
    channel = grpc.insecure_channel("localhost:50051")
    stub = pb2_grpc.MLServiceStub(channel)

    print("--- 1. List Classes ---")
    try:
        resp_classes = stub.ListModelClasses(pb2.ListModelClassesRequest())
        for cls in resp_classes.classes:
            print(cls.name, cls.default_params_json)
    except grpc.RpcError as e:
        print(f"RPC Failed: {e.details()}")
        return

    print("\n--- 2. Train Model (SYNTHETIC MODE) ---")
    # Используем специальное имя для генерации данных
    dataset_name = "synthetic"

    train_req = pb2.TrainModelRequest(
        dataset_name=dataset_name,
        model_class="logreg",
        hyperparams_json=json.dumps({"max_iter": 200}),
        target_column="",  # Для синтетики не важно
        feature_columns_csv=""
    )

    try:
        train_resp = stub.TrainModel(train_req)
        print(f"Trained model: ID={train_resp.model_id}, Class={train_resp.model_class}")
    except grpc.RpcError as e:
        print(f"Train Failed: {e.details()}")
        return

    print("\n--- 3. Predict ---")
    data = [0.1, 0.2, 0.3, 0.4]  # Пример синтетических признаков
    pred_req = pb2.PredictRequest(
        model_id=train_resp.model_id,
        data=data,
        n_features=4,
    )
    try:
        pred_resp = stub.Predict(pred_req)
        print(f"Predictions: {pred_resp.predictions}")
    except grpc.RpcError as e:
        print(f"Predict Failed: {e.details()}")

    print("\n--- 4. Retrain ---")
    retrain_req = pb2.RetrainModelRequest(
        model_id=train_resp.model_id,
        hyperparams_json=json.dumps({"max_iter": 300}),
    )
    try:
        retrain_resp = stub.RetrainModel(retrain_req)
        print(f"Retrained model: ID={retrain_resp.model_id}")
    except grpc.RpcError as e:
        print(f"Retrain Failed: {e.details()}")


if __name__ == "__main__":
    main()
