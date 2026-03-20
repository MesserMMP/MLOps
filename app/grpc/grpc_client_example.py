"""Example gRPC client demonstrating all available RPCs.

Usage (with both gRPC server and MinIO/ClearML running):

    make grpc-server          # in one terminal
    python -m app.grpc.grpc_client_example   # in another
"""

import json
import grpc

from app.grpc import ml_service_pb2 as pb2
from app.grpc import ml_service_pb2_grpc as pb2_grpc


def main():
    channel = grpc.insecure_channel("localhost:50051")
    stub = pb2_grpc.MLServiceStub(channel)

    # ── 1. List model classes ────────────────────────────────────────────────
    print("--- 1. List Model Classes ---")
    try:
        resp_classes = stub.ListModelClasses(pb2.ListModelClassesRequest())
        for cls in resp_classes.classes:
            print(cls.name, cls.default_params_json)
    except grpc.RpcError as e:
        print(f"RPC Failed: {e.details()}")
        return

    # ── 2. List datasets (initially empty) ──────────────────────────────────
    print("\n--- 2. List Datasets (before upload) ---")
    try:
        resp_ds = stub.ListDatasets(pb2.ListDatasetsRequest())
        print(f"Datasets: {[d.name for d in resp_ds.datasets]}")
    except grpc.RpcError as e:
        print(f"RPC Failed: {e.details()}")

    # ── 3. Upload a tiny synthetic CSV dataset via gRPC ──────────────────────
    print("\n--- 3. Upload Dataset ---")
    csv_content = b"feat_0,feat_1,feat_2,feat_3,label\n0.1,0.2,0.3,0.4,0\n0.5,0.6,0.7,0.8,1\n"
    try:
        resp_up = stub.UploadDataset(
            pb2.UploadDatasetRequest(filename="grpc_sample.csv", content=csv_content)
        )
        print(f"Uploaded: name={resp_up.name}, path={resp_up.path}")
    except grpc.RpcError as e:
        print(f"RPC Failed: {e.details()}")

    # ── 4. List datasets (now shows the uploaded file) ───────────────────────
    print("\n--- 4. List Datasets (after upload) ---")
    try:
        resp_ds = stub.ListDatasets(pb2.ListDatasetsRequest())
        for d in resp_ds.datasets:
            print(f"  {d.name}: {d.description}")
    except grpc.RpcError as e:
        print(f"RPC Failed: {e.details()}")

    # ── 5. Delete dataset ────────────────────────────────────────────────────
    print("\n--- 5. Delete Dataset ---")
    try:
        resp_del = stub.DeleteDataset(pb2.DeleteDatasetRequest(name="grpc_sample.csv"))
        print(resp_del.detail)
    except grpc.RpcError as e:
        print(f"RPC Failed: {e.details()}")

    # ── 6. Train a model using synthetic data ────────────────────────────────
    print("\n--- 6. Train Model (synthetic mode) ---")
    train_req = pb2.TrainModelRequest(
        dataset_name="synthetic",
        model_class="logreg",
        hyperparams_json=json.dumps({"max_iter": 200}),
        target_column="",
        feature_columns_csv="",
    )
    try:
        train_resp = stub.TrainModel(train_req)
        print(f"Trained model: ID={train_resp.model_id}, Class={train_resp.model_class}")
    except grpc.RpcError as e:
        print(f"Train Failed: {e.details()}")
        return

    # ── 7. Predict ───────────────────────────────────────────────────────────
    print("\n--- 7. Predict ---")
    pred_req = pb2.PredictRequest(
        model_id=train_resp.model_id,
        data=[0.1, 0.2, 0.3, 0.4],
        n_features=4,
    )
    try:
        pred_resp = stub.Predict(pred_req)
        print(f"Predictions: {pred_resp.predictions}")
    except grpc.RpcError as e:
        print(f"Predict Failed: {e.details()}")

    # ── 8. Retrain ───────────────────────────────────────────────────────────
    print("\n--- 8. Retrain ---")
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
