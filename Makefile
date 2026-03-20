.PHONY: minio-up minio-down backend grpc-server dashboard \
        dvc-status dvc-init clearml-up clearml-down \
        docker-build docker-up docker-down \
        k8s-deploy k8s-delete

# MinIO (s3 backend)
minio-up:
	docker compose -f docker-compose.minio.yml up -d

minio-down:
	docker compose -f docker-compose.minio.yml down

# DVC
dvc-init:
	dvc init
	dvc remote add -d minio s3://mlops-datasets || true
	dvc remote modify minio endpointurl http://localhost:9000
	dvc remote modify minio access_key_id minioadmin
	dvc remote modify minio secret_access_key minioadmin
	dvc remote modify minio use_ssl false

dvc-status:
	dvc status

# ClearML server
clearml-up:
	docker compose -f docker-compose.clearml.yml up -d

clearml-down:
	docker compose -f docker-compose.clearml.yml down

# --- Docker (all services) ---

# Build all service images
docker-build:
	docker build -t mlops-backend:latest -f Dockerfile .
	docker build -t mlops-dashboard:latest -f Dockerfile.dashboard .

# Start all services (app + dashboard + minio) via docker-compose.yml
docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

# --- Minikube / Kubernetes ---
# Prerequisites:
#   minikube start --driver=docker
#   eval $(minikube docker-env)   # so kubectl can pull local images
#   make docker-build              # build images into minikube's Docker daemon

k8s-deploy:
	kubectl apply -f k8s/pvc.yaml
	kubectl apply -f k8s/secrets.yaml
	kubectl apply -f k8s/minio.yaml
	kubectl apply -f k8s/backend.yaml
	kubectl apply -f k8s/grpc.yaml
	kubectl apply -f k8s/dashboard.yaml

k8s-delete:
	kubectl delete -f k8s/dashboard.yaml --ignore-not-found
	kubectl delete -f k8s/grpc.yaml --ignore-not-found
	kubectl delete -f k8s/backend.yaml --ignore-not-found
	kubectl delete -f k8s/minio.yaml --ignore-not-found
	kubectl delete -f k8s/secrets.yaml --ignore-not-found
	kubectl delete -f k8s/pvc.yaml --ignore-not-found

# --- App (local dev, no Docker) ---

# Вставь сюда ключи из http://localhost:8080 -> Profile -> Settings -> Workspace
CLEARML_API_ACCESS_KEY = "YOUR_API_ACCESS_KEY"
CLEARML_API_SECRET_KEY = "YOUR_API_SECRET_KEY"

# Общие переменные для экспорта
define EXPORT_ENVS
	export AWS_ACCESS_KEY_ID=minioadmin && \
	export AWS_SECRET_ACCESS_KEY=minioadmin && \
	export AWS_ENDPOINT_URL=http://localhost:9000 && \
	export CLEARML_WEB_HOST="http://localhost:8080" && \
	export CLEARML_API_HOST="http://localhost:8008" && \
	export CLEARML_FILES_HOST="http://localhost:8081" && \
	export CLEARML_API_ACCESS_KEY=$(CLEARML_API_ACCESS_KEY) && \
	export CLEARML_API_SECRET_KEY=$(CLEARML_API_SECRET_KEY)
endef

# Запуск REST API
backend:
	$(EXPORT_ENVS) && uvicorn app.main:app --reload

# Запуск gRPC сервера
grpc-server:
	$(EXPORT_ENVS) && python -m app.grpc.grpc_server

dashboard:
	streamlit run dashboard/app.py