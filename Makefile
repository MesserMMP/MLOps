.PHONY: minio-up minio-down backend dashboard dvc-status dvc-init

# Запуск MinIO (локальное s3 для DVC и дальше для моделей)
minio-up:
	docker compose -f docker-compose.minio.yml up -d

minio-down:
	docker compose -f docker-compose.minio.yml down

# Инициализация DVC и remote к MinIO (делается один раз после клона репо)
dvc-init:
	dvc init
	# добавить remote, если ещё не добавлен
	dvc remote add -d minio s3://mlops-datasets || true
	dvc remote modify minio endpointurl http://localhost:9000
	dvc remote modify minio access_key_id minioadmin
	dvc remote modify minio secret_access_key minioadmin
	dvc remote modify minio use_ssl false

# Проверка состояния DVC
dvc-status:
	dvc status

# Запуск backend (FastAPI)
backend:
	uvicorn app.main:app --reload

# Запуск Streamlit-дашборда
dashboard:
	streamlit run dashboard/app.py
