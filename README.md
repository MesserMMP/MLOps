# MLOps ML Service

Учебный сервис для MLOps‑ДЗ, реализующий полный цикл работы с ML-моделями: от загрузки и версионирования данных до трекинга экспериментов и инференса.

## Основные возможности

- **API**: REST (FastAPI) и gRPC интерфейсы.
- **Data Management**: Загрузка CSV-датасетов с автоматическим версионированием через **DVC** и хранением в **MinIO** (S3).
- **Experiment Tracking**: Полная интеграция с **ClearML** — каждая тренировка создает эксперимент с логами, гиперпараметрами и артефактами.
- **Model Registry**: Обученные модели сохраняются в S3 (через ClearML) и доступны для инференса даже после перезапуска сервиса.
- **Dashboard**: Интерактивный UI на Streamlit (Status / Datasets / Training / Inference).
- **Docker**: Весь стек поднимается одной командой (`make docker-up`).
- **Kubernetes**: Манифесты для деплоя в Minikube находятся в `k8s/`.

---

## Архитектура

```
app/
  api/routes.py           — тонкий слой REST-маршрутов
  grpc/grpc_server.py     — gRPC-сервер (ListDatasets, UploadDataset, DeleteDataset + модели)
  services/
    model_service.py      — бизнес-логика обучения / инференса
    dataset_service.py    — бизнес-логика датасетов
  models/registry.py      — ModelRegistry (класс со списком доступных алгоритмов)
  datasets/registry.py    — DatasetRegistry (класс-реестр загруженных датасетов)
  utils/
    clearml_wrapper.py    — обёртка над ClearML SDK
    logging_config.py     — настройка логгера
  core/                   — shim-файлы для обратной совместимости
dashboard/                — Streamlit-дашборд
data/                     — локальное хранилище датасетов (DVC-tracked, в .gitignore)
proto/ml_service.proto    — gRPC-контракт
k8s/                      — Kubernetes-манифесты (Minikube)
```

---

## Зависимости

Проект содержит два файла зависимостей:

- `requirements.txt` — верхнеуровневые прямые зависимости.
- `requirements-freeze.txt` — **полностью зафиксированный** список всех транзитивных зависимостей (результат `pip freeze`).

Для воспроизводимой установки используйте:

```bash
pip install -r requirements-freeze.txt
```

---

## Вариант 1: Запуск через Docker Compose (рекомендуется)

### Предварительная настройка

1. Запустите ClearML сервер (вне Docker Compose, в отдельном compose-файле):
   ```bash
   make clearml-up
   ```
2. В ClearML Web UI (http://localhost:8080) создайте credentials и скопируйте ключи.
3. Создайте файл `.env` в корне проекта:
   ```env
   CLEARML_API_ACCESS_KEY=<your_key>
   CLEARML_API_SECRET_KEY=<your_secret>
   ```

### Запуск всего стека

```bash
make docker-up
```

Эта команда соберёт образы и запустит: MinIO + Backend (FastAPI) + gRPC-сервер + Dashboard.

| Сервис       | URL                       |
|-------------|---------------------------|
| Swagger UI  | http://localhost:8000/docs |
| Health      | http://localhost:8000/health |
| Dashboard   | http://localhost:8501     |
| MinIO UI    | http://localhost:9001     |
| gRPC        | localhost:50051           |

Остановить:
```bash
make docker-down
```

---

## Вариант 2: Запуск в Minikube

### Предварительные требования

- `minikube` установлен
- `kubectl` установлен

### Шаги

```bash
# 1. Запустить minikube с Docker-драйвером
minikube start --driver=docker

# 2. Переключить Docker на minikube daemon
eval $(minikube docker-env)

# 3. Собрать образы внутри minikube
make docker-build

# 4. Заполнить секреты в k8s/secrets.yaml (замените REPLACE_ME)
# Или создайте через kubectl:
kubectl create secret generic mlops-secrets \
  --from-literal=minio-access-key=minioadmin \
  --from-literal=minio-secret-key=minioadmin \
  --from-literal=clearml-access-key=<YOUR_KEY> \
  --from-literal=clearml-secret-key=<YOUR_SECRET>

# 5. Применить манифесты
make k8s-deploy

# 6. Получить URL сервисов
minikube service mlops-backend --url
minikube service mlops-dashboard --url
minikube service mlops-minio --url
```

> **Примечание:** ClearML в этой конфигурации запускается отдельно на хост-машине через `make clearml-up`.
> Для деплоя ClearML тоже в Minikube — добавьте соответствующие манифесты в `k8s/`.

Удалить ресурсы из кластера:
```bash
make k8s-delete
```

---

## Вариант 3: Локальный запуск (без Docker)

### 1. Установка зависимостей

```bash
pip install -r requirements-freeze.txt
```

### 2. Запуск инфраструктуры

```bash
make minio-up
make clearml-up
```

- **MinIO Console**: http://localhost:9001 (user: `minioadmin`, pass: `minioadmin`)
- **ClearML Web UI**: http://localhost:8080

### 3. Настройка DVC (один раз)

```bash
make dvc-init
```
*(Если бакет `mlops-datasets` не создался автоматически, создайте его вручную через консоль MinIO).*

### 4. Настройка ClearML Credentials (один раз)

1. Зайдите в ClearML Web UI (http://localhost:8080).
2. Перейдите в **Settings → Workspace → Create new credentials**.
3. Скопируйте `Access Key` и `Secret Key`.
4. Откройте `Makefile` и вставьте ключи в переменные `CLEARML_API_ACCESS_KEY` и `CLEARML_API_SECRET_KEY`.

### 5. Запуск сервисов

```bash
make backend    # REST API на :8000
make grpc-server  # gRPC на :50051
make dashboard  # Streamlit на :8501
```

---

## gRPC Интерфейс

gRPC предоставляет полный набор операций — включая работу с датасетами.

### Доступные RPC

| RPC              | Описание                                      |
|-----------------|-----------------------------------------------|
| ListModelClasses | Список доступных алгоритмов                   |
| TrainModel      | Обучение новой модели                         |
| RetrainModel    | Переобучение существующей модели              |
| Predict         | Инференс                                      |
| ListDatasets    | Список загруженных датасетов                  |
| UploadDataset   | Загрузка датасета (bytes)                     |
| DeleteDataset   | Удаление датасета из реестра                  |

### Запуск gRPC сервера

```bash
make grpc-server
```

Сервер слушает порт `50051`.

### Запуск примера клиента

```bash
python -m app.grpc.grpc_client_example
```

Сценарий клиента:
1. Список классов моделей.
2. Список датасетов (пустой).
3. Загрузка синтетического CSV через gRPC.
4. Список датасетов (показывает загруженный).
5. Удаление датасета.
6. Обучение модели в режиме `synthetic`.
7. Инференс.
8. Переобучение.

### Регенерация pb2-файлов (при изменении .proto)

```bash
python -m grpc_tools.protoc \
  -I proto \
  --python_out=app/grpc \
  --grpc_python_out=app/grpc \
  proto/ml_service.proto

# Исправить импорт в сгенерированном файле:
sed -i 's/^import ml_service_pb2/from app.grpc import ml_service_pb2/' \
  app/grpc/ml_service_pb2_grpc.py
```

---

## Сценарий использования (REST / Dashboard)

### 1. Работа с данными (DVC + MinIO)
- Перейдите на вкладку **Datasets** в дашборде.
- Загрузите CSV-файл (например, `Iris.csv`).
- **Что происходит:**
  - Файл сохраняется локально.
  - Выполняется `dvc add` и `dvc push`.
  - Файл улетает в MinIO (бакет `mlops-datasets`).
  - DVC-файл (`.dvc`) можно закоммитить в Git.

### 2. Обучение модели (ClearML)
- Перейдите на вкладку **Training**.
- Выберите датасет, алгоритм (`logreg` / `rf`) и укажите `target_column`.
- Нажмите **Train**.
- **Что происходит:**
  - Инициализируется **Task** в ClearML.
  - Логируются гиперпараметры и метаданные датасета.
  - Модель обучается и сохраняется в MinIO через ClearML.

### 3. Инференс (Model Registry)
- Перейдите на вкладку **Inference**.
- Выберите модель из списка (из ClearML).
- Введите данные JSON: `[[5.1, 3.5, 1.4, 0.2]]`.
- Нажмите **Predict**.

### 4. Переобучение (Retrain)
- На вкладке **Training** выберите существующую модель.
- Измените гиперпараметры и нажмите **Retrain**.
- Создаётся **новый эксперимент** в ClearML, наследующий метаданные от старой задачи.

---

## Полезные команды

```bash
# Инфраструктура
make minio-up / minio-down
make clearml-up / clearml-down

# Docker
make docker-build   # только сборка образов
make docker-up      # сборка + запуск
make docker-down    # остановка

# Kubernetes
make k8s-deploy     # применить все манифесты
make k8s-delete     # удалить все ресурсы

# DVC
make dvc-status
```
