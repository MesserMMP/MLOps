# MLOps ML Service

Учебный сервис для MLOps‑ДЗ, реализующий полный цикл работы с ML-моделями: от загрузки и версионирования данных до трекинга экспериментов и инференса.

## Основные возможности

- **API**: REST (FastAPI) и gRPC интерфейсы.
- **Data Management**: Загрузка CSV-датасетов с автоматическим версионированием через **DVC** и хранением в **MinIO** (S3).
- **Experiment Tracking**: Полная интеграция с **ClearML** — каждая тренировка создает эксперимент с логами, гиперпараметрами и артефактами.
- **Model Registry**: Обученные модели сохраняются в S3 (через ClearML) и доступны для инференса даже после перезапуска сервиса.
- **Dashboard**: Интерактивный UI на Streamlit (Status / Datasets / Training / Inference).

---

## Архитектура

- `app/` — Backend (FastAPI + gRPC).
  - `datasets/` — Логика работы с данными (DVC, реестр).
  - `core/clearml_wrapper.py` — Интеграция с ClearML SDK.
- `dashboard/` — Streamlit‑дашборд.
- `data/` — Локальное хранилище датасетов (под управлением DVC, игнорируется Git).
- `proto/ml_service.proto` — gRPC-контракт.

---

## Предварительная настройка

Для работы всех компонентов используется `docker-compose` (для MinIO и ClearML Server).

### 1. Установка зависимостей

```
pip install -r requirements.txt
```

### 2. Запуск инфраструктуры

Поднимаем MinIO и ClearML Server одной командой (через Makefile):

```
make minio-up
make clearml-up
```

*Подождите несколько минут, пока сервисы (особенно ClearML) инициализируются.*

- **MinIO Console**: http://localhost:9001 (user: `minioadmin`, pass: `minioadmin`)
- **ClearML Web UI**: http://localhost:8080

### 3. Настройка DVC (один раз)

Инициализируем DVC и настраиваем remote на локальный MinIO:

```
make dvc-init
```
*(Если бакет `mlops-datasets` не создался автоматически, создайте его вручную через консоль MinIO).*

### 4. Настройка ClearML Credentials (один раз)

1. Зайдите в ClearML Web UI (http://localhost:8080).
2. Перейдите в **Settings (шестеренка) -> Workspace -> Create new credentials**.
3. Скопируйте `Access Key` и `Secret Key`.
4. Откройте `Makefile` и вставьте ключи в переменные `CLEARML_API_ACCESS_KEY` и `CLEARML_API_SECRET_KEY`.

---

## Запуск сервисов

### Запуск Backend (API)

```
make backend
```
Команда автоматически прокинет все необходимые переменные окружения для доступа к MinIO и ClearML.

- Swagger UI: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### Запуск Dashboard

В новом терминале:

```
make dashboard
```
Откроется по адресу: http://localhost:8501.

---

## gRPC Интерфейс

В проекте уже сгенерированы необходимые файлы (`pb2` и `pb2_grpc`), поэтому можно сразу запускать сервер и клиент.

### 1. Запуск gRPC сервера

**Важно:** Запускать сервер нужно через `make`, чтобы передать ему ключи доступа к ClearML и MinIO.

```
make grpc-server
```

Сервер слушает порт `50051`.

### 2. Запуск примера клиента

Клиент демонстрирует полный цикл работы, используя **синтетические данные** (чтобы тест не зависел от загруженных файлов).

```
python -m app.grpc.grpc_client_example
```

Сценарий клиента:
1. Получение списка классов моделей.
2. Обучение модели (режим `synthetic`).
3. Инференс на синтетических данных.
4. Переобучение модели.

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
- Выберите датасет, алгоритм (`logreg` / `rf`) и укажите целевую колонку (`target_column`).
- Нажмите **Train**.
- **Что происходит:**
  - Инициализируется **Task** в ClearML.
  - Логируются гиперпараметры и метаданные датасета.
  - Модель обучается и сохраняется как **Artifact** в MinIO.
  - Задача завершается со статусом `Completed`.
  - Эксперимент можно посмотреть в ClearML UI.

### 3. Инференс (Model Registry)
- Перейдите на вкладку **Inference**.
- Выберите модель из списка (список подтягивается из ClearML, видны только успешно обученные модели).
- Введите данные JSON (например, `[[5.1, 3.5, 1.4, 0.2]]`).
- Нажмите **Predict**.
- **Что происходит:**
  - Сервис скачивает файл модели из MinIO (если нет в кэше).
  - Выполняется предсказание.

### 4. Переобучение (Retrain)
- На вкладке **Training** выберите существующую модель.
- Измените гиперпараметры.
- Нажмите **Retrain**.
- Создается **новый эксперимент** в ClearML, наследующий метаданные (датасет, колонки) от старой задачи.

---

## Полезные команды

- **Остановить инфраструктуру:**
  ```
  make minio-down
  make clearml-down
  ```

- **Проверить статус DVC:**
  ```
  make dvc-status
  ```

- **Сгенерировать gRPC код (при изменении .proto):**
  ```
  python -m grpc_tools.protoc -I proto --python_out=app/grpc --grpc_python_out=app/grpc proto/ml_service.proto && \
  sed -i '' 's/import ml_service_pb2/from . import ml_service_pb2/' app/grpc/ml_service_pb2_grpc.py
  ```