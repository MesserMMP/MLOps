# MLOps ML Service (Stage 1)

Учебный сервис для MLOps‑ДЗ. На этом этапе реализованы:

- REST‑API для обучения, переобучения, инференса и управления моделями.
- gRPC‑API с теми же возможностями и примером клиента.
- Работа с реальными CSV‑датасетами (train/test по признакам и целевой колонке).
- Интерактивный Streamlit‑дашборд (Status / Datasets / Training / Inference).
- Логирование всех ключевых действий.

## Архитектура

Основные компоненты:

- `app/` — backend (FastAPI + gRPC + работа с датасетами и моделями).
- `dashboard/` — Streamlit‑дашборд поверх REST‑эндоинтов.
- `proto/ml_service.proto` — описание gRPC‑контракта.

Реализация требований ДЗ (основная часть):

1. **Обучение моделей с разными гиперпараметрами, ≥2 классов**

   - REST: `POST /models/train`  
     Параметры:
     - `dataset_name`: имя загруженного датасета (файл в `data/`).
     - `model_class`: `logreg` или `rf`.
     - `hyperparams`: словарь гиперпараметров (перекрывает дефолты).
     - `target_column`: имя целевой колонки в CSV.
     - `feature_columns`: список колонок‑фич (или `null` для авто‑режима).

   - В `MODEL_CLASSES` зарегистрированы как минимум:
     - `logreg` → `sklearn.linear_model.LogisticRegression`
     - `rf` → `sklearn.ensemble.RandomForestClassifier`

2. **Список доступных классов моделей**

   - REST: `GET /models/classes`
   - gRPC: `ListModelClasses`

3. **Предсказание конкретной модели и хранение нескольких моделей**

   - `TRAINED_MODELS` хранит метаданные и объекты моделей в памяти:
     - `model_id`, `class`, `dataset_name`, `params`,
       `target_column`, `feature_columns`.
   - REST:
     - `GET /models/list` — список обученных моделей.
     - `POST /models/predict` — инференс по выбранной модели.
   - gRPC:
     - `Predict` — инференс по `model_id`.

4. **Переобучение и удаление моделей**

   - REST:
     - `POST /models/retrain` — переобучение выбранной модели на том же CSV,
       создаётся новая модель с новым `model_id`.
     - `DELETE /models/{model_id}` — удаление модели.
   - gRPC:
     - `RetrainModel` — RPC для переобучения.

5. **Эндпоинт статуса сервиса**

   - REST: `GET /health`
   - Дашборд: вкладка **Status**, которая его дергает.

6. **Список загруженных датасетов**

   - REST:
     - `GET /datasets` — список зарегистрированных датасетов.
     - `POST /datasets/upload` — загрузка `csv`/`json`, сохранение в `data/`,
       регистрация в in‑memory реестре.
   - Дашборд: вкладка **Datasets** (список + загрузка).

7. **Интерактивный дашборд**

   - Streamlit‑приложение c вкладками:
     - **Status** — проверка `/health`.
     - **Datasets** — просмотр и загрузка датасетов.
     - **Training** — выбор датасета и класса модели, задание:
       - `target_column`,
       - `feature_columns` (через строку с перечислением или auto),
       - `hyperparams` (JSON),
       затем:
       - `Train new model` — `POST /models/train`,
       - `Retrain selected model` — `POST /models/retrain`.
     - **Inference** — выбор модели и запуск `POST /models/predict`
       с произвольными строками признаков.

## Работа с CSV‑датасетами

Файлы сохраняются в директорию `data/` и регистрируются в in‑memory реестре.  
При обучении:

- CSV читается через `pandas.read_csv`.
- По `target_column` формируется `y`.
- По `feature_columns` (или автоматически) формируется `X`:
  - если `feature_columns == null`, используются все **числовые**
    колонки, кроме `target_column` и типичных индексов
    (`id`, `index`, `Unnamed: 0`, и т.п.).
- Так модель явно знает, где признаки, а где целевая переменная.

При переобучении используются те же `dataset_name`, `target_column`
и `feature_columns`, что были сохранены для исходной модели.

## Запуск backend

Установить зависимости (вариант):

```
pip install -r requirements.txt
```

Запуск FastAPI‑сервиса:

```
uvicorn app.main:app --reload
```

Основные URL:

- Swagger: http://localhost:8000/docs
- OpenAPI: http://localhost:8000/openapi.json
- Health: http://localhost:8000/health

## Запуск gRPC‑сервера и клиента

Сборка gRPC‑стабов (при изменении `proto/ml_service.proto`):

```
python -m grpc_tools.protoc \
  -I proto \
  --python_out=app/grpc \
  --grpc_python_out=app/grpc \
  proto/ml_service.proto
```

Запуск gRPC‑сервера:

```
python -m app.grpc.grpc_server
```

Тестовый клиент:

```
python -m app.grpc.grpc_client_example
```

Клиент демонстрирует:

- получение списка классов моделей,
- обучение модели,
- инференс,
- переобучение модели.

## Запуск дашборда

В отдельном терминале (при запущенном FastAPI‑сервисе):

```
streamlit run dashboard/app.py
```

По умолчанию откроется http://localhost:8501.

## Пример полного сценария проверки

1. Запустить backend (`uvicorn`) и дашборд (Streamlit).
2. На вкладке **Datasets**:
   - загрузить CSV (например, `Iris.csv` или Breast Cancer),
   - убедиться, что он появился в списке.
3. На вкладке **Training**:
   - выбрать датасет и класс модели (`logreg` или `rf`),
   - указать `target_column` (например, `Species` или `diagnosis`),
   - при необходимости оставить `feature_columns` пустым
     (будет auto = все числовые колонки без индексов),
   - при желании отредактировать `hyperparams` (JSON),
   - нажать **Train new model** и получить `model_id`.
4. На вкладке **Inference**:
   - выбрать обученную модель,
   - указать JSON‑массив строк признаков (например, одна строка
     из CSV без target),
   - нажать **Predict** и увидеть список предсказаний.
5. Вернуться на **Training**, выбрать существующую модель в блоке
   «retrain existing model», изменить гиперпараметры и нажать
   **Retrain selected model** — появится новый `model_id`.