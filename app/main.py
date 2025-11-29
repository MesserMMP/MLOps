from fastapi import FastAPI

from app.api.routes import router as api_router


app = FastAPI(
    title="ML MLOps Service",
    description="Учебный сервис для ML MLOps ДЗ",
    version="0.1.0",
)

app.include_router(api_router)
