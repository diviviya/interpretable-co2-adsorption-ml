from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .routers import health, predict
from .services.model_service import model_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once at startup: load the 49 MB model bundle into memory, so each
    # prediction doesn't have to reload it from disk.
    model_service.load()
    yield


app = FastAPI(title="CO2 Uptake Predictor API", version="0.2.0", lifespan=lifespan)

# CORS: lets the frontend (served from port 5500) call this API (port 8000).
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_list,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(predict.router)
