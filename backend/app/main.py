from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import REPO_ROOT, settings
from .routers import admin, auth, health, predict
from .services import auth_service, history_service, registry_service
from .services.model_service import model_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once at startup. The registry says which model version is active
    # (the first time, it creates the registry with the notebook model, v1),
    # and that bundle is loaded into memory so predictions don't reload it.
    history_service.init_db()
    auth_service.init_db()
    # Create/update the admin account from backend/.env and say so in the terminal.
    print("[startup]", auth_service.ensure_admin())
    model_service.load(registry_service.active_entry())
    yield


app = FastAPI(title="CO2 Uptake Predictor API", version="0.4.0", lifespan=lifespan)

# CORS: lets the frontend (served from port 5500) call this API (port 8000).
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_list,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(auth.router)
app.include_router(admin.router)

# The web pages (frontend/) are served by this same server, so the whole app
# lives at ONE address: http://127.0.0.1:8000/ on this laptop, your laptop's
# Wi-Fi IP for teammates, or a Cloudflare Tunnel link for anyone.
# This line must stay LAST: "/" matches everything, so the /api/... and
# /docs routes above have to be registered first to keep working.
app.mount("/", StaticFiles(directory=REPO_ROOT / "frontend", html=True), name="frontend")

