from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# config.py lives in backend/app/, so two folders up is the repo root
# (the folder that holds artifacts/, results/, frontend/, backend/).
REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO_ROOT / "artifacts"
RESULTS = REPO_ROOT / "results"


class Settings(BaseSettings):
    # Values can be overridden in backend/.env (see .env.example).
    model_config = SettingsConfigDict(env_file=REPO_ROOT / "backend" / ".env", extra="ignore")

    cors_origins: str = "http://localhost:5500,http://127.0.0.1:5500"
    admin_key: str = ""          # used in a later step (admin pages)
    model_version: str = "v1-ensemble"

    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
