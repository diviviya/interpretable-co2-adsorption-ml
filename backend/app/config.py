from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# config.py lives in backend/app/, so two folders up is the repo root
# (the folder that holds artifacts/, results/, frontend/, backend/).
REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO_ROOT / "artifacts"
RESULTS = REPO_ROOT / "results"


class Settings(BaseSettings):
    # Values can be overridden in backend/.env. extra="ignore" means old
    # lines such as ADMIN_KEY=... are simply skipped, not an error.
    model_config = SettingsConfigDict(env_file=REPO_ROOT / "backend" / ".env", extra="ignore")

    cors_origins: str = "http://localhost:5500,http://127.0.0.1:5500"
    model_version: str = "v1-ensemble"

    # The admin account. The server creates (or updates) it at startup from
    # these two values, so there is no "sign up as admin" on the website.
    admin_username: str = ""
    admin_password: str = ""

    # How long a login lasts before the user has to log in again.
    session_days: int = 7

    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
