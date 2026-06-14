from __future__ import annotations

from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ASTRA_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    env: str = "development"
    db_url: str = "sqlite+aiosqlite:///./data/astra.db"
    chroma_path: str = "./data/chroma_db"
    data_path: str = "./data"
    recipes_path: str = "./recipes"
    api_host: str = "0.0.0.0"
    api_port: int = 8200
    log_level: str = "INFO"
    autonomy_mode: Literal["guided", "supervised", "full_autonomy"] = "supervised"


settings = Settings()
