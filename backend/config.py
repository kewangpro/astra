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

    # ── Inference providers ───────────────────────────────────────────────────
    # Lead agent (planning / pivots) — MLX on local MacBook
    lead_provider: Literal["ollama", "mlx", "mock"] = "mlx"
    lead_model: str = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
    ollama_base_url: str = "http://mac-mini.local:11434"

    # Code / error-fix agent — MLX on local MacBook
    code_provider: Literal["mlx", "ollama", "mock"] = "mlx"
    code_model: str = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"

    # ── Sandbox execution ─────────────────────────────────────────────────────
    # Remote host for training (empty = local subprocess)
    sandbox_host: str = ""
    # Base data directory on the remote sandbox host
    sandbox_data_path: str = "/tmp/astra"
    # Hostname/IP the training script uses to POST telemetry back to this MacBook
    telemetry_host: str = "macbook.local"


settings = Settings()
