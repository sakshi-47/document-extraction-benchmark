"""Typed settings for the whole benchmark. No import-time side effects.

Credentials are resolved separately by `docbench.auth`, so a Settings instance
is always safe to print, log or dump in full. Nothing here reads a dotenv file
either: the CLI loads one explicitly at startup, which keeps this class
behaving identically under pytest, CI, Kaggle and Docker.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DOCBENCH_", extra="forbid")

    data_dir: Path = Path("./data")
    cache_dir: Path = Path("./cache")
    output_dir: Path = Path("./outputs")

    # --- Scoring -------------------------------------------------------------
    #: An extraction counts as a "silent failure" only if the model was confident.
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

    # --- Splits --------------------------------------------------------------
    #: Changing this reshuffles every split. Treat it as frozen once results exist.
    split_salt: str = "docbench-v1"
    train_ratio: float = Field(default=0.8, gt=0.0, lt=1.0)
    val_ratio: float = Field(default=0.1, ge=0.0, lt=1.0)

    # --- Guardrails ----------------------------------------------------------
    #: Refuse absurdly large images rather than exhausting memory.
    max_image_bytes: int = Field(default=25 * 1024 * 1024, gt=0)
    #: Bounded concurrency for extractor calls; paired with a rate limiter.
    max_concurrency: int = Field(default=4, ge=1, le=32)

    @field_validator("data_dir", "cache_dir", "output_dir")
    @classmethod
    def _expand(cls, v: Path) -> Path:
        return v.expanduser().resolve()

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.cache_dir, self.output_dir):
            d.mkdir(parents=True, exist_ok=True)


def load_settings() -> Settings:
    """Load and validate settings, failing fast with a readable message."""
    return Settings()
