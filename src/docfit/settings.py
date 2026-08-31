"""Typed settings. No import-time side effects, no credentials.

Credentials are resolved separately by `docfit.auth`, so a Settings instance is
always safe to print, log or dump in full.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DOCFIT_", extra="forbid")

    data_dir: Path = Path("./data")
    output_dir: Path = Path("./outputs")

    #: Changing this reshuffles every split. Treat it as frozen once results exist.
    split_salt: str = "docfit-v1"
    train_ratio: float = Field(default=0.8, gt=0.0, lt=1.0)
    val_ratio: float = Field(default=0.1, ge=0.0, lt=1.0)

    #: Silent-failure threshold, passed through to docfail's metrics.
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

    @field_validator("data_dir", "output_dir")
    @classmethod
    def _expand(cls, v: Path) -> Path:
        return v.expanduser().resolve()

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.output_dir):
            d.mkdir(parents=True, exist_ok=True)


def load_settings() -> Settings:
    return Settings()
