"""Run configuration, validated on load.

Every run is described entirely by a YAML file, so a result can be traced to
the exact configuration that produced it. Nothing is passed as a loose CLI
argument that would not survive into the run metadata.

`extra="forbid"` throughout is deliberate. A typo'd key in a training config --
`learing_rate` -- would otherwise be silently ignored, the default used, and
the run quietly not be the run the file describes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

# Refuse to parse absurd config files rather than exhausting memory on one.
MAX_CONFIG_BYTES = 256 * 1024


class LoraConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    r: int = Field(default=16, ge=1, le=256)
    alpha: int = Field(default=32, ge=1)
    dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    bias: Literal["none", "all", "lora_only"] = "none"


class CheckpointConfig(BaseModel):
    """Checkpointing is not optional on a platform with a session limit.

    /kaggle/working does not survive between sessions, so the Hub repo is the
    checkpoint store, not local disk. Adapters are tens of megabytes, which
    makes pushing every few hundred steps entirely affordable.
    """

    model_config = ConfigDict(extra="forbid")

    every_steps: int = Field(default=250, ge=1)
    hub_repo_id: str | None = None
    push_to_hub: bool = True
    keep_last: int = Field(default=3, ge=1)

    @model_validator(mode="after")
    def _hub_needs_repo(self) -> CheckpointConfig:
        if self.push_to_hub and not self.hub_repo_id:
            raise ValueError("push_to_hub is set but hub_repo_id is empty")
        return self


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    base_model: str
    seed: int = 42

    max_steps: int = Field(default=2000, ge=1)
    batch_size: int = Field(default=1, ge=1)
    grad_accum_steps: int = Field(default=8, ge=1)
    learning_rate: float = Field(default=1e-4, gt=0.0)
    warmup_ratio: float = Field(default=0.03, ge=0.0, lt=1.0)
    max_image_pixels: int = Field(default=768 * 768, gt=0)

    load_in_4bit: bool = True
    gradient_checkpointing: bool = True

    #: Left unset so it is derived from the detected device rather than copied
    #: from a recipe written for an A100. See docbench.hardware.
    precision: Literal["auto", "float16", "bfloat16", "float32"] = "auto"

    lora: LoraConfig = LoraConfig()
    checkpoint: CheckpointConfig = CheckpointConfig(hub_repo_id=None, push_to_hub=False)

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum_steps


def load_run_config(path: Path | str) -> RunConfig:
    """Load and validate a run config from YAML."""
    path = Path(path)
    size = path.stat().st_size
    if size > MAX_CONFIG_BYTES:
        raise ValueError(f"config file is implausibly large ({size} bytes); refusing")

    with path.open("rb") as fh:
        # safe_load only: full load constructs arbitrary Python objects from
        # the document, which is remote code execution on an untrusted file.
        raw: Any = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ValueError(f"config must be a YAML mapping, got {type(raw).__name__}")
    return RunConfig.model_validate(raw)
