"""Training entry point.

Runs identically from a shell and from a single Kaggle cell:

    python -m docbench.train.loop --config configs/smoke.yaml
    python -m docbench.train.loop --config configs/qwen2vl_lora.yaml --resume

Implemented in milestone 3. The surface is fixed now so the notebook launcher,
the config schema and CI can be built and tested against it first.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from docbench.hardware import detect_device
from docbench.train.config import RunConfig, load_run_config


def resolve_precision(config: RunConfig) -> str:
    """Settle the training dtype, preferring the device over the config file.

    An explicit bfloat16 on a pre-Ampere card is a configuration error, not a
    preference to be honoured quietly: it either raises deep inside the trainer
    or silently downcasts. Fail here, where the message can say why.
    """
    device = detect_device()
    if config.precision == "auto":
        return device.torch_dtype
    if config.precision == "bfloat16" and device.kind == "cuda" and not device.supports_bf16:
        raise ValueError(
            f"config requests bfloat16 but {device.name} (compute "
            f"{device.compute_capability}) does not support it. "
            "Kaggle's P100 and T4 are pre-Ampere; use float16 or 'auto'."
        )
    return config.precision


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fine-tune a VLM adapter for field extraction.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="continue from the last checkpoint")
    args = parser.parse_args(argv)

    config = load_run_config(args.config)
    device = detect_device()
    precision = resolve_precision(config)

    print(f"run            : {config.name}")
    print(f"base model     : {config.base_model}")
    print(f"device         : {device.describe()}")
    print(f"precision      : {precision}")
    print(f"grad scaling   : {device.needs_grad_scaling}")
    print(f"effective batch: {config.effective_batch_size}")
    print(f"max steps      : {config.max_steps}")
    print(f"resume         : {args.resume}")

    raise SystemExit("training loop lands in milestone 3; configuration validated")


if __name__ == "__main__":
    raise SystemExit(main())
