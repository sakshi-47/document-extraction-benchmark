"""GPU capability detection, and the precision decision that follows from it.

Kaggle's free tier serves P100 (Pascal, compute 6.0) and T4 (Turing, 7.5).
Neither supports bfloat16 -- that arrives with Ampere, compute 8.0. Almost
every fine-tuning recipe published in the last few years assumes `bf16=True`
on an A100, and on this hardware that either raises or, worse, silently falls
back and trains at a precision the config does not describe.

So precision is derived from the device rather than copied from a tutorial, and
the choice is recorded in the run metadata. fp16 has a much narrower dynamic
range than bf16 and needs gradient scaling to avoid underflow; that is a real
constraint on free-tier training, not a footnote.

torch is an optional dependency. Everything here degrades to "unknown" without
it so the benchmark and metric layers stay importable on a laptop.
"""

from __future__ import annotations

from dataclasses import dataclass

BF16_MIN_COMPUTE_CAPABILITY = (8, 0)


@dataclass(frozen=True)
class Device:
    kind: str  # "cuda" | "mps" | "cpu" | "unknown"
    name: str
    compute_capability: tuple[int, int] | None
    total_memory_gb: float | None
    supports_bf16: bool

    @property
    def torch_dtype(self) -> str:
        """The dtype to train in on this device."""
        if self.kind != "cuda":
            return "float32"
        return "bfloat16" if self.supports_bf16 else "float16"

    @property
    def needs_grad_scaling(self) -> bool:
        """fp16 training underflows without a gradient scaler; bf16 does not."""
        return self.torch_dtype == "float16"

    def describe(self) -> str:
        memory = f"{self.total_memory_gb:.1f}GB" if self.total_memory_gb else "unknown memory"
        capability = (
            ".".join(str(x) for x in self.compute_capability)
            if self.compute_capability
            else "unknown"
        )
        return (
            f"{self.name} ({self.kind}, compute {capability}, {memory}) "
            f"-> {self.torch_dtype}" + (" with gradient scaling" if self.needs_grad_scaling else "")
        )


def detect_device() -> Device:
    """Inspect the current accelerator. Never raises; returns 'unknown' without torch."""
    try:
        import torch
    except ImportError:
        return Device("unknown", "torch not installed", None, None, False)

    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        capability = (props.major, props.minor)
        return Device(
            kind="cuda",
            name=props.name,
            compute_capability=capability,
            total_memory_gb=props.total_memory / 1024**3,
            supports_bf16=capability >= BF16_MIN_COMPUTE_CAPABILITY,
        )

    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return Device("mps", "Apple GPU", None, None, False)

    return Device("cpu", "cpu", None, None, False)
