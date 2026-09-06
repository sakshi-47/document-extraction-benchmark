"""Image degradation conditions.

Each transform is a named, seeded, reproducible function from image to image.
The experiment grid is the cross product of {condition} x {severity}, so every
condition must accept a severity in [0, 1] and be deterministic given a seed.

Milestone 2 implements: blur, glare, perspective skew, low light, occlusion,
JPEG recompression, and screen-replay (moire + rescan).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

import numpy as np

Image: TypeAlias = np.ndarray
Transform: TypeAlias = Callable[[Image, float, np.random.Generator], Image]

CONDITIONS: dict[str, Transform] = {}


def register(name: str) -> Callable[[Transform], Transform]:
    """Register a degradation condition under a stable name."""

    def deco(fn: Transform) -> Transform:
        if name in CONDITIONS:
            raise ValueError(f"condition already registered: {name}")
        CONDITIONS[name] = fn
        return fn

    return deco


@register("identity")
def identity(image: Image, severity: float, rng: np.random.Generator) -> Image:
    """The control condition. Every experiment reports against this baseline."""
    return image
