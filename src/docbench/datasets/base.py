"""Dataset adapter interface.

Adapters are deliberately thin: each yields Documents carrying an image path,
a field schema, and ground truth keyed by field name. Swapping CORD for
MIDV-500 must not touch the degradation, extraction or scoring layers.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from docbench.types import FieldSpec


class Document(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str
    image_path: Path
    #: Accepted surface variants per field. Always a sequence, never a bare str.
    gold: dict[str, tuple[str, ...]]


class DatasetAdapter(Protocol):
    """Contract every corpus adapter satisfies."""

    name: str

    @property
    def schema(self) -> Sequence[FieldSpec]:
        """Fields this corpus annotates, with their criticality."""
        ...

    def documents(self) -> Iterator[Document]:
        """Yield documents in a stable, reproducible order."""
        ...
