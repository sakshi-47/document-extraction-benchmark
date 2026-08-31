"""The contract every benchmarked system satisfies.

A system is anything that turns a document image into field predictions: an
OCR-and-rules pipeline, a frontier VLM behind an API, or a locally served
fine-tuned model. They are compared on identical inputs with identical scoring,
so the interface has to carry the measurement alongside the output -- a system
that returns predictions but no usage cannot appear on a cost axis.

Model output is untrusted input. Whatever comes back is parsed against the
document schema and never executed, never rendered as markup, and never used
to build a query.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from docfail.types import FieldPrediction, FieldSpec
from pydantic import BaseModel, ConfigDict

from docfit.bench.cost import Usage


class ExtractionResult(BaseModel):
    """One system's output for one document, with its measurement."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    doc_id: str
    predictions: tuple[FieldPrediction, ...]
    usage: Usage | None = None
    wall_ms: float
    cold: bool = False
    #: Set when the system failed outright, so failures are visible in results
    #: rather than silently becoming empty predictions.
    error: str | None = None


class System(Protocol):
    """A benchmarkable extraction system."""

    name: str

    @property
    def is_metered(self) -> bool:
        """True if this system reports token usage and can be placed on a cost axis.

        A self-hosted model returns False: its cost is amortised GPU time, not
        tokens, and is computed separately from throughput and instance price.
        """
        ...

    def extract(self, image_path: Path, schema: Sequence[FieldSpec]) -> ExtractionResult:
        """Extract every field in `schema` from one document.

        Implementations return a prediction for every schema field, using
        value=None for a genuine abstention. An abstention and a failure are
        different outcomes: the first is scored as MISSING, the second sets
        `error` and is excluded from accuracy with its exclusion reported.
        """
        ...
