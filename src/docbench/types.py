"""Core domain types.

The vocabulary here is deliberately KYC-shaped: what matters is not "was the
string right" but "what would this error have done to a verification decision".
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Criticality(StrEnum):
    """How much a field matters to the downstream verification decision."""

    CRITICAL = "critical"  # name, date of birth, document number, expiry
    STANDARD = "standard"  # address, issuing authority
    COSMETIC = "cosmetic"  # formatting-only, no decision impact


class FieldKind(StrEnum):
    """Drives normalisation: a date is compared differently from a name."""

    TEXT = "text"
    NAME = "name"
    DATE = "date"
    NUMBER = "number"
    MONEY = "money"
    ALNUM_ID = "alnum_id"


class MatchStatus(StrEnum):
    """Outcome of comparing one predicted field against ground truth.

    WRONG is the dangerous class. A missing field fails loudly and routes to
    human review; a wrong-but-well-formed field passes validation silently and
    reaches a decision. The severity weights in `metrics.cfer` encode that.
    """

    EXACT = "exact"  # byte-identical
    NORMALIZED = "normalized"  # equal after canonicalisation
    WRONG = "wrong"  # predicted, present in gold, different
    MISSING = "missing"  # gold present, nothing predicted
    SPURIOUS = "spurious"  # predicted, gold absent (hallucination)
    TRUE_NEGATIVE = "true_negative"  # both absent - correct abstention

    @property
    def is_correct(self) -> bool:
        return self in (MatchStatus.EXACT, MatchStatus.NORMALIZED, MatchStatus.TRUE_NEGATIVE)


class FieldSpec(BaseModel):
    """Static description of a field in a document schema."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    kind: FieldKind = FieldKind.TEXT
    criticality: Criticality = Criticality.STANDARD


class FieldPrediction(BaseModel):
    """One extracted field, with whatever confidence the extractor reported."""

    model_config = ConfigDict(extra="forbid")

    name: str
    value: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class FieldOutcome(BaseModel):
    """Scored result for a single field. The unit of all downstream analysis."""

    model_config = ConfigDict(extra="forbid")

    name: str
    kind: FieldKind
    criticality: Criticality
    status: MatchStatus
    gold: tuple[str, ...] = ()
    predicted: str | None = None
    confidence: float | None = None
    well_formed: bool = True

    @property
    def is_silent_failure(self) -> bool:
        """Wrong, but well-formed - the class that survives format validation.

        Confidence is applied by the caller (it needs the configured threshold),
        so this property is the format-only half of the definition.
        """
        return self.status is MatchStatus.WRONG and self.well_formed
