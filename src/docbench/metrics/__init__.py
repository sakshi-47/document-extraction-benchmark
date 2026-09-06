"""Evaluation metrics for document field extraction."""

from docbench.metrics.cfer import (
    ErrorProfile,
    bootstrap_ci,
    critical_field_error_rate,
    error_profile,
    silent_failure_rate,
)
from docbench.metrics.fields import score_document, score_field
from docbench.metrics.normalize import canonicalize

__all__ = [
    "ErrorProfile",
    "bootstrap_ci",
    "canonicalize",
    "critical_field_error_rate",
    "error_profile",
    "score_document",
    "score_field",
    "silent_failure_rate",
]
