"""Field-kind-aware canonicalisation.

Comparing extracted fields as raw text conflates two different things: a
genuine extraction error, and a formatting difference no downstream system
would care about. `01/02/1990` and `1990-02-01` are the same date; `SMITH` and
`Smith` are the same name. Canonicalising before comparison separates "the
model got it wrong" from "the model formatted it differently".

Every function here is pure and total: none raises on malformed input, and each
returns None when a value cannot be interpreted as the requested kind. That
distinction is load-bearing. An unparseable date is reported as not
well-formed, which is exactly what rules a case OUT of the silent-failure class
-- it fails validation loudly instead of passing silently.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date

from dateutil import parser as date_parser

from docbench.types import FieldKind

# Bound input length before any regex work. These values come from model
# output, which is untrusted; unbounded input into repeated regex passes is a
# denial-of-service shape (CWE-1333).
MAX_FIELD_CHARS = 512

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_NON_ALNUM = re.compile(r"[^A-Za-z0-9]")
_NUMERIC_STRIP = re.compile(r"[^\d.,\-]")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")


def _prepare(value: str) -> str:
    """Unicode-normalise, collapse whitespace, and bound the length."""
    text = unicodedata.normalize("NFKC", value)[:MAX_FIELD_CHARS]
    return _WS.sub(" ", text).strip()


def _canon_text(value: str) -> str:
    return _PUNCT.sub("", _prepare(value)).casefold().strip()


def _canon_name(value: str) -> str:
    # Names vary by ordering and separators far beyond their actual content.
    # Sorting tokens makes "Raut, Sakshi" equal to "Sakshi Raut".
    tokens = _PUNCT.sub(" ", _prepare(value)).casefold().split()
    return " ".join(sorted(tokens))


def _canon_alnum_id(value: str) -> str:
    return _NON_ALNUM.sub("", _prepare(value)).upper()


def _canon_date(value: str) -> str | None:
    """Parse a date to ISO-8601, or None if it cannot be interpreted as one.

    Order matters here, and getting it wrong is itself a silent failure.
    `dateutil.parse(..., dayfirst=True)` applies the day-first heuristic even to
    unambiguous ISO-8601 input: it reads "1990-02-01" as the 2nd of January.
    Strict ISO parsing therefore has to run first, and only genuinely ambiguous
    input falls through to the heuristic.
    """
    text = _prepare(value)
    if not text:
        return None

    # Unambiguous by construction; never subject it to the heuristic.
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError:
        pass

    try:
        # dayfirst=True: these corpora are non-US, so 01/02/1990 is 1 February.
        # A deliberate, documented assumption rather than an inherited default
        # -- ambiguous dates are themselves a source of the silent failures
        # this project measures, so the assumption has to be stated to be
        # arguable. Recorded in the README caveats.
        parsed = date_parser.parse(text, dayfirst=True, fuzzy=False)
    except (ValueError, OverflowError, TypeError):
        return None
    return parsed.date().isoformat()


def _canon_numeric(value: str) -> str | None:
    text = _THOUSANDS.sub("", _NUMERIC_STRIP.sub("", _prepare(value)))
    if not text or text in {"-", ".", "-.", ","}:
        return None
    try:
        parsed = float(text.replace(",", "."))
    except ValueError:
        return None
    return format(parsed, ".4f").rstrip("0").rstrip(".") or "0"


_CANONICALIZERS = {
    FieldKind.TEXT: _canon_text,
    FieldKind.NAME: _canon_name,
    FieldKind.ALNUM_ID: _canon_alnum_id,
    FieldKind.DATE: _canon_date,
    FieldKind.NUMBER: _canon_numeric,
    FieldKind.MONEY: _canon_numeric,
}


def canonicalize(value: str | None, kind: FieldKind) -> str | None:
    """Canonical form of `value` for its kind, or None if uninterpretable.

    Returning None for a value that cannot be parsed as the requested kind is
    the signal `well_formed` is derived from downstream.
    """
    if value is None:
        return None
    if not isinstance(value, str):  # extractor backends are untrusted input
        raise TypeError(f"expected str or None, got {type(value).__name__}")
    return _CANONICALIZERS[kind](value) or None
