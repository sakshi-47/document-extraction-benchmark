"""Resolving API tokens, without them ever reaching source or notebook output.

Order of resolution: Kaggle Secrets first (the platform this trains on), then
the process environment. Nothing else. There is no argument to pass a token in,
because a token passed in is a token that ends up in a notebook cell, and an
.ipynb keeps its cell outputs forever -- including in every commit made after.

Tokens are returned wrapped so they cannot be printed by accident. Unwrap only
at the point of use, never into a log line or an f-string.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


class MissingTokenError(RuntimeError):
    """Raised when a required token is not present in any supported source."""


@dataclass(frozen=True)
class Token:
    """A credential that does not render itself.

    __repr__ and __str__ are both redacted so an accidental print, an f-string,
    a traceback frame, or a rich table cannot leak the value.
    """

    _value: str
    source: str

    def __repr__(self) -> str:
        return f"<Token from {self.source}: redacted>"

    def __str__(self) -> str:
        return f"<Token from {self.source}: redacted>"

    def reveal(self) -> str:
        """Return the raw value. Call at the point of use only."""
        return self._value


def _from_kaggle(name: str) -> str | None:
    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError:
        return None
    try:
        value = UserSecretsClient().get_secret(name)
    except Exception:
        # A secret that is not attached to this notebook raises. That is a
        # normal miss, not an error -- fall through to the environment.
        return None
    return value or None


def get_token(name: str, required: bool = True) -> Token | None:
    """Resolve `name` from Kaggle Secrets, then the environment.

    On Kaggle, attach the secret under Add-ons -> Secrets. Locally, export it
    or put it in a gitignored dotenv file. Never pass it as an argument.
    """
    value = _from_kaggle(name)
    source = "kaggle-secrets"

    if not value:
        value = os.environ.get(name) or None
        source = "environment"

    if not value:
        if required:
            raise MissingTokenError(
                f"{name} not found. On Kaggle: Add-ons -> Secrets. "
                f"Locally: export {name} or add it to your gitignored dotenv file."
            )
        return None

    return Token(_value=value, source=source)
