"""Typed errors for the Jev schema store and controller (README taxonomy).

Every Jev failure carries a stable, machine-readable `code` that the
controller maps onto `JobEvent.error_code` and the CLI onto exit codes.
"""

from __future__ import annotations


class JevError(RuntimeError):
    """A Jev failure with a stable code (e.g. ``jev_schema_invalid``)."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class StoreError(JevError):
    """Schema-store failure.

    Codes: ``invalid_name``, ``not_found``, ``readonly`` (built-in),
    ``limit`` (MAX_USER_SCHEMAS), ``corrupt`` (file on disk unreadable).
    """
