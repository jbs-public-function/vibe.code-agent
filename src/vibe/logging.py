"""Structured logging configuration shared by controller, gateway, and admin (ADR-10, IMP-03.8/06.8).

Components log with ``logging.getLogger("vibe.<component>")`` and attach structured
context via ``extra={...}``, e.g. ``logger.info("decision", extra={"alias": ..., "status": 503})``.
`configure()` installs a stdout handler that renders one line per record:

    ts=2026-09-15T12:34:56Z level=info logger=vibe.gateway msg=decision alias=qwen status=503 elapsed_ms=12

Set ``VIBE_LOG_JSON=1`` for JSON lines instead of key=value. Level comes from
``VIBE_LOG_LEVEL`` (default ``info``).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time

# Standard LogRecord attributes (and well-known extras we never want to
# double-render). Everything else in record.__dict__ was passed via extra={}.
_STANDARD_ATTRS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "message",
    "module",
    "msecs",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "taskName",
    "thread",
    "threadName",
}


def _quote(value: str) -> str:
    """Shell-style quoting so a key=value line stays greppable."""
    if value and all(c.isalnum() or c in "._-/:@," for c in value):
        return value
    return json.dumps(value, ensure_ascii=False)


class _KeyValueFormatter(logging.Formatter):
    """``ts=… level=… logger=… msg=… k=v…`` — one line per record (IMP-06.8)."""

    def format(self, record: logging.LogRecord) -> str:
        pairs = [
            ("ts", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created))),
            ("level", record.levelname.lower()),
            ("logger", record.name),
            ("msg", record.getMessage()),
        ]
        for key, value in record.__dict__.items():
            if key in _STANDARD_ATTRS or key in {p[0] for p in pairs}:
                continue
            pairs.append((key, _quote(str(value))))
        if record.exc_info:
            pairs.append(("exc", _quote(self.formatException(record.exc_info))))
        return " ".join(f"{k}={v}" for k, v in pairs)


class _JsonFormatter(logging.Formatter):
    """One JSON object per line (``VIBE_LOG_JSON=1``)."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _STANDARD_ATTRS or key in entry:
                continue
            entry[key] = value
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str, ensure_ascii=False)


def configure(level: str | None = None) -> None:
    """Install the shared stdout handler on the root logger.

    Idempotent: repeated calls (e.g. one per component in `vibe serve`) replace
    the previous handler instead of duplicating output. ``level`` overrides
    ``VIBE_LOG_LEVEL``; when both are absent the default is ``info``.
    """
    level_name = (level or os.environ.get("VIBE_LOG_LEVEL") or "info").upper()
    use_json = os.environ.get("VIBE_LOG_JSON", "") == "1"

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter() if use_json else _KeyValueFormatter())

    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level_name)


def get_logger(name: str) -> logging.Logger:
    """Logger under the ``vibe.`` family; `configure()` must have run (or the
    root logger defaulted) for output to appear."""
    return logging.getLogger(name)
