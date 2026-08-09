"""Structured logs to stderr; the durable record is the trace, not the log."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def setup(level: str = "INFO") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-5s %(name)s | %(message)s"))
    root = logging.getLogger("isc")
    root.setLevel(level)
    root.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup()
    return logging.getLogger(f"isc.{name}")
