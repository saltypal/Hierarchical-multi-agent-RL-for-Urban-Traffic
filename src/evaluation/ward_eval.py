"""Ward-level evaluation helpers."""

from __future__ import annotations

from typing import Any

from .common.metrics import summarize_by_key


def build_ward_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate normalized evaluation rows by ward."""
    return summarize_by_key(rows, "ward_id")
