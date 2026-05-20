"""Area-model evaluation helpers."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def build_area_prediction_summary(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize area-pressure signals emitted during ward runs."""
    grouped: dict[str, list[float]] = defaultdict(list)
    for result in results:
        for record in result.get("ward_tick_records", []):
            grouped[record["ward_id"]].append(float(record.get("pressure", 0.0)))

    rows: list[dict[str, Any]] = []
    for ward_id, pressures in sorted(grouped.items()):
        rows.append({
            "ward_id": ward_id,
            "mean_predicted_pressure": sum(pressures) / len(pressures) if pressures else 0.0,
            "max_predicted_pressure": max(pressures) if pressures else 0.0,
        })
    return rows
