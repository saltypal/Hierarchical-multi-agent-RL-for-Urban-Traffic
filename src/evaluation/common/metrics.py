"""Canonical evaluation metrics and aggregation helpers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np


METRIC_FIELDS = [
    "time",
    "avg_speed",
    "congestion_score",
    "queue_length",
    "throughput",
    "trip_completion",
    "travel_time",
    "waiting_time",
    "ambulance_delay",
    "incident_delay",
    "reroute_count",
]


@dataclass
class SimulationMetrics:
    avg_speed: float = 0.0
    congestion_score: float = 0.0
    queue_length: float = 0.0
    throughput: float = 0.0
    trip_completion: float = 0.0
    travel_time: float = 0.0
    waiting_time: float = 0.0
    ambulance_delay: float = 0.0
    incident_delay: float = 0.0
    reroute_count: int = 0


def normalize_run_metrics(result: dict[str, Any]) -> dict[str, Any]:
    """Convert a runtime result into the evaluation metric schema."""
    tick_records = result.get("tick_records", [])
    final_tick = tick_records[-1] if tick_records else {}
    return {
        "time": int(result.get("total_ticks", 0)),
        "avg_speed": float(result.get("avg_speed", final_tick.get("avg_speed", 0.0))),
        "congestion_score": float(result.get("avg_congestion", final_tick.get("congestion_score", 0.0))),
        "queue_length": float(result.get("avg_queue", final_tick.get("queue_length", 0.0))),
        "throughput": float(result.get("total_arrived", final_tick.get("throughput", 0.0))),
        "trip_completion": float(result.get("total_arrived", final_tick.get("trip_completion", 0.0))),
        "travel_time": float(result.get("avg_travel_time", final_tick.get("travel_time", 0.0))),
        "waiting_time": float(result.get("avg_waiting_time", final_tick.get("waiting_time", 0.0))),
        "ambulance_delay": float(result.get("ambulance_delay", final_tick.get("ambulance_delay", 0.0))),
        "incident_delay": float(result.get("incident_delay", final_tick.get("incident_delay", 0.0))),
        "reroute_count": int(result.get("reroute_count", final_tick.get("reroute_count", 0))),
    }


def summarize_by_key(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    """Aggregate normalized run rows by a single grouping key."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)

    summaries: list[dict[str, Any]] = []
    for group_value, group_rows in sorted(grouped.items()):
        summary: dict[str, Any] = {key: group_value, "runs": len(group_rows)}
        for field in METRIC_FIELDS:
            values = [float(row.get(field, 0.0)) for row in group_rows]
            summary[field] = float(np.mean(values)) if values else 0.0
        summaries.append(summary)
    return summaries


def aggregate_full_sweep(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build full-sweep aggregate metrics and grouped summaries."""
    if not rows:
        return {
            "total_runs": 0,
            "overall": {},
            "by_scenario": [],
            "by_ward": [],
        }

    overall: dict[str, Any] = {"total_runs": len(rows)}
    for field in METRIC_FIELDS:
        overall[field] = float(np.mean([float(row.get(field, 0.0)) for row in rows]))

    return {
        "total_runs": len(rows),
        "overall": overall,
        "by_scenario": summarize_by_key(rows, "scenario_id"),
        "by_ward": summarize_by_key(rows, "ward_id"),
    }
