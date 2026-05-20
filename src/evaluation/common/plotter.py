"""Shared plotting utilities for evaluation outputs."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:  # pragma: no cover
    HAS_MATPLOTLIB = False


INTERVENTION_MARKERS = {
    "ward": (30, "#8b5cf6"),
    "area": (60, "#06b6d4"),
    "city": (120, "#3b82f6"),
}


def _add_intervention_markers(ax: Any, max_time: int) -> None:
    for _, (interval, color) in INTERVENTION_MARKERS.items():
        for tick in range(interval, max_time + 1, interval):
            ax.axvline(tick, color=color, linewidth=0.6, alpha=0.15)


def plot_scenario_series(
    results: list[dict[str, Any]],
    output_dir: Path,
) -> list[Path]:
    """Plot scenario-level mean time series across wards."""
    if not HAS_MATPLOTLIB:
        return []

    grouped: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for result in results:
        tick_records = result.get("tick_records", [])
        if tick_records:
            grouped[str(result["scenario_id"])].append(tick_records)

    created: list[Path] = []
    for scenario_id, series_list in grouped.items():
        if not series_list:
            continue

        max_len = min(len(series) for series in series_list)
        times = [int(series_list[0][i]["time"]) for i in range(max_len)]
        mean_speed = []
        mean_congestion = []
        mean_queue = []
        mean_throughput = []
        mean_ambulance_delay = []

        for idx in range(max_len):
            samples = [series[idx] for series in series_list]
            mean_speed.append(sum(float(s["avg_speed"]) for s in samples) / len(samples))
            mean_congestion.append(sum(float(s["congestion_score"]) for s in samples) / len(samples))
            mean_queue.append(sum(float(s["queue_length"]) for s in samples) / len(samples))
            mean_throughput.append(sum(float(s["throughput"]) for s in samples) / len(samples))
            mean_ambulance_delay.append(sum(float(s["ambulance_delay"]) for s in samples) / len(samples))

        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        axes = axes.flatten()
        plots = [
            ("Congestion vs Time", mean_congestion, "congestion_score"),
            ("Avg Speed vs Time", mean_speed, "avg_speed"),
            ("Queue vs Time", mean_queue, "queue_length"),
            ("Throughput vs Time", mean_throughput, "throughput"),
            ("Ambulance Progression", mean_ambulance_delay, "ambulance_delay"),
        ]
        for idx, (title, values, ylabel) in enumerate(plots):
            ax = axes[idx]
            ax.plot(times, values, linewidth=1.5)
            ax.set_title(title)
            ax.set_xlabel("tick")
            ax.set_ylabel(ylabel)
            _add_intervention_markers(ax, times[-1] if times else 0)
            ax.grid(alpha=0.2)

        axes[-1].axis("off")
        fig.suptitle(f"Scenario Summary: {scenario_id}")
        fig.tight_layout()
        out_path = output_dir / f"{scenario_id}_summary.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        created.append(out_path)

    return created
