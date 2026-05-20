"""Structured ablation-based evaluation framework for the HMRL traffic stack.

Evaluates the system under multiple traffic scenarios comparing progressively
more intelligent configurations:

    Ward Level:   Baseline (no RL)  vs  Ward RL
    Area Level:   No intelligence   vs  Ward RL + Area GNN
    City Level:   No intelligence   vs  Full hierarchy (Ward + Area + City)

City-level evaluation uses stitched HSR_Layout + BTM_Layout networks.

Generates publication-quality comparison plots with intervention timing markers.

Usage:
    python evaluate.py --mode full
    python evaluate.py --mode quick
    python evaluate.py --mode ward --scenario normal
    python evaluate.py --mode area --scenario peak_congestion
    python evaluate.py --mode city --scenario chaos_mode
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.scenarios import list_scenarios
from src.runtime import run_simulation

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from tqdm.auto import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("evaluate")

# Representative wards: one from each area
EVAL_WARDS = ["ward_070", "ward_071", "ward_017", "ward_018", "ward_072"]
CITY_AREAS = ["HSR_Layout", "BTM_Layout"]
MAX_TICKS = 900  # 15 minutes — fast evaluation

METRIC_KEYS = [
    "avg_speed", "avg_congestion", "avg_queue", "total_arrived",
    "avg_travel_time", "avg_waiting_time", "ambulance_delay",
    "incident_delay", "reroute_count",
]

# Color scheme for publication plots
COLORS = {
    "baseline": "#94a3b8",
    "with_intelligence": "#6366f1",
}


@dataclass
class EvalConfig:
    """An evaluation configuration specifying which layers are active."""
    label: str
    use_rl: bool
    use_area: bool
    use_city: bool
    color: str


# ======================================================================
# Evaluation Configurations per Scope
# ======================================================================

WARD_CONFIGS = [
    EvalConfig("No RL (Baseline)", use_rl=False, use_area=False, use_city=False, color=COLORS["baseline"]),
    EvalConfig("Ward RL", use_rl=True, use_area=False, use_city=False, color=COLORS["with_intelligence"]),
]

AREA_CONFIGS = [
    EvalConfig("No Intelligence", use_rl=False, use_area=False, use_city=False, color=COLORS["baseline"]),
    EvalConfig("RL + Area GNN", use_rl=True, use_area=True, use_city=False, color=COLORS["with_intelligence"]),
]

CITY_CONFIGS = [
    EvalConfig("No Intelligence", use_rl=False, use_area=False, use_city=False, color=COLORS["baseline"]),
    EvalConfig("Full Hierarchy", use_rl=True, use_area=True, use_city=True, color=COLORS["with_intelligence"]),
]


# ======================================================================
# Core Evaluation Runner
# ======================================================================

def run_single_eval(
    scope: str,
    identifier: str,
    scenario_id: str,
    config: EvalConfig,
    algorithm: str = "ppo",
    area_ids: list[str] | None = None,
    max_ticks: int = MAX_TICKS,
) -> dict[str, Any]:
    """Run a single evaluation case and return metrics."""
    kwargs: dict[str, Any] = {
        "scope": scope,
        "identifier": identifier,
        "project_root": PROJECT_ROOT,
        "gui": False,
        "scenario_id": scenario_id,
        "max_ticks": max_ticks,
        "algorithm": algorithm,
        "dashboard": False,
        "use_rl": config.use_rl,
        "use_area": config.use_area,
        "use_city": config.use_city,
        "collect_tick_records": True,
        "persist_results": False,
    }
    if area_ids:
        kwargs["area_ids"] = area_ids

    result = run_simulation(**kwargs)
    return result


def extract_metrics(result: dict[str, Any]) -> dict[str, float]:
    """Extract standardized metrics from a simulation result."""
    return {
        "avg_speed": float(result.get("avg_speed", 0.0)),
        "congestion": float(result.get("avg_congestion", 0.0)),
        "queue_length": float(result.get("avg_queue", 0.0)),
        "throughput": float(result.get("total_arrived", 0.0)),
        "travel_time": float(result.get("avg_travel_time", 0.0)),
        "waiting_time": float(result.get("avg_waiting_time", 0.0)),
        "ambulance_delay": float(result.get("ambulance_delay", 0.0)),
        "incident_delay": float(result.get("incident_delay", 0.0)),
        "reroute_count": int(result.get("reroute_count", 0)),
    }


# ======================================================================
# Plotting
# ======================================================================

def plot_comparison_bars(
    results: dict[str, dict[str, dict[str, float]]],
    title: str,
    save_path: Path,
    config_labels: list[str],
    config_colors: list[str],
) -> None:
    """Plot grouped bar charts comparing configurations across scenarios.

    results: {scenario_id: {config_label: {metric: value}}}
    """
    if not HAS_MPL:
        return

    metrics_to_plot = [
        ("avg_speed", "Avg Speed (m/s)", True),
        ("congestion", "Congestion Score", False),
        ("queue_length", "Queue Length", False),
        ("throughput", "Throughput (vehicles)", True),
        ("travel_time", "Travel Time (s)", False),
        ("waiting_time", "Waiting Time (s)", False),
    ]

    scenarios = sorted(results.keys())
    n_scenarios = len(scenarios)
    n_configs = len(config_labels)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    bar_width = 0.35
    x = np.arange(n_scenarios)

    for idx, (metric, ylabel, higher_better) in enumerate(metrics_to_plot):
        ax = axes[idx]

        for ci, (label, color) in enumerate(zip(config_labels, config_colors)):
            values = []
            for scenario in scenarios:
                val = results.get(scenario, {}).get(label, {}).get(metric, 0.0)
                values.append(val)
            offset = (ci - (n_configs - 1) / 2) * bar_width
            bars = ax.bar(x + offset, values, bar_width * 0.9, label=label, color=color, alpha=0.85)

            # Add value labels
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f"{val:.1f}", ha="center", va="bottom", fontsize=7)

        ax.set_xlabel("Scenario")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([s[:12] for s in scenarios], rotation=30, ha="right", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    LOGGER.info("Comparison plot saved → %s", save_path)


def plot_timeseries_overlay(
    tick_data: dict[str, list[dict]],
    title: str,
    save_path: Path,
    config_colors: dict[str, str],
) -> None:
    """Plot time-series overlay of speed/congestion for multiple configs."""
    if not HAS_MPL or not tick_data:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    series_defs = [
        ("avg_speed", "Avg Speed (m/s)", axes[0, 0]),
        ("congestion_score", "Congestion Score", axes[0, 1]),
        ("queue_length", "Queue Length", axes[1, 0]),
        ("throughput", "Throughput", axes[1, 1]),
    ]

    for label, ticks in tick_data.items():
        times = [t["time"] for t in ticks]
        color = config_colors.get(label, "#333")

        for metric_key, ylabel, ax in series_defs:
            values = [t.get(metric_key, 0.0) for t in ticks]
            ax.plot(times, values, label=label, linewidth=1.2, color=color, alpha=0.8)

    # Add intervention markers
    for _, ylabel, ax in series_defs:
        max_t = max(max(t["time"] for t in ticks) for ticks in tick_data.values()) if tick_data else 0
        for tick in range(30, max_t + 1, 30):
            ax.axvline(tick, color="#8b5cf6", linewidth=0.4, alpha=0.1)
        for tick in range(60, max_t + 1, 60):
            ax.axvline(tick, color="#06b6d4", linewidth=0.5, alpha=0.15)
        for tick in range(120, max_t + 1, 120):
            ax.axvline(tick, color="#3b82f6", linewidth=0.6, alpha=0.2)

        ax.set_xlabel("Tick (seconds)")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    LOGGER.info("Time-series plot saved → %s", save_path)


def plot_improvement_summary(
    improvements: dict[str, dict[str, float]],
    title: str,
    save_path: Path,
) -> None:
    """Plot percentage improvement over baseline per scenario."""
    if not HAS_MPL or not improvements:
        return

    metrics = ["avg_speed", "congestion", "queue_length", "throughput", "waiting_time"]
    scenarios = sorted(improvements.keys())

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(scenarios))
    width = 0.12

    for i, metric in enumerate(metrics):
        values = [improvements[s].get(metric, 0.0) for s in scenarios]
        ax.bar(x + i * width, values, width, label=metric.replace("_", " ").title())

    ax.set_xlabel("Scenario")
    ax.set_ylabel("% Improvement over Baseline")
    ax.set_title(title)
    ax.set_xticks(x + width * len(metrics) / 2)
    ax.set_xticklabels(scenarios, rotation=30, ha="right")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(fontsize=7, ncol=3)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    LOGGER.info("Improvement plot saved → %s", save_path)


# ======================================================================
# Evaluation Sweep Functions
# ======================================================================

def evaluate_ward_level(scenarios: list[str], output_dir: Path, algorithm: str = "ppo") -> dict:
    """Ward-level: baseline vs RL for each ward across scenarios."""
    LOGGER.info("=== Ward-Level Evaluation ===")
    results: dict[str, dict[str, dict[str, float]]] = {}
    tick_data_per_scenario: dict[str, dict[str, list[dict]]] = {}
    all_raw: list[dict] = []

    total = len(scenarios) * len(EVAL_WARDS) * len(WARD_CONFIGS)
    pbar = tqdm(total=total, desc="Ward Evaluation")

    for scenario_id in scenarios:
        results[scenario_id] = {}
        tick_data_per_scenario[scenario_id] = {}

        for config in WARD_CONFIGS:
            aggregated: dict[str, list[float]] = defaultdict(list)
            all_ticks: list[dict] = []

            for ward_id in EVAL_WARDS:
                pbar.set_postfix(ward=ward_id, scenario=scenario_id[:10], config=config.label[:10])
                try:
                    result = run_single_eval("ward", ward_id, scenario_id, config, algorithm)
                    metrics = extract_metrics(result)
                    for k, v in metrics.items():
                        aggregated[k].append(v)
                    all_ticks.extend(result.get("tick_records", []))
                    all_raw.append({"scenario": scenario_id, "ward": ward_id, "config": config.label, **metrics})
                except Exception as e:
                    LOGGER.warning("Failed: %s/%s/%s — %s", scenario_id, ward_id, config.label, e)
                pbar.update(1)

            # Aggregate across wards
            avg_metrics = {k: float(np.mean(v)) if v else 0.0 for k, v in aggregated.items()}
            results[scenario_id][config.label] = avg_metrics
            tick_data_per_scenario[scenario_id][config.label] = all_ticks

    pbar.close()

    # Plot comparison bars
    plot_comparison_bars(
        results, "Ward-Level: Baseline vs RL Agent",
        output_dir / "ward_comparison.png",
        [c.label for c in WARD_CONFIGS],
        [c.color for c in WARD_CONFIGS],
    )

    # Plot time-series for first scenario
    first_scenario = scenarios[0]
    if first_scenario in tick_data_per_scenario:
        plot_timeseries_overlay(
            tick_data_per_scenario[first_scenario],
            f"Ward Time-Series — {first_scenario}",
            output_dir / f"ward_timeseries_{first_scenario}.png",
            {c.label: c.color for c in WARD_CONFIGS},
        )

    # Compute improvements
    improvements = _compute_improvements(results, WARD_CONFIGS)
    plot_improvement_summary(
        improvements,
        "Ward RL Improvement over Baseline (%)",
        output_dir / "ward_improvement.png",
    )

    return {"results": results, "raw": all_raw, "improvements": improvements}


def evaluate_area_level(scenarios: list[str], output_dir: Path, algorithm: str = "ppo") -> dict:
    """Area-level: no intelligence vs RL + Area GNN across scenarios."""
    LOGGER.info("=== Area-Level Evaluation ===")
    results: dict[str, dict[str, dict[str, float]]] = {}
    tick_data_per_scenario: dict[str, dict[str, list[dict]]] = {}
    all_raw: list[dict] = []

    total = len(scenarios) * len(EVAL_WARDS) * len(AREA_CONFIGS)
    pbar = tqdm(total=total, desc="Area Evaluation")

    for scenario_id in scenarios:
        results[scenario_id] = {}
        tick_data_per_scenario[scenario_id] = {}

        for config in AREA_CONFIGS:
            aggregated: dict[str, list[float]] = defaultdict(list)
            all_ticks: list[dict] = []

            for ward_id in EVAL_WARDS:
                pbar.set_postfix(ward=ward_id, scenario=scenario_id[:10], config=config.label[:10])
                try:
                    result = run_single_eval("ward", ward_id, scenario_id, config, algorithm)
                    metrics = extract_metrics(result)
                    for k, v in metrics.items():
                        aggregated[k].append(v)
                    all_ticks.extend(result.get("tick_records", []))
                    all_raw.append({"scenario": scenario_id, "ward": ward_id, "config": config.label, **metrics})
                except Exception as e:
                    LOGGER.warning("Failed: %s/%s/%s — %s", scenario_id, ward_id, config.label, e)
                pbar.update(1)

            avg_metrics = {k: float(np.mean(v)) if v else 0.0 for k, v in aggregated.items()}
            results[scenario_id][config.label] = avg_metrics
            tick_data_per_scenario[scenario_id][config.label] = all_ticks

    pbar.close()

    plot_comparison_bars(
        results, "Area-Level: No Intelligence vs RL + Area GNN",
        output_dir / "area_comparison.png",
        [c.label for c in AREA_CONFIGS],
        [c.color for c in AREA_CONFIGS],
    )

    first_scenario = scenarios[0]
    if first_scenario in tick_data_per_scenario:
        plot_timeseries_overlay(
            tick_data_per_scenario[first_scenario],
            f"Area Time-Series — {first_scenario}",
            output_dir / f"area_timeseries_{first_scenario}.png",
            {c.label: c.color for c in AREA_CONFIGS},
        )

    improvements = _compute_improvements(results, AREA_CONFIGS)
    plot_improvement_summary(
        improvements,
        "Area (RL + GNN) Improvement over Baseline (%)",
        output_dir / "area_improvement.png",
    )

    return {"results": results, "raw": all_raw, "improvements": improvements}


def evaluate_city_level(scenarios: list[str], output_dir: Path, algorithm: str = "ppo") -> dict:
    """City-level: no intelligence vs full hierarchy (HSR + BTM Layout)."""
    LOGGER.info("=== City-Level Evaluation (HSR + BTM Layout) ===")
    results: dict[str, dict[str, dict[str, float]]] = {}
    tick_data_per_scenario: dict[str, dict[str, list[dict]]] = {}
    all_raw: list[dict] = []

    total = len(scenarios) * len(CITY_CONFIGS)
    pbar = tqdm(total=total, desc="City Evaluation")

    for scenario_id in scenarios:
        results[scenario_id] = {}
        tick_data_per_scenario[scenario_id] = {}

        for config in CITY_CONFIGS:
            pbar.set_postfix(scenario=scenario_id[:10], config=config.label[:10])
            try:
                result = run_single_eval(
                    "city", "city_eval", scenario_id, config,
                    algorithm, area_ids=CITY_AREAS,
                )
                metrics = extract_metrics(result)
                results[scenario_id][config.label] = metrics
                tick_data_per_scenario[scenario_id][config.label] = result.get("tick_records", [])
                all_raw.append({"scenario": scenario_id, "config": config.label, **metrics})
            except Exception as e:
                LOGGER.warning("Failed: %s/%s — %s", scenario_id, config.label, e)
            pbar.update(1)

    pbar.close()

    plot_comparison_bars(
        results, "City-Level: No Intelligence vs Full Hierarchy",
        output_dir / "city_comparison.png",
        [c.label for c in CITY_CONFIGS],
        [c.color for c in CITY_CONFIGS],
    )

    first_scenario = scenarios[0]
    if first_scenario in tick_data_per_scenario:
        plot_timeseries_overlay(
            tick_data_per_scenario[first_scenario],
            f"City Time-Series — {first_scenario}",
            output_dir / f"city_timeseries_{first_scenario}.png",
            {c.label: c.color for c in CITY_CONFIGS},
        )

    improvements = _compute_improvements(results, CITY_CONFIGS)
    plot_improvement_summary(
        improvements,
        "Full Hierarchy Improvement over Baseline (%)",
        output_dir / "city_improvement.png",
    )

    return {"results": results, "raw": all_raw, "improvements": improvements}


def _compute_improvements(
    results: dict[str, dict[str, dict[str, float]]],
    configs: list[EvalConfig],
) -> dict[str, dict[str, float]]:
    """Compute percentage improvement of the last config over the first (baseline)."""
    improvements: dict[str, dict[str, float]] = {}
    baseline_label = configs[0].label
    intelligence_label = configs[-1].label

    for scenario_id, scenario_data in results.items():
        baseline = scenario_data.get(baseline_label, {})
        intelligence = scenario_data.get(intelligence_label, {})
        imp: dict[str, float] = {}

        for metric in ["avg_speed", "congestion", "queue_length", "throughput", "waiting_time"]:
            base_val = baseline.get(metric, 0.0)
            intel_val = intelligence.get(metric, 0.0)

            if abs(base_val) < 1e-8:
                imp[metric] = 0.0
            elif metric in ("congestion", "queue_length", "waiting_time"):
                # Lower is better
                imp[metric] = ((base_val - intel_val) / abs(base_val)) * 100
            else:
                # Higher is better
                imp[metric] = ((intel_val - base_val) / abs(base_val)) * 100

        improvements[scenario_id] = imp

    return improvements


# ======================================================================
# Report Generation
# ======================================================================

def generate_report(
    ward_results: dict | None,
    area_results: dict | None,
    city_results: dict | None,
    output_dir: Path,
) -> None:
    """Generate a text summary report of all evaluation results."""
    lines = [
        "=" * 70,
        "HMRL Hierarchical Traffic — Evaluation Report",
        "=" * 70,
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Evaluation wards: {EVAL_WARDS}",
        f"City areas: {CITY_AREAS}",
        f"Max ticks: {MAX_TICKS}",
        "",
    ]

    for level, data in [("Ward", ward_results), ("Area", area_results), ("City", city_results)]:
        if data is None:
            continue
        lines.append(f"\n{'='*40}")
        lines.append(f"{level}-Level Results")
        lines.append(f"{'='*40}")

        improvements = data.get("improvements", {})
        for scenario, imp in improvements.items():
            lines.append(f"\n  {scenario}:")
            for metric, pct in imp.items():
                direction = "↑" if pct > 0 else "↓" if pct < 0 else "→"
                lines.append(f"    {metric:<20} {direction} {pct:+.1f}%")

    report_path = output_dir / "evaluation_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("Report saved → %s", report_path)

    # Save raw results as JSON
    manifest = {
        "ward": ward_results.get("raw") if ward_results else None,
        "area": area_results.get("raw") if area_results else None,
        "city": city_results.get("raw") if city_results else None,
    }
    json_path = output_dir / "evaluation_results.json"
    json_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    LOGGER.info("JSON results saved → %s", json_path)


# ======================================================================
# CLI
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="HMRL Evaluation Framework")
    parser.add_argument(
        "--mode", default="full",
        choices=["full", "quick", "ward", "area", "city"],
        help="Evaluation mode",
    )
    parser.add_argument("--scenario", default=None, help="Specific scenario to evaluate")
    parser.add_argument("--algorithm", default="ppo", choices=["ppo", "dqn"])
    parser.add_argument("--max-ticks", type=int, default=MAX_TICKS)
    args = parser.parse_args()

    global MAX_TICKS
    MAX_TICKS = args.max_ticks

    output_dir = PROJECT_ROOT / "results" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = [args.scenario] if args.scenario else list_scenarios()

    print("=" * 60)
    print("HMRL — Structured Ablation Evaluation")
    print("=" * 60)
    print(f"Mode: {args.mode} | Scenarios: {scenarios}")
    print(f"Ticks: {MAX_TICKS} | Algorithm: {args.algorithm}")
    print()

    ward_results = None
    area_results = None
    city_results = None

    if args.mode in ("full", "ward"):
        ward_results = evaluate_ward_level(scenarios, output_dir, args.algorithm)

    if args.mode in ("full", "area"):
        area_results = evaluate_area_level(scenarios, output_dir, args.algorithm)

    if args.mode in ("full", "city"):
        city_results = evaluate_city_level(scenarios, output_dir, args.algorithm)

    if args.mode == "quick":
        quick_scenarios = scenarios[:2]
        ward_results = evaluate_ward_level(quick_scenarios, output_dir, args.algorithm)

    generate_report(ward_results, area_results, city_results, output_dir)

    print("\n" + "=" * 60)
    print("✅ Evaluation complete!")
    print(f"   Results: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
