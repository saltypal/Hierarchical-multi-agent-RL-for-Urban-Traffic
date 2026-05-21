"""
scratch_eval_script.py
======================
Evaluate Baseline, DQN, and PPO using a SINGLE shared global model per algorithm.
All wards in EVAL_WARDS are run with the same loaded agent; metrics are averaged
across wards.  Saves metrics JSON + a grouped bar-chart PNG to results/.
"""

import sys
import json
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np

# ── project root on sys.path ────────────────────────────────────────────────
PROJECT_ROOT = Path(r"d:\Bunker\BaseCamp\Hierarchical-multi-agent-RL-for-Urban-Traffic")
sys.path.insert(0, str(PROJECT_ROOT))

from evaluate import run_single_eval, EvalConfig, extract_metrics   # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── knobs ─────────────────────────────────────────────────────────────────────
EVAL_WARDS  = ["ward_070", "ward_071", "ward_017", "ward_018", "ward_072"]
SCENARIO    = "normal"
MAX_TICKS   = 900
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── shared EvalConfig objects (one per mode, not per ward) ───────────────────
BASELINE_CFG = EvalConfig("No RL",  use_rl=False, use_area=False, use_city=False, color="#94a3b8")
DQN_CFG      = EvalConfig("DQN",    use_rl=True,  use_area=False, use_city=False, color="#f59e0b")
PPO_CFG      = EvalConfig("PPO",    use_rl=True,  use_area=False, use_city=False, color="#6366f1")

RUNS = [
    ("Baseline", BASELINE_CFG, "none"),   # algorithm arg doesn't matter when use_rl=False
    ("DQN",      DQN_CFG,      "dqn"),
    ("PPO",      PPO_CFG,      "ppo"),
]


# ── helper: run ONE config across all wards and average ──────────────────────
def eval_config_avg(label: str, cfg: EvalConfig, algorithm: str) -> dict:
    """
    Load the shared global model once (reused by runtime for all wards),
    run each ward, and return the averaged metric dict.
    """
    log.info("=== %s (algorithm=%s) ===", label, algorithm)
    aggregated: dict[str, list] = defaultdict(list)

    for ward_id in EVAL_WARDS:
        log.info("  ward %s …", ward_id)
        try:
            result  = run_single_eval(
                scope="ward",
                identifier=ward_id,
                scenario_id=SCENARIO,
                config=cfg,
                algorithm=algorithm,
                max_ticks=MAX_TICKS,
            )
            metrics = extract_metrics(result)
            for k, v in metrics.items():
                aggregated[k].append(v)
        except Exception as exc:
            log.warning("  FAILED for ward %s: %s", ward_id, exc)

    # average across wards; fallback to 0 if every ward failed
    avg: dict[str, float] = {}
    for k, values in aggregated.items():
        avg[k] = float(np.mean(values)) if values else 0.0

    log.info("  → %s", avg)
    return avg


# ── run all three configs ────────────────────────────────────────────────────
all_results: dict[str, dict] = {}
for run_label, run_cfg, run_algo in RUNS:
    all_results[run_label] = eval_config_avg(run_label, run_cfg, run_algo)

# ── save JSON ────────────────────────────────────────────────────────────────
json_path = RESULTS_DIR / "eval_metrics_comparison.json"
with open(json_path, "w") as f:
    json.dump(all_results, f, indent=2)
log.info("Metrics saved → %s", json_path)

# ── grouped bar chart ────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    METRIC_DEFS = [
        ("avg_speed",    "Avg Speed (m/s)",        True,  "#22d3ee"),
        ("congestion",   "Congestion Score",        True, "#f43f5e"),
        ("queue_length", "Avg Queue Length",        True, "#fb923c"),
        ("throughput",   "Throughput (vehicles)",   True,  "#4ade80"),
        ("travel_time",  "Avg Travel Time (s)",     True, "#a78bfa"),
        ("waiting_time", "Avg Waiting Time (s)",    True    , "#f472b6"),
        ("ambulance_delay", "Ambulance Delay (s)",    True, "#046582"),
        ("incident_delay", "Incident Delay (s)",    True, "#8b5c04"),
        ("reroute_count", "Reroute Count (s)",    True, "#7248f4"),
    ]

    runs       = list(all_results.keys())          # ["Baseline", "DQN", "PPO"]
    n_runs     = len(runs)
    n_metrics  = len(METRIC_DEFS)
    bar_w      = 0.22
    group_gap  = 1.0                               # distance between metric groups

    fig, ax = plt.subplots(figsize=(16, 6))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    run_colors = [cfg.color for _, cfg, _ in RUNS]   # per-run colours
    x_ticks       = []
    x_tick_labels = []

    for m_idx, (key, ylabel, higher_better, _bar_color) in enumerate(METRIC_DEFS):
        group_center = m_idx * group_gap
        x_ticks.append(group_center)
        x_tick_labels.append(ylabel)

        for r_idx, run_label in enumerate(runs):
            val = all_results[run_label].get(key, 0.0)
            # offset bars symmetrically around the group center
            offset = (r_idx - (n_runs - 1) / 2) * bar_w
            x_pos  = group_center + offset
            color  = run_colors[r_idx]
            bar    = ax.bar(x_pos, val, bar_w * 0.9,
                            color=color, alpha=0.88, zorder=3,
                            edgecolor="#0f172a", linewidth=0.6)
            # value label on top
            ax.text(x_pos, val * 1.01, f"{val:.1f}",
                    ha="center", va="bottom", fontsize=7,
                    color="white", fontweight="bold")

        # subtle shading every other metric group
        if m_idx % 2 == 0:
            ax.axvspan(group_center - group_gap * 0.5,
                       group_center + group_gap * 0.5,
                       color="white", alpha=0.03, zorder=0)

    # ── axes styling ──────────────────────────────────────────────────────────
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels, color="#cbd5e1", fontsize=9, rotation=15, ha="right")
    ax.tick_params(axis="y", colors="#94a3b8", labelsize=8)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.yaxis.grid(True, color="#334155", linewidth=0.5, linestyle="--", zorder=0)
    ax.set_axisbelow(True)

    # ── legend ────────────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color=run_colors[i], label=runs[i])
        for i in range(n_runs)
    ]
    ax.legend(handles=legend_patches,
              loc="upper right", framealpha=0.25,
              facecolor="#1e293b", edgecolor="#475569",
              labelcolor="white", fontsize=9)

    ax.set_title("RL vs Baseline — Avg across Wards\n"
                 f"({SCENARIO} scenario · {MAX_TICKS} ticks · wards: {', '.join(EVAL_WARDS)})",
                 color="white", fontsize=11, pad=12)

    fig.tight_layout()
    chart_path = RESULTS_DIR / "eval_comparison_chart.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info("Chart saved → %s", chart_path)

except ImportError:
    log.warning("matplotlib not available – skipping chart generation.")

print("\n✅  Done.")
print(f"   JSON  → {json_path}")
print(f"   Chart → {RESULTS_DIR / 'eval_comparison_chart.png'}")
