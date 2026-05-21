#!/usr/bin/env python
"""
fast_pipeline.py — Complete training + 3-level evaluation pipeline.
====================================================================

Trains PPO → DQN → STGCN, then evaluates at three hierarchy levels:
  Ward  : Baseline  vs  DQN  vs  PPO
  Area  : No Intelligence  vs  RL + Area GNN
  City  : No Intelligence  vs  Full Hierarchy (Ward RL + Area GNN + City)

Usage:
    python fast_pipeline.py
    python fast_pipeline.py --skip-rl --skip-gnn     # eval only
    python fast_pipeline.py --episodes 50 --eval-ticks 500
"""

import sys
import os
import time
import json
import logging
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np

# ── project setup ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("HMRL_MAP_DIR", "processed")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fast_pipeline")

# ── constants ────────────────────────────────────────────────────────────────
WARD_IDS = ["ward_070", "ward_071", "ward_072", "ward_017", "ward_018"]
SCENARIO_IDS = ["normal", "peak_congestion", "ambulance_emergency"]
GNN_AREAS = ["HSR_Layout", "BTM_Layout"]
EVAL_SCENARIO = "normal"

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── colours for bar charts ───────────────────────────────────────────────────
C_BASELINE  = "#9ca3af"
C_DQN       = "#f59e0b"
C_PPO       = "#6366f1"
C_NOINTEL   = "#9ca3af"
C_WITHSYS   = "#06b6d4"
C_FULLHIER  = "#10b981"


def parse_args():
    p = argparse.ArgumentParser(description="Fast training + 3-level evaluation pipeline")
    p.add_argument("--episodes", "-e", type=int, default=80)
    p.add_argument("--max-steps", "-s", type=int, default=360)
    p.add_argument("--gnn-epochs", type=int, default=100)
    p.add_argument("--eval-ticks", type=int, default=600)
    p.add_argument("--skip-rl", action="store_true")
    p.add_argument("--skip-gnn", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    return p.parse_args()


# ======================================================================
# STAGE 1 & 2: RL Training
# ======================================================================

def train_rl(algorithm: str, episodes: int, max_steps: int, collect_gnn: bool):
    from src.rl.train import train_global_agent

    log.info("=" * 60)
    log.info("  TRAINING %s  |  %d episodes  |  %d steps/ep", algorithm.upper(), episodes, max_steps)
    log.info("=" * 60)

    t0 = time.time()
    result = train_global_agent(
        ward_ids=WARD_IDS,
        scenario_ids=SCENARIO_IDS,
        project_root=PROJECT_ROOT,
        algorithm=algorithm,
        episodes=episodes,
        gui=False,
        collect_gnn_data=collect_gnn,
        max_simulation_steps=max_steps,
    )
    elapsed = time.time() - t0
    log.info("✅ %s done in %.1f min → %s", algorithm.upper(), elapsed / 60, result["model_path"])
    return result


# ======================================================================
# STAGE 3: STGCN Training
# ======================================================================

def train_stgcn(epochs: int):
    import torch
    from src.topology import Topology
    from src.controllers.area_controller import AreaForecaster

    gnn_dir = PROJECT_ROOT / "models" / "gnn"
    data_path = gnn_dir / "global_temporal_data.pt"

    if not data_path.exists():
        log.error("No GNN training data at %s — run RL training first!", data_path)
        return

    all_data = torch.load(data_path, weights_only=False)
    log.info("Loaded %d GNN training samples", len(all_data))

    combined_path = gnn_dir / "combined_training_data.pt"
    torch.save(all_data, combined_path)

    topology = Topology(PROJECT_ROOT)
    plot_dir = RESULTS_DIR / "training"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for area_id in GNN_AREAS:
        log.info("  TRAINING STGCN for %s  |  %d epochs", area_id, epochs)
        t0 = time.time()
        forecaster = AreaForecaster(area_id, topology, model_dir=gnn_dir, model_type="stgcn")
        result = forecaster.train_offline(combined_path, epochs=epochs, save_dir=gnn_dir)
        log.info("  ✅ STGCN %s done in %.1f min  |  MSE: %.6f",
                 area_id, (time.time() - t0) / 60, result["final_mse"])

        _save_loss_plot(result["losses"], f"STGCN Training Loss — {area_id}",
                        plot_dir / f"stgcn_{area_id}_loss.png")


# ======================================================================
# STAGE 4: Three-Level Evaluation
# ======================================================================

def run_full_evaluation(max_ticks: int):
    from evaluate import run_single_eval, EvalConfig, extract_metrics

    all_level_results = {}

    # ── Ward Level: Baseline vs DQN vs PPO ────────────────────────────────
    log.info("=" * 60)
    log.info("  WARD-LEVEL EVALUATION  |  Baseline vs DQN vs PPO")
    log.info("=" * 60)

    ward_configs = [
        ("Baseline", EvalConfig("No RL", use_rl=False, use_area=False, use_city=False, color=C_BASELINE), "dqn"),
        ("DQN",      EvalConfig("DQN",   use_rl=True,  use_area=False, use_city=False, color=C_DQN),      "dqn"),
        ("PPO",      EvalConfig("PPO",   use_rl=True,  use_area=False, use_city=False, color=C_PPO),      "ppo"),
    ]

    ward_results = _eval_ward_level(ward_configs, max_ticks)
    all_level_results["ward"] = ward_results

    # ── Area Level: No Intelligence vs RL + Area GNN ──────────────────────
    log.info("=" * 60)
    log.info("  AREA-LEVEL EVALUATION  |  No Intelligence vs RL + Area GNN")
    log.info("=" * 60)

    area_configs = [
        ("No Intelligence", EvalConfig("No Intelligence", use_rl=False, use_area=False, use_city=False, color=C_NOINTEL), "dqn"),
        ("RL + Area GNN",   EvalConfig("RL + Area GNN",   use_rl=True,  use_area=True,  use_city=False, color=C_WITHSYS), "dqn"),
    ]

    area_results = _eval_ward_level(area_configs, max_ticks)
    all_level_results["area"] = area_results

    # ── City Level: No Intelligence vs Full Hierarchy ─────────────────────
    log.info("=" * 60)
    log.info("  CITY-LEVEL EVALUATION  |  No Intelligence vs Full Hierarchy")
    log.info("=" * 60)

    city_configs = [
        ("No Intelligence", EvalConfig("No Intelligence", use_rl=False, use_area=False, use_city=False, color=C_NOINTEL), "dqn"),
        ("Full Hierarchy",  EvalConfig("Full Hierarchy",  use_rl=True,  use_area=True,  use_city=True,  color=C_FULLHIER), "dqn"),
    ]

    city_results = _eval_city_level(city_configs, max_ticks)
    all_level_results["city"] = city_results

    # ── Save everything ───────────────────────────────────────────────────
    json_path = RESULTS_DIR / "full_evaluation_results.json"
    with open(json_path, "w") as f:
        json.dump(all_level_results, f, indent=2)
    log.info("All results saved → %s", json_path)

    # ── Generate all plots ────────────────────────────────────────────────
    _plot_ward_level(ward_results, ward_configs, max_ticks)
    _plot_area_level(area_results, area_configs, max_ticks)
    _plot_city_level(city_results, city_configs, max_ticks)
    _plot_hierarchy_comparison(all_level_results)
    _plot_all_individual_metrics(ward_results, ward_configs, "ward")
    _plot_all_individual_metrics(area_results, area_configs, "area")
    _plot_all_individual_metrics(city_results, city_configs, "city")

    return all_level_results


def _eval_ward_level(configs, max_ticks):
    from evaluate import run_single_eval, extract_metrics
    results = {}
    for label, cfg, algo in configs:
        log.info("  --- %s ---", label)
        aggregated: dict[str, list] = defaultdict(list)
        for ward_id in WARD_IDS:
            log.info("    %s → %s", label, ward_id)
            try:
                result = run_single_eval(
                    scope="ward", identifier=ward_id, scenario_id=EVAL_SCENARIO,
                    config=cfg, algorithm=algo, max_ticks=max_ticks,
                )
                metrics = extract_metrics(result)
                for k, v in metrics.items():
                    aggregated[k].append(v)
            except Exception as exc:
                log.warning("    FAILED %s/%s: %s", label, ward_id, exc)
        avg = {k: float(np.mean(v)) if v else 0.0 for k, v in aggregated.items()}
        results[label] = avg
        log.info("  %s → speed=%.2f  cong=%.3f  queue=%.1f  thru=%.0f",
                 label, avg.get("avg_speed", 0), avg.get("congestion", 0),
                 avg.get("queue_length", 0), avg.get("throughput", 0))
    return results


def _eval_city_level(configs, max_ticks):
    from evaluate import run_single_eval, extract_metrics
    import traceback
    results = {}
    for label, cfg, algo in configs:
        log.info("  --- %s ---", label)
        try:
            result = run_single_eval(
                scope="city", identifier="city_eval", scenario_id=EVAL_SCENARIO,
                config=cfg, algorithm=algo, area_ids=GNN_AREAS, max_ticks=max_ticks,
            )
            metrics = extract_metrics(result)
            results[label] = metrics
            log.info("  %s → speed=%.2f  cong=%.3f  queue=%.1f  thru=%.0f",
                     label, metrics.get("avg_speed", 0), metrics.get("congestion", 0),
                     metrics.get("queue_length", 0), metrics.get("throughput", 0))
        except Exception as exc:
            log.error("  FAILED %s: %s", label, exc)
            log.error(traceback.format_exc())
            results[label] = {}
    return results


# ======================================================================
# Plotting — Clean white matplotlib style
# ======================================================================

def _get_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("default")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })
    return plt


def _save_loss_plot(losses, title, save_path):
    try:
        plt = _get_plt()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(losses, linewidth=1.5, color="#6366f1")
        ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
        ax.set_title(title)
        fig.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150); plt.close(fig)
    except Exception:
        pass


METRIC_DEFS = [
    ("avg_speed",       "Avg Speed (m/s)",              True),
    ("congestion",      "Congestion Score",              False),
    ("queue_length",    "Avg Queue Length",               False),
    ("throughput",      "Throughput (vehicles)",          True),
    ("waiting_time",    "Avg Waiting Time (s)",           False),
    ("ambulance_delay", "Ambulance Delay (cum.)",         False),
]


def _grouped_bar_chart(results, configs, title, save_path):
    """Grouped bar chart for any set of configs — clean white style."""
    try:
        plt = _get_plt()
        import matplotlib.patches as mpatches
    except ImportError:
        return

    runs = list(results.keys())
    run_colors = [c.color for _, c, _ in configs]
    n_runs = len(runs)
    bar_w = max(0.15, 0.55 / n_runs)

    fig, ax = plt.subplots(figsize=(14, 5.5))

    x_positions = np.arange(len(METRIC_DEFS))

    for r_idx, run_label in enumerate(runs):
        vals = [results[run_label].get(key, 0.0) for key, _, _ in METRIC_DEFS]
        offset = (r_idx - (n_runs - 1) / 2) * bar_w
        bars = ax.bar(x_positions + offset, vals, bar_w * 0.88,
                      label=run_label, color=run_colors[r_idx], alpha=0.85,
                      edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(abs(bar.get_height()) * 0.02, 0.1),
                    f"{val:.1f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_xticks(x_positions)
    ax.set_xticklabels([ylabel.replace(" (", "\n(") for _, ylabel, _ in METRIC_DEFS], fontsize=8.5)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Chart saved → %s", save_path)


def _improvement_chart(results, baseline_label, title, save_path):
    """% improvement of each config over the baseline — white style."""
    try:
        plt = _get_plt()
    except ImportError:
        return

    baseline = results.get(baseline_label, {})
    if not baseline:
        return

    others = {k: v for k, v in results.items() if k != baseline_label}
    if not others:
        return

    metrics = ["avg_speed", "congestion", "queue_length", "throughput", "waiting_time"]
    lower_better = {"congestion", "queue_length", "waiting_time"}
    colors_cycle = [C_DQN, C_PPO, C_WITHSYS, C_FULLHIER]

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(metrics))
    bar_w = max(0.15, 0.55 / len(others))

    for i, (label, rval_dict) in enumerate(others.items()):
        improvements = []
        for m in metrics:
            bval = baseline.get(m, 0.0)
            rval = rval_dict.get(m, 0.0)
            if abs(bval) < 1e-8:
                improvements.append(0.0)
            elif m in lower_better:
                improvements.append(((bval - rval) / abs(bval)) * 100)
            else:
                improvements.append(((rval - bval) / abs(bval)) * 100)

        offset = (i - (len(others) - 1) / 2) * bar_w
        color = colors_cycle[i % len(colors_cycle)]
        bars = ax.bar(x + offset, improvements, bar_w * 0.85, label=label,
                      color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, improvements):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.3 if val >= 0 else -1.5),
                    f"{val:+.1f}%", ha="center",
                    va="bottom" if val >= 0 else "top",
                    fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics], fontsize=9)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("% Improvement over Baseline", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.legend(framealpha=0.9, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Chart saved → %s", save_path)


def _plot_all_individual_metrics(results, configs, level_name):
    """One bar chart per metric — white style."""
    try:
        plt = _get_plt()
    except ImportError:
        return

    plot_dir = RESULTS_DIR / f"{level_name}_individual_metrics"
    plot_dir.mkdir(parents=True, exist_ok=True)

    runs = list(results.keys())
    run_colors = [c.color for _, c, _ in configs]

    for key, title, higher_better in METRIC_DEFS:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        vals = [results[r].get(key, 0.0) for r in runs]
        bars = ax.bar(runs, vals, color=run_colors, alpha=0.85,
                      edgecolor="white", linewidth=0.8, width=0.5)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    f"{val:.2f}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")

        # Highlight best
        if vals and any(v != 0 for v in vals):
            best_idx = int(np.argmax(vals)) if higher_better else int(np.argmin(vals))
            bars[best_idx].set_edgecolor("#22d3ee")
            bars[best_idx].set_linewidth(2.5)

        ax.set_ylabel(title, fontsize=10)
        ax.set_title(f"{level_name.upper()} Level — {title}", fontsize=11, fontweight="bold", pad=8)
        ax.spines[["top", "right"]].set_visible(False)

        fig.tight_layout()
        fig.savefig(plot_dir / f"{key}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    log.info("%s individual metric plots → %s", level_name.upper(), plot_dir)


# ── Level-specific wrappers ──────────────────────────────────────────────────

def _plot_ward_level(results, configs, max_ticks):
    out = RESULTS_DIR / "ward_level"
    out.mkdir(parents=True, exist_ok=True)
    _grouped_bar_chart(
        results, configs,
        f"Ward Level — Baseline vs DQN vs PPO\n({EVAL_SCENARIO} · {max_ticks} ticks · {len(WARD_IDS)} wards avg)",
        out / "ward_comparison.png",
    )
    _improvement_chart(
        results, "Baseline",
        "Ward Level — RL Improvement over Baseline (%)",
        out / "ward_improvement.png",
    )


def _plot_area_level(results, configs, max_ticks):
    out = RESULTS_DIR / "area_level"
    out.mkdir(parents=True, exist_ok=True)
    _grouped_bar_chart(
        results, configs,
        f"Area Level — No Intelligence vs RL + Area GNN\n({EVAL_SCENARIO} · {max_ticks} ticks · {len(WARD_IDS)} wards avg)",
        out / "area_comparison.png",
    )
    _improvement_chart(
        results, "No Intelligence",
        "Area Level — RL + GNN Improvement over Baseline (%)",
        out / "area_improvement.png",
    )


def _plot_city_level(results, configs, max_ticks):
    out = RESULTS_DIR / "city_level"
    out.mkdir(parents=True, exist_ok=True)
    _grouped_bar_chart(
        results, configs,
        f"City Level — No Intelligence vs Full Hierarchy\n({EVAL_SCENARIO} · {max_ticks} ticks · HSR+BTM network)",
        out / "city_comparison.png",
    )
    _improvement_chart(
        results, "No Intelligence",
        "City Level — Full Hierarchy Improvement over Baseline (%)",
        out / "city_improvement.png",
    )


def _plot_hierarchy_comparison(all_level_results):
    """Side-by-side comparison across all 3 hierarchy levels."""
    try:
        plt = _get_plt()
    except ImportError:
        return

    levels = ["ward", "area", "city"]
    level_labels = ["Ward Level\n(RL Agent)", "Area Level\n(RL + GNN)", "City Level\n(Full Hierarchy)"]
    metrics = ["avg_speed", "congestion", "queue_length", "throughput", "waiting_time"]
    lower_better = {"congestion", "queue_length", "waiting_time"}
    level_colors = [C_PPO, C_WITHSYS, C_FULLHIER]

    fig, ax = plt.subplots(figsize=(13, 5.5))
    x = np.arange(len(metrics))
    bar_w = 0.22

    for lv_idx, (level, lv_label, color) in enumerate(zip(levels, level_labels, level_colors)):
        lv_data = all_level_results.get(level, {})
        if not lv_data:
            continue

        config_names = list(lv_data.keys())
        baseline_key = config_names[0]
        best_rl_key = config_names[-1]
        baseline = lv_data[baseline_key]
        rl_data = lv_data[best_rl_key]

        improvements = []
        for m in metrics:
            bval = baseline.get(m, 0.0)
            rval = rl_data.get(m, 0.0)
            if abs(bval) < 1e-8:
                improvements.append(0.0)
            elif m in lower_better:
                improvements.append(((bval - rval) / abs(bval)) * 100)
            else:
                improvements.append(((rval - bval) / abs(bval)) * 100)

        offset = (lv_idx - 1) * bar_w
        bars = ax.bar(x + offset, improvements, bar_w * 0.85, label=lv_label,
                      color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, improvements):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.3 if val >= 0 else -1.5),
                    f"{val:+.1f}%", ha="center",
                    va="bottom" if val >= 0 else "top",
                    fontsize=7, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics], fontsize=10)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("% Improvement over No-Intelligence Baseline", fontsize=10)
    ax.set_title("Hierarchical Comparison — Each Level's Best vs Its Baseline",
                 fontsize=13, fontweight="bold", pad=10)
    ax.legend(framealpha=0.9, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    path = RESULTS_DIR / "hierarchy_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Hierarchy comparison chart → %s", path)


# ======================================================================
# MAIN
# ======================================================================

def main():
    args = parse_args()
    pipeline_start = time.time()

    print("=" * 60)
    print("  FAST PIPELINE — Train + 3-Level Evaluate")
    print("=" * 60)
    print(f"  Wards      : {WARD_IDS}")
    print(f"  Scenarios   : {SCENARIO_IDS}")
    print(f"  RL Episodes : {args.episodes}")
    print(f"  Sim steps   : {args.max_steps}/episode")
    print(f"  GNN epochs  : {args.gnn_epochs}")
    print(f"  Eval ticks  : {args.eval_ticks}")
    print(f"  Eval levels : Ward / Area / City")
    print("=" * 60)

    # ── Stage 1 & 2: RL ──────────────────────────────────────────────────
    if not args.skip_rl:
        train_rl("ppo", args.episodes, args.max_steps, collect_gnn=True)
        train_rl("dqn", args.episodes, args.max_steps, collect_gnn=False)
    else:
        log.info("⏭️  Skipping RL training (--skip-rl)")

    # ── Stage 3: STGCN ────────────────────────────────────────────────────
    if not args.skip_gnn:
        train_stgcn(args.gnn_epochs)
    else:
        log.info("⏭️  Skipping STGCN training (--skip-gnn)")

    # ── Stage 4: 3-Level Evaluation ───────────────────────────────────────
    if not args.skip_eval:
        run_full_evaluation(args.eval_ticks)
    else:
        log.info("⏭️  Skipping evaluation (--skip-eval)")

    elapsed = time.time() - pipeline_start
    print()
    print("=" * 60)
    print(f"  ✅  PIPELINE COMPLETE in {elapsed/60:.1f} minutes")
    print(f"  Results → {RESULTS_DIR}")
    print()
    print("  📊 Outputs:")
    print(f"     Ward plots  → {RESULTS_DIR / 'ward_level'}")
    print(f"     Area plots  → {RESULTS_DIR / 'area_level'}")
    print(f"     City plots  → {RESULTS_DIR / 'city_level'}")
    print(f"     Hierarchy   → {RESULTS_DIR / 'hierarchy_comparison.png'}")
    print(f"     Per-metric  → {RESULTS_DIR / '*_individual_metrics/'}")
    print(f"     JSON        → {RESULTS_DIR / 'full_evaluation_results.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
