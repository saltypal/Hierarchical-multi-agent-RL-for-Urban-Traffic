"""
plot_results.py
---------------
Generate bar graphs from evaluation metrics JSON files.

Usage:
    python plot_results.py                                 # auto-detect results/full_evaluation_results.json
    python plot_results.py --file results/full_evaluation_results.json
    python plot_results.py --file results/eval_metrics_comparison.json --level ward
    python plot_results.py --output my_custom_name        # saves as my_custom_name_<metric>.png
    python plot_results.py --all-in-one                   # one big combined figure instead of per-metric files
    python plot_results.py --show                         # also open the plots interactively
"""

import argparse
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # non-interactive by default; overridden with --show
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ──────────────────────────────────────────────────────────────
# Style
# ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.edgecolor":   "#cccccc",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.color":       "#eeeeee",
    "grid.linestyle":   "-",
    "grid.linewidth":   0.8,
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.frameon":   False,
})

# Colour palette (one colour per model / system variant)
PALETTE = [
    "#4361EE",  # blue
    "#F72585",  # pink/magenta
    "#7209B7",  # purple
    "#4CC9F0",  # cyan
    "#F4A261",  # orange
    "#2A9D8F",  # teal
]

# Metrics that are "lower-is-better" (displayed with a dagger in title)
LOWER_IS_BETTER = {"congestion", "queue_length", "travel_time", "waiting_time",
                   "ambulance_delay", "incident_delay"}

# Human-readable metric labels
METRIC_LABELS = {
    "avg_speed":       "Avg Speed (m/s)",
    "congestion":      "Congestion Ratio",
    "queue_length":    "Queue Length (vehicles)",
    "throughput":      "Throughput (vehicles)",
    "travel_time":     "Travel Time (s)",
    "waiting_time":    "Waiting Time (s)",
    "ambulance_delay": "Ambulance Delay (s)",
    "incident_delay":  "Incident Delay (s)",
    "reroute_count":   "Reroute Count",
}


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def detect_level(data: dict) -> str:
    """Auto-detect whether the JSON is multi-level or flat (ward/area/city)."""
    first_val = next(iter(data.values()))
    if isinstance(first_val, dict):
        inner = next(iter(first_val.values()))
        if isinstance(inner, dict):
            return "multi"   # full_evaluation_results.json style
    return "flat"            # eval_metrics_comparison.json style


def get_all_metrics(models: dict) -> list:
    metrics = set()
    for v in models.values():
        metrics.update(v.keys())
    return sorted(metrics)


def bar_chart(ax, models: dict, metric: str, title: str):
    """Draw a grouped/single bar chart for one metric on the given axes."""
    labels = list(models.keys())
    values = [models[m].get(metric, 0.0) for m in labels]
    x = np.arange(len(labels))
    colours = [PALETTE[i % len(PALETTE)] for i in range(len(labels))]

    bars = ax.bar(x, values, width=0.55, color=colours, zorder=3, edgecolor="white", linewidth=0.5)

    # Value annotations on top of bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.01,
            f"{val:.2f}" if val < 10_000 else f"{val:,.0f}",
            ha="center", va="bottom", fontsize=9, color="#333333"
        )

    label = METRIC_LABELS.get(metric, metric.replace("_", " ").title())
    suffix = "  (lower is better)" if metric in LOWER_IS_BETTER else "  (higher is better)"
    ax.set_title(label + suffix, fontweight="bold", pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.1f}"))
    ax.set_ylabel(label)


def plot_level(level_name: str, models: dict, output_prefix: str,
               all_in_one: bool, show: bool, out_dir: Path):
    """
    Generate bar graphs for one level (ward / area / city).
    If all_in_one: one figure with subplots. Otherwise: one file per metric.
    """
    metrics = [m for m in get_all_metrics(models) if m != "reroute_count" or
               any(models[mod].get("reroute_count", 0) > 0 for mod in models)]

    if all_in_one:
        ncols = 3
        nrows = (len(metrics) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
        fig.suptitle(f"{level_name.upper()} LEVEL — Model Comparison", fontsize=15, fontweight="bold", y=1.01)
        axes_flat = axes.flatten() if nrows > 1 else [axes] if ncols == 1 else axes.flatten()

        for i, metric in enumerate(metrics):
            bar_chart(axes_flat[i], models, metric, metric)

        # hide spare axes
        for j in range(len(metrics), len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.tight_layout()
        fname = out_dir / f"{output_prefix}_{level_name}_all_metrics.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"  Saved: {fname}")
        if show:
            matplotlib.use("TkAgg")
            plt.show()
        plt.close(fig)
    else:
        for metric in metrics:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            bar_chart(ax, models, metric, metric)
            fig.suptitle(f"{level_name.upper()} — {METRIC_LABELS.get(metric, metric)}", fontsize=13, fontweight="bold")
            fig.tight_layout()
            fname = out_dir / f"{output_prefix}_{level_name}_{metric}.png"
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            print(f"  Saved: {fname}")
            if show:
                plt.show()
            plt.close(fig)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate bar graph plots from evaluation metric JSON files."
    )
    parser.add_argument(
        "--file", "-f",
        type=Path,
        default=None,
        help="Path to JSON metrics file. Auto-detects full_evaluation_results.json if omitted."
    )
    parser.add_argument(
        "--level", "-l",
        choices=["ward", "area", "city", "all"],
        default="all",
        help="Which level to plot (only applies to full_evaluation_results.json). Default: all."
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output filename prefix (without extension). E.g. 'my_run' -> my_run_ward_avg_speed.png"
    )
    parser.add_argument(
        "--out-dir", "-d",
        type=Path,
        default=None,
        help="Directory to save plots. Defaults to results/plots/ next to the JSON file."
    )
    parser.add_argument(
        "--all-in-one",
        action="store_true",
        help="Put all metrics for a level into one combined figure instead of per-metric files."
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also display plots interactively (requires a display/GUI)."
    )
    args = parser.parse_args()

    # ── Resolve file path ────────────────────────────────────────
    if args.file is None:
        candidates = [
            Path("results/full_evaluation_results.json"),
            Path("results/eval_metrics_comparison.json"),
        ]
        for c in candidates:
            if c.exists():
                args.file = c
                break
        if args.file is None:
            print("ERROR: Could not find a metrics JSON file. Use --file to specify one.")
            sys.exit(1)

    if not args.file.exists():
        print(f"ERROR: File not found: {args.file}")
        sys.exit(1)

    print(f"\nLoading: {args.file}")
    data = load_json(args.file)

    # ── Resolve output directory ─────────────────────────────────
    out_dir = args.out_dir or (args.file.parent / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Resolve output prefix ────────────────────────────────────
    output_prefix = args.output or args.file.stem

    # ── Dispatch by format ───────────────────────────────────────
    style = detect_level(data)

    if style == "flat":
        # e.g. eval_metrics_comparison.json — flat: { "Baseline": {metrics}, "DQN": {metrics} }
        print(f"\nDetected: flat metrics file (single level)")
        level_name = args.level if args.level != "all" else "ward"
        plot_level(level_name, data, output_prefix, args.all_in_one, args.show, out_dir)

    else:
        # e.g. full_evaluation_results.json — { "ward": { models... }, "area": {...}, "city": {...} }
        print(f"\nDetected: multi-level metrics file")
        levels_to_plot = ["ward", "area", "city"] if args.level == "all" else [args.level]

        for lvl in levels_to_plot:
            if lvl not in data:
                print(f"  WARNING: level '{lvl}' not found in file, skipping.")
                continue
            print(f"\nPlotting: {lvl.upper()} level")
            plot_level(lvl, data[lvl], output_prefix, args.all_in_one, args.show, out_dir)

    print(f"\nAll plots saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
