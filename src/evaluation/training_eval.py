"""Training evaluation metrics and plotting.

Separate from simulation evaluation — this module evaluates model training
performance: reward curves, losses, entropy for RL; MAE/RMSE/R² for GNN.

Generates publication-quality plots saved to results/training/.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

logger = logging.getLogger(__name__)

PLOT_DIR_DEFAULT = Path("results/training")


def plot_rl_training(
    training_result: dict[str, Any],
    algorithm: str,
    output_dir: Path | None = None,
) -> list[Path]:
    """Plot RL training metrics: reward curve, moving average.

    Args:
        training_result: Dict from train_global_agent() with episode_rewards, etc.
        algorithm: Algorithm name (ppo/dqn).
        output_dir: Output directory for plots.

    Returns:
        List of paths to saved plots.
    """
    if not HAS_MPL:
        return []

    output_dir = output_dir or PLOT_DIR_DEFAULT
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    rewards = training_result.get("episode_rewards", [])
    if rewards:
        fig, ax = plt.subplots(figsize=(10, 5))

        # Raw rewards
        ax.plot(rewards, alpha=0.3, color="#6366f1", linewidth=0.8, label="Episode Reward")

        # Moving average
        window = min(10, len(rewards))
        if len(rewards) >= window:
            avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
            ax.plot(range(window - 1, len(rewards)), avg, color="#6366f1", linewidth=2, label=f"{window}-ep Moving Avg")

        ax.set_xlabel("Episode")
        ax.set_ylabel("Cumulative Reward")
        ax.set_title(f"{algorithm.upper()} Ward Training — Reward Curve")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = output_dir / f"ward_{algorithm}_reward_curve.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)
        logger.info("RL reward curve saved → %s", path)

    return saved


def plot_gnn_training(
    gcn_result: dict[str, Any] | None = None,
    stgcn_result: dict[str, Any] | None = None,
    output_dir: Path | None = None,
) -> list[Path]:
    """Plot GNN training metrics: loss curves, comparative chart.

    Returns:
        List of paths to saved plots.
    """
    if not HAS_MPL:
        return []

    output_dir = output_dir or PLOT_DIR_DEFAULT
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    # Individual loss curves
    for result, label, color in [
        (gcn_result, "GCN", "#6366f1"),
        (stgcn_result, "STGCN", "#f43f5e"),
    ]:
        if result is None:
            continue
        losses = result.get("losses", [])
        if not losses:
            continue

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(losses, linewidth=1.5, color=color)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title(f"{label} Training Loss — {result.get('area_id', 'all')}")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = output_dir / f"area_{label.lower()}_loss.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)

    # Comparative plot
    if gcn_result and stgcn_result and gcn_result.get("losses") and stgcn_result.get("losses"):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(gcn_result["losses"], label="GCN", linewidth=1.5, color="#6366f1")
        ax1.plot(stgcn_result["losses"], label="STGCN", linewidth=1.5, color="#f43f5e")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("MSE Loss")
        ax1.set_title("Training Loss Comparison")
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Bar chart comparison
        names = ["GCN", "STGCN"]
        mses = [gcn_result["final_mse"], stgcn_result["final_mse"]]
        bars = ax2.bar(names, mses, color=["#6366f1", "#f43f5e"], alpha=0.85)
        for bar, mse in zip(bars, mses):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                     f"{mse:.5f}", ha="center", va="bottom", fontsize=10)
        ax2.set_ylabel("Final MSE")
        ax2.set_title("Final MSE Comparison")
        ax2.grid(axis="y", alpha=0.3)

        fig.suptitle(f"GCN vs STGCN — Area Model Comparison", fontsize=13, fontweight="bold")
        fig.tight_layout()
        path = output_dir / "area_model_comparison.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)

    return saved


def compute_gnn_eval_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> dict[str, float]:
    """Compute regression metrics for GNN evaluation.

    Returns:
        Dict with MAE, RMSE, R² values.
    """
    predictions = np.asarray(predictions).flatten()
    targets = np.asarray(targets).flatten()

    mae = float(np.mean(np.abs(predictions - targets)))
    rmse = float(np.sqrt(np.mean((predictions - targets) ** 2)))

    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = float(1.0 - ss_res / (ss_tot + 1e-8))

    return {"mae": mae, "rmse": rmse, "r2": r2}
