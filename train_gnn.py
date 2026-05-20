"""Area GNN Forecaster Training CLI.

Trains GCN and/or STGCN models on collected ward temporal data.
Saves training loss curve plots and model weights.

Usage:
    python train_gnn.py --area all --model-type both --epochs 100
    python train_gnn.py --area HSR_Layout --model-type stgcn --epochs 50
"""

import sys
import logging
import argparse
from pathlib import Path

import numpy as np
import torch

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.topology import Topology
from src.controllers.area_controller import AreaForecaster

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _save_loss_plot(losses: list[float], title: str, save_path: Path) -> None:
    """Save a training loss curve plot."""
    if not HAS_MPL or not losses:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, linewidth=1.5, color="#6366f1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Loss plot saved → %s", save_path)


def train_for_area(
    area_id: str,
    topology: Topology,
    data_path: Path,
    gnn_dir: Path,
    epochs: int,
    model_type: str,
    plot_dir: Path,
) -> dict:
    """Train the specified model type for an area."""
    print(f"\n{'='*60}")
    print(f"Training {model_type.upper()} for: {area_id}")
    print(f"{'='*60}")

    forecaster = AreaForecaster(area_id, topology, model_dir=gnn_dir, model_type=model_type)
    print(f"Device: {forecaster.device.type.upper()} | Wards: {forecaster.n_wards}")

    result = forecaster.train_offline(data_path, epochs=epochs, save_dir=gnn_dir)

    print(f"✅ {model_type.upper()} Complete for {area_id}!")
    print(f"   Final MSE: {result['final_mse']:.6f} | Samples: {result['samples']}")

    # Save loss plot
    _save_loss_plot(
        result["losses"],
        f"{model_type.upper()} Training Loss — {area_id}",
        plot_dir / f"area_{model_type}_{area_id}_loss.png",
    )

    return result


def main():
    parser = argparse.ArgumentParser(description="Area GNN Forecaster Training CLI")
    parser.add_argument(
        "--area", "-a", default="all",
        help="Area to train on (e.g. HSR_Layout, BTM_Layout, or 'all').",
    )
    parser.add_argument(
        "--model-type", "-m", default="both",
        choices=["gcn", "stgcn", "both"],
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--epochs", "-e", type=int, default=100,
        help="Number of training epochs per area.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Hierarchical Traffic — Area GNN Training Console")
    print("=" * 60)

    topology = Topology(PROJECT_ROOT)
    gnn_dir = PROJECT_ROOT / "models" / "gnn"
    plot_dir = PROJECT_ROOT / "results" / "training"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Find dataset
    data_path = gnn_dir / "global_temporal_data.pt"
    if not data_path.exists():
        print("Error: No GNN training dataset found at", data_path)
        print("   Run ward training first to generate temporal data.")
        sys.exit(1)

    all_data = torch.load(data_path, weights_only=False)
    print(f"Loaded {len(all_data)} training samples from {data_path.name}")

    # Save combined path for AreaForecaster compatibility
    combined_path = gnn_dir / "combined_training_data.pt"
    torch.save(all_data, combined_path)

    available_areas = topology.get_all_area_ids()
    print(f"Available areas: {available_areas}")

    # Determine which areas to train
    if args.area == "all":
        target_areas = [a for a in ["HSR_Layout", "BTM_Layout"] if a in available_areas]
    else:
        if args.area not in available_areas:
            print(f"Error: '{args.area}' not in registry. Available: {available_areas}")
            sys.exit(1)
        target_areas = [args.area]

    # Determine model types
    model_types = ["gcn", "stgcn"] if args.model_type == "both" else [args.model_type]

    # Train
    all_results = {}
    for model_type in model_types:
        for area_id in target_areas:
            result = train_for_area(
                area_id, topology, combined_path, gnn_dir,
                args.epochs, model_type, plot_dir,
            )
            all_results[f"{model_type}_{area_id}"] = result

    # Print comparison table
    print(f"\n{'='*60}")
    print("Training Summary")
    print(f"{'='*60}")
    print(f"{'Model':<8} {'Area':<15} {'Final MSE':<12} {'Samples':<10}")
    print("-" * 50)
    for key, res in all_results.items():
        print(f"{res['model_type']:<8} {res['area_id']:<15} {res['final_mse']:<12.6f} {res['samples']:<10}")

    # Save comparative plot if both models trained
    if args.model_type == "both" and HAS_MPL:
        for area_id in target_areas:
            gcn_key = f"gcn_{area_id}"
            stgcn_key = f"stgcn_{area_id}"
            if gcn_key in all_results and stgcn_key in all_results:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(all_results[gcn_key]["losses"], label="GCN", linewidth=1.5, color="#6366f1")
                ax.plot(all_results[stgcn_key]["losses"], label="STGCN", linewidth=1.5, color="#f43f5e")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("MSE Loss")
                ax.set_title(f"GCN vs STGCN Training Loss — {area_id}")
                ax.legend()
                ax.grid(alpha=0.3)
                fig.tight_layout()
                cmp_path = plot_dir / f"area_comparison_{area_id}.png"
                fig.savefig(cmp_path, dpi=150)
                plt.close(fig)
                print(f"Comparison plot → {cmp_path}")

    print("\n✅ Training session complete!")


if __name__ == "__main__":
    main()
