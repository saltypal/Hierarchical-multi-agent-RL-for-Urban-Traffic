import sys
import logging
import argparse
from pathlib import Path
import torch

PROJECT_ROOT = Path("d:/Bunker/BaseCamp/Hierarchical-multi-agent-RL-for-Urban-Traffic")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.topology import Topology
from src.controllers.area_controller import AreaForecaster

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def train_for_area(area_id: str, topology: Topology, combined_path: Path, gnn_dir: Path, epochs: int):
    print(f"\n======================================================")
    print(f"Starting Area GNN Training for: {area_id}")
    print(f"======================================================")
    
    # Initialize forecaster, passing gnn_dir as model_dir to load existing weights (incremental training/fine-tuning)
    forecaster = AreaForecaster(area_id, topology, model_dir=gnn_dir)
    
    print(f"Training GNN for {area_id} on {forecaster.device.type.upper()}...")
    losses = forecaster.train_offline(combined_path, epochs=epochs, save_dir=gnn_dir)
    print(f"Completed {area_id}! Final MSE Loss: {losses[-1]:.6f}")

def main():
    parser = argparse.ArgumentParser(description="Area GNN Forecaster Training CLI")
    parser.add_argument(
        "--area", "-a",
        default="all",
        help="Constituency/Area to train GNN on (e.g. HSR_Layout, BTM_Layout, BTM_Layout_1, Jayanagar, or 'all')."
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=100,
        help="Number of training epochs per area."
    )
    args = parser.parse_args()

    print("======================================================")
    print("Hierarchical Traffic GNN Offline Training Console")
    print("======================================================")
    
    topology = Topology(PROJECT_ROOT)
    gnn_dir = PROJECT_ROOT / 'models' / 'gnn'
    temporal_data_path = gnn_dir / 'global_temporal_data.pt'
    global_data_path = gnn_dir / 'global_gnn_data.pt'

    source_path = temporal_data_path if temporal_data_path.exists() else global_data_path

    if not source_path.exists():
        print("Error: no GNN training dataset found!")
        print("   Please run the ward training notebook or the temporal dataset export first.")
        sys.exit(1)
        
    all_data = torch.load(source_path, weights_only=False)
    print(f"Loaded {len(all_data)} training samples from {source_path.name}")
    
    combined_path = gnn_dir / 'combined_training_data.pt'
    torch.save(all_data, combined_path)
    
    available_areas = topology.get_all_area_ids()
    print(f"Available areas in topology: {available_areas}")
    
    if args.area == "all":
        # Train sequentially on all registered areas
        print("\n[*] Option 'all' selected. Training sequentially on all areas to learn generalizable weights...")
        for area_id in ["HSR_Layout", "BTM_Layout", "BTM_Layout_1", "Jayanagar"]:
            if area_id in available_areas:
                train_for_area(area_id, topology, combined_path, gnn_dir, args.epochs)
            else:
                print(f"[-] Skipping area {area_id} (not in topology registry)")
    else:
        if args.area not in available_areas:
            print(f"Error: Specified area '{args.area}' is not in topology registry.")
            print(f"       Available areas: {available_areas}")
            sys.exit(1)
        train_for_area(args.area, topology, combined_path, gnn_dir, args.epochs)

    print("\nJoint GNN Training Session Complete!")
    print(f"Model saved to: {gnn_dir}/area_model.pt")

if __name__ == "__main__":
    main()
