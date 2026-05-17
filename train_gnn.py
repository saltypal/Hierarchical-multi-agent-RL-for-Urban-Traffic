import sys
import logging
from pathlib import Path
import torch

PROJECT_ROOT = Path("d:/Bunker/BaseCamp/Hierarchical-multi-agent-RL-for-Urban-Traffic")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.topology import Topology
from src.controllers.area_controller import AreaForecaster

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def main():
    print("======================================================")
    print("🧠 Starting Area GNN Forecaster Training")
    print("======================================================")
    
    topology = Topology(PROJECT_ROOT)
    AREA_ID = 'HSR_Layout'
    
    forecaster = AreaForecaster(AREA_ID, topology)
    
    gnn_dir = PROJECT_ROOT / 'models' / 'gnn'
    global_data_path = gnn_dir / 'global_gnn_data.pt'
    
    if not global_data_path.exists():
        print("❌ Error: global_gnn_data.pt not found!")
        print("   Please run `python generate_gnn_dataset.py` first.")
        sys.exit(1)
        
    all_data = torch.load(global_data_path, weights_only=False)
    print(f"⚡ Loaded {len(all_data)} tensor snapshots from {global_data_path.name}")
    
    # Optional: We save combined_training_data.pt for legacy compatibility if needed
    combined_path = gnn_dir / 'combined_training_data.pt'
    torch.save(all_data, combined_path)
    
    print(f"\n🚀 Training on {forecaster.device.type.upper()}...")
    losses = forecaster.train_offline(combined_path, epochs=100, save_dir=gnn_dir)
    print(f"\n✅ Training Complete! Final MSE Loss: {losses[-1]:.6f}")
    print(f"✅ GNN Weights saved to: {gnn_dir}/area_model.pt")

if __name__ == "__main__":
    main()
