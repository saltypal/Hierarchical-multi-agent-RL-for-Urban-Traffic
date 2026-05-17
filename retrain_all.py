import sys
import os
import argparse
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Project Root Setup
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rl.train import train_global_agent
from src.topology import Topology
from src.controllers.area_controller import AreaForecaster

def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='valid')
    return y_smooth

def parse_args():
    parser = argparse.ArgumentParser(description="Hierarchical RL and GNN retraining pipeline.")
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of episodes to train each RL agent (default: 20 for fast verification)."
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Show SUMO GUI during training."
    )
    parser.add_argument(
        "--gnn-epochs",
        type=int,
        default=100,
        help="Number of offline training epochs for the Area GNN (default: 100)."
    )
    parser.add_argument(
        "--map-dir",
        type=str,
        default="processed",
        help="Map namespace directory name under maps/ (default: 'processed')."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set the dynamic map directory environment variable
    os.environ["HMRL_MAP_DIR"] = args.map_dir
    logger.info(f"Using map directory namespace: {args.map_dir}")
    
    # Load all 16 Wards from Registry
    registry_path = PROJECT_ROOT / 'configs' / 'hierarchy' / 'ward_registry.json'
    if not registry_path.exists():
        logger.error(f"Registry path not found: {registry_path}")
        sys.exit(1)
        
    with registry_path.open('r', encoding='utf-8') as f:
        registry = json.load(f)
        WARD_IDS = list(registry.get('wards', {}).keys())

    SCENARIOS = [
        'normal', 
        'peak_congestion', 
        'traffic_surge', 
        'blocked_road', 
        'ambulance_emergency', 
        'vip_convoy', 
        'chaos_mode',
        'breakdown',
        'asymmetric_overload',
        'low_baseline'
    ]
    
    ALGORITHMS = ['ppo', 'a2c', 'dqn']
    
    print("======================================================")
    print("      Starting Hierarchical Retraining Pipeline       ")
    print("======================================================")
    print(f"Configured Wards: {len(WARD_IDS)}")
    print(f"Configured Scenarios: {len(SCENARIOS)}")
    print(f"RL Episodes Per Agent: {args.episodes}")
    print(f"GNN Epochs: {args.gnn_epochs}")
    print("======================================================")
    
    all_rewards = {}
    
    for algo in ALGORITHMS:
        print(f"\n[RL TRAINING] Starting Global Training: {algo.upper()}")
        print("-" * 50)
        
        # We only collect GNN training data during PPO training to have a single clean snapshot.
        collect_gnn = (algo == 'ppo')
        
        result = train_global_agent(
            ward_ids=WARD_IDS,
            scenario_ids=SCENARIOS,
            project_root=PROJECT_ROOT,
            algorithm=algo,
            episodes=args.episodes,
            gui=args.gui,
            collect_gnn_data=collect_gnn
        )
        
        all_rewards[algo] = result['episode_rewards']
        print(f"\n[SUCCESS] {algo.upper()} training complete in {result['training_time']/60:.2f} minutes.")
        print(f"Model saved to: {result['model_path']}")
        
    # --- Generate GNN Dataset and train GNN ---
    print("\n[GNN TRAINING] Starting Offline GNN Forecaster Training")
    print("-" * 50)
    
    topology = Topology(PROJECT_ROOT)
    AREA_ID = 'HSR_Layout'
    forecaster = AreaForecaster(AREA_ID, topology)
    
    gnn_dir = PROJECT_ROOT / 'models' / 'gnn'
    global_data_path = gnn_dir / 'global_gnn_data.pt'
    
    if not global_data_path.exists():
        logger.error(f"GNN global data file not found: {global_data_path}")
        logger.info("Skipping GNN training. Run with ppo to generate the data first.")
    else:
        all_data = torch.load(global_data_path, weights_only=False)
        print(f"Loaded {len(all_data)} tensor snapshots from {global_data_path.name}")
        
        combined_path = gnn_dir / 'combined_training_data.pt'
        torch.save(all_data, combined_path)
        
        print(f"Training on device: {forecaster.device}")
        losses = forecaster.train_offline(combined_path, epochs=args.gnn_epochs, save_dir=gnn_dir)
        print(f"[SUCCESS] GNN offline training complete. Final Loss: {losses[-1]:.6f}")
        print(f"GNN weights saved to: {gnn_dir}/area_model.pt")

    # --- Plot Performance Comparisons ---
    print("\n[PLOTTING] Generating Performance Comparison Plot")
    print("-" * 50)
    
    plt.figure(figsize=(12, 6))
    colors = {'ppo': '#0072B2', 'a2c': '#D55E00', 'dqn': '#009E73'}
    
    for algo, rewards in all_rewards.items():
        if len(rewards) > 10:
            smoothed = smooth(rewards, 10)
            plt.plot(smoothed, label=f'{algo.upper()} (Moving Avg)', color=colors[algo], linewidth=2)
            plt.plot(rewards, alpha=0.2, color=colors[algo])
        else:
            plt.plot(rewards, label=algo.upper(), color=colors[algo], linewidth=2)
            
    plt.title('Global Multi-Ward RL Training Performance', fontsize=16, fontweight='bold')
    plt.xlabel('Episodes', fontsize=14)
    plt.ylabel('Cumulative Reward', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    save_dir = PROJECT_ROOT / 'results' / 'training' / 'wards' / 'image'
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / 'global_training_performance.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"[SUCCESS] Performance comparison plot saved to {save_path}")
    print("\n======================================================")
    print("       Retraining Pipeline Executed Successfully       ")
    print("======================================================")

if __name__ == "__main__":
    main()
