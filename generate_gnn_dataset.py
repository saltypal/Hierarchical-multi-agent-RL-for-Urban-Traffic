import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path("d:/Bunker/BaseCamp/Hierarchical-multi-agent-RL-for-Urban-Traffic")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rl.train import train_global_agent

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def main():
    print("======================================================")
    print("🚀 Compiling GNN Dataset (Fast 20-Episode PPO Run)")
    print("======================================================")
    
    # Run PPO on 16 wards for just 20 episodes to collect pristine GNN tensor states
    result = train_global_agent(
        ward_ids=[
            "ward_070", "ward_071", "ward_072", 
            "ward_017", "ward_018", "ward_019", "ward_020",
            "ward_007", "ward_008", "ward_009", "ward_010", 
            "ward_011", "ward_012", "ward_013", "ward_014", "ward_015"
        ],
        scenario_ids=["normal", "peak_congestion", "chaos_mode"],
        project_root=PROJECT_ROOT,
        algorithm="ppo",
        episodes=20,
        gui=False,
        collect_gnn_data=True
    )
    
    print("\n✅ Dataset successfully compiled to models/gnn/global_gnn_data.pt!")

if __name__ == "__main__":
    main()
