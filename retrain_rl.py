#!/usr/bin/env python
"""Robust RL Retraining and Monitoring Runner.

Triggers sequential training for PPO and DQN agents, captures stdout,
monitors for exceptions, and manages logging.
"""

import sys
import os
import argparse
import logging
from pathlib import Path
import json

# Force project root onto path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
log_dir = PROJECT_ROOT / "results" / "training"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "retrain_rl.log"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode="w", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

# Try imports and handle gracefully
try:
    from src.rl.train import train_global_agent
    from configs.scenarios import list_scenarios
except Exception as e:
    logger.exception("Failed to import required modules from the codebase.")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Robust RL Retraining and Monitoring CLI")
    parser.add_argument(
        "--episodes", "-e", type=int, default=300,
        help="Number of training episodes for each RL algorithm (default: 300)."
    )
    parser.add_argument(
        "--max-steps", "-s", type=int, default=1200,
        help="Maximum simulation steps per episode (default: 1200)."
    )
    parser.add_argument(
        "--map-dir", "-m", type=str, default="processed",
        help="Map namespace directory name under maps/ (default: 'processed')."
    )
    parser.add_argument(
        "--verify", "-v", action="store_true",
        help="Run in verification/smoke-test mode (2 episodes each, no GUI)."
    )
    parser.add_argument(
        "--gui", action="store_true",
        help="Display SUMO GUI during training (not recommended for speed)."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set the map directory environment variable
    os.environ["HMRL_MAP_DIR"] = args.map_dir
    logger.info("=" * 60)
    logger.info("    HIERARCHICAL TRAFFIC RL RETRAINING CONSOLE")
    logger.info("=" * 60)
    logger.info(f"Target Map Directory : {args.map_dir}")
    logger.info(f"SUMO GUI Enabled     : {args.gui}")
    
    episodes = 2 if args.verify else args.episodes
    max_steps = args.max_steps
    
    logger.info(f"Execution Mode       : {'VERIFICATION (SMOKE TEST)' if args.verify else 'FULL RETRAINING'}")
    logger.info(f"Episodes Per Model   : {episodes}")
    logger.info(f"Max Simulation Ticks : {max_steps}")
    logger.info(f"Logs Saved To        : {log_file}")
    logger.info("=" * 60)
    
    # Load all Wards from ward registry
    registry_path = PROJECT_ROOT / "configs" / "hierarchy" / "ward_registry.json"
    if not registry_path.exists():
        logger.error(f"Ward registry not found: {registry_path}")
        sys.exit(1)
        
    try:
        with registry_path.open("r", encoding="utf-8") as fh:
            registry = json.load(fh)
        ward_ids = list(registry.get("wards", {}).keys())
    except Exception as e:
        logger.exception("Failed to parse ward registry.")
        sys.exit(1)
        
    # Filter wards if using Processed_Map_2 or any custom map to prevent non-existent map directory lookup
    map_path = PROJECT_ROOT / "maps" / args.map_dir
    if map_path.exists():
        available_wards = [d.name for d in map_path.iterdir() if d.is_dir() and d.name.startswith("ward_")]
        ward_ids = [w for w in ward_ids if w in available_wards]
    else:
        logger.error(f"Map directory does not exist: {map_path}")
        sys.exit(1)
    
    logger.info(f"Loaded Wards ({len(ward_ids)}) : {ward_ids}")
    
    try:
        scenarios = list_scenarios()
    except Exception as e:
        logger.exception("Failed to load scenario lists.")
        sys.exit(1)
        
    logger.info(f"Loaded Scenarios ({len(scenarios)}) : {scenarios}")
    logger.info("-" * 60)
    
    # Check SUMO installation
    if "SUMO_HOME" not in os.environ:
        logger.warning("SUMO_HOME environment variable is not set. Ensure SUMO is in PATH.")
    else:
        logger.info(f"SUMO_HOME is set to: {os.environ['SUMO_HOME']}")
        
    # ----------------------------------------------------
    # Stage 1: PPO Training
    # ----------------------------------------------------
    logger.info("[STAGE 1/2] Starting PPO Training...")
    try:
        ppo_results = train_global_agent(
            ward_ids=ward_ids,
            scenario_ids=scenarios,
            project_root=PROJECT_ROOT,
            algorithm="ppo",
            episodes=episodes,
            gui=args.gui,
            collect_gnn_data=True,
            max_simulation_steps=max_steps
        )
        logger.info("✅ PPO Training Complete!")
        logger.info(f"   Model Saved To: {ppo_results['model_path']}")
        if ppo_results.get("gnn_data_path"):
            logger.info(f"   GNN Dataset:    {ppo_results['gnn_data_path']}")
    except Exception as e:
        logger.exception("❌ Error occurred during PPO training!")
        logger.info(f"Please inspect the logs at {log_file} for stack traces.")
        sys.exit(1)
        
    logger.info("-" * 60)
    
    # ----------------------------------------------------
    # Stage 2: DQN Training
    # ----------------------------------------------------
    logger.info("[STAGE 2/2] Starting DQN Training...")
    try:
        dqn_results = train_global_agent(
            ward_ids=ward_ids,
            scenario_ids=scenarios,
            project_root=PROJECT_ROOT,
            algorithm="dqn",
            episodes=episodes,
            gui=args.gui,
            collect_gnn_data=False,
            max_simulation_steps=max_steps
        )
        logger.info("✅ DQN Training Complete!")
        logger.info(f"   Model Saved To: {dqn_results['model_path']}")
    except Exception as e:
        logger.exception("❌ Error occurred during DQN training!")
        logger.info(f"Please inspect the logs at {log_file} for stack traces.")
        sys.exit(1)
        
    logger.info("=" * 60)
    logger.info("    ALL RL TRAINING PHASES COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
