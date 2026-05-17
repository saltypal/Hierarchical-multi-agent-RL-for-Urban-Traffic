import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path("d:/Bunker/BaseCamp/Hierarchical-multi-agent-RL-for-Urban-Traffic")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rl.train import train_global_agent

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("RetrainMap2")

MAP2_DIR_NAME = "Processed_Map_2"
ALGORITHMS = ["ppo", "a2c", "dqn"]
SCENARIOS = ["normal", "peak_congestion", "chaos_mode"]


def _load_ward_ids() -> list[str]:
    registry_path = PROJECT_ROOT / "configs" / "hierarchy" / "ward_registry.json"
    with registry_path.open("r", encoding="utf-8") as f:
        registry = json.load(f)
    return sorted(registry.get("wards", {}).keys())


def main() -> None:
    os.environ["HMRL_MAP_DIR"] = MAP2_DIR_NAME

    ward_ids = _load_ward_ids()
    if not ward_ids:
        raise RuntimeError("No wards found in ward_registry.json")

    logger.info(
        "Starting map2 retraining on %d wards with map dir '%s'",
        len(ward_ids), MAP2_DIR_NAME,
    )

    for algorithm in ALGORITHMS:
        logger.info("=== Training %s ===", algorithm.upper())
        result = train_global_agent(
            ward_ids=ward_ids,
            scenario_ids=SCENARIOS,
            project_root=PROJECT_ROOT,
            algorithm=algorithm,
            episodes=400,
            gui=False,
            collect_gnn_data=(algorithm == "ppo"),
        )
        logger.info(
            "%s complete. model=%s episodes=%s",
            algorithm.upper(),
            result.get("model_path"),
            result.get("episodes"),
        )

    logger.info(
        "Retraining complete. GNN dataset refreshed at models/gnn/global_gnn_data.pt",
    )


if __name__ == "__main__":
    main()
