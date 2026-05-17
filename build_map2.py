import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path("d:/Bunker/BaseCamp/Hierarchical-multi-agent-RL-for-Urban-Traffic")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.ward_processor import process_ward
from src.traffic_generator import TrafficGenerator

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("BuildMap2")

MAP2_DIR_NAME = "Processed_Map_2"


def main() -> None:
    os.environ["HMRL_MAP_DIR"] = MAP2_DIR_NAME

    registry_path = PROJECT_ROOT / "configs" / "hierarchy" / "ward_registry.json"
    with registry_path.open("r", encoding="utf-8") as f:
        registry = json.load(f)

    wards = sorted(registry.get("wards", {}).keys())
    logger.info("Building %s for %d wards...", MAP2_DIR_NAME, len(wards))

    # Filter only major primary/secondary roads to speed up training/inference as requested!
    extra_args = [
        "--keep-edges.by-type", "highway.primary,highway.primary_link,highway.secondary,highway.secondary_link,highway.trunk,highway.trunk_link,highway.motorway,highway.motorway_link",
        "--keep-edges.components", "1",  # Keep only the largest connected component
    ]
    for ward_id in wards:
        try:
            logger.info("--- Processing %s ---", ward_id)
            process_ward(
                ward_id=ward_id,
                project_root=PROJECT_ROOT,
                strict_mode=True,
                output_dir_name=MAP2_DIR_NAME,
                extra_netconvert_args=extra_args,
            )
            gen = TrafficGenerator(PROJECT_ROOT, ward_id)
            gen.generate_ward_routes(num_vehicles=1000)
            gen.generate_ward_sumocfg()
        except Exception as exc:
            logger.error("Failed to build %s: %s", ward_id, exc)

    logger.info("Map2 build complete.")


if __name__ == "__main__":
    main()
