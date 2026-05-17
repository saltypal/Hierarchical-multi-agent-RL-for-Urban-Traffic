import sys
import logging
from pathlib import Path
import json

PROJECT_ROOT = Path("d:/Bunker/BaseCamp/Hierarchical-multi-agent-RL-for-Urban-Traffic")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.ward_processor import process_ward
from src.traffic_generator import TrafficGenerator

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("Rebuild")

def main():
    registry_path = PROJECT_ROOT / "configs" / "hierarchy" / "ward_registry.json"
    with registry_path.open("r", encoding="utf-8") as f:
        registry = json.load(f)
        
    wards = list(registry.get("wards", {}).keys())
    logger.info(f"Rebuilding network topology and routes for {len(wards)} wards...")
    
    for ward_id in wards:
        try:
            logger.info(f"--- Processing {ward_id} ---")
            # This triggers netconvert with the new join-same and edges.join flags!
            # It also recreates boundaries.json cleanly.
            process_ward(ward_id, PROJECT_ROOT, strict_mode=False)
            
            # Regenerate the routes without dictionary corruption!
            gen = TrafficGenerator(PROJECT_ROOT, ward_id)
            gen.generate_ward_routes(num_vehicles=1000)
            gen.generate_ward_sumocfg()
        except Exception as e:
            logger.error(f"Failed to rebuild {ward_id}: {e}")

if __name__ == "__main__":
    main()
