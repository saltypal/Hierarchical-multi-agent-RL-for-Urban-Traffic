import sys
from pathlib import Path
import json

PROJECT_ROOT = Path("d:/Bunker/BaseCamp/Hierarchical-multi-agent-RL-for-Urban-Traffic")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.traffic_generator import TrafficGenerator

def main():
    path = PROJECT_ROOT / "configs" / "hierarchy" / "ward_registry.json"
    with path.open("r", encoding="utf-8") as f:
        registry = json.load(f)
    wards = list(registry["wards"].keys())
    
    print(f"[*] Generating routes and sumocfg for {len(wards)} wards...")
    
    for ward_id in wards:
        try:
            gen = TrafficGenerator(PROJECT_ROOT, ward_id)
            gen.generate_ward_routes(num_vehicles=500)
            gen.generate_ward_sumocfg()
            print(f"[+] Done: {ward_id}")
        except Exception as e:
            print(f"[-] Failed: {ward_id} -> {e}")

if __name__ == "__main__":
    main()
