import sys
import json
from pathlib import Path

PROJECT_ROOT = Path("d:/Bunker/BaseCamp/Hierarchical-multi-agent-RL-for-Urban-Traffic")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.ward_processor import process_ward

def main():
    path = PROJECT_ROOT / "configs" / "hierarchy" / "ward_registry.json"
    with path.open("r", encoding="utf-8") as f:
        registry = json.load(f)
    wards = list(registry["wards"].keys())
    
    print(f"[*] Starting preprocessing for {len(wards)} wards...")
    success_count = 0
    for ward_id in wards:
        print(f"\n--- Processing {ward_id} ---")
        try:
            result = process_ward(ward_id, PROJECT_ROOT, strict_mode=True)
            print(f"[+] Success: {ward_id}")
            success_count += 1
        except Exception as e:
            print(f"[-] Failed for {ward_id}: {e}")
            
    print(f"\n[*] Preprocessing finished. {success_count}/{len(wards)} succeeded.")

if __name__ == "__main__":
    main()
