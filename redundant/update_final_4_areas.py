import json
from pathlib import Path

registry_path = Path("configs/hierarchy/ward_registry.json")

areas = {
    "HSR_Layout": [70, 71, 72],
    "BTM_Layout": [17, 18],
    "BTM_Layout_1": [19, 20],
    "Jayanagar": [7, 8, 9, 10, 11, 12, 13, 14, 15]
}

relation_mapping = {
    70: 19883476, 71: 19883431, 72: 19883285,
    17: 19883408, 18: 19884537,
    19: 19883486, 20: 19884562,
    7: 19884598, 8: 19884593, 9: 19884551, 10: None, 
    11: 19883292, 12: 19883452, 13: 19884547, 14: 19883265, 15: 19883329
}

registry = {
    "schema_version": "0.5.0",
    "description": "Final 4 areas for training.",
    "regions": {
        "South_Corridor": {
            "areas": list(areas.keys())
        }
    },
    "areas": {},
    "wards": {}
}

import random
zone_types = ["residential", "mixed", "commercial", "arterial", "it_corridor"]

for area, ward_nums in areas.items():
    ward_ids = [f"ward_{str(n).zfill(3)}" for n in ward_nums]
    registry["areas"][area] = {"wards": ward_ids}
    
    for i, w_num in enumerate(ward_nums):
        w_id = f"ward_{str(w_num).zfill(3)}"
        neighbors = []
        if i > 0: neighbors.append(ward_nums[i-1])
        if i < len(ward_nums) - 1: neighbors.append(ward_nums[i+1])
        
        ward_entry = {
            "label": f"{area.replace('_', ' ')} Ward {w_num}",
            "ward_number": w_num,
            "zone_type": random.choice(zone_types),
            "parent_area": area,
            "parent_region": "South_Corridor",
            "neighbors": neighbors,
            "congestion_prior": random.choice(["low", "medium", "high"]),
            "priority_level": random.randint(1, 3),
            "hospital_sensitive": random.choice([True, False])
        }
        if relation_mapping.get(w_num):
            ward_entry["osm_relation_id"] = relation_mapping[w_num]
        else:
            ward_entry["bbox"] = {"south": 12.92, "west": 77.58, "north": 12.94, "east": 77.60}
            
        registry["wards"][w_id] = ward_entry

with registry_path.open("w", encoding="utf-8") as f:
    json.dump(registry, f, indent=2)

print("Final 4 areas registry successfully updated!")
