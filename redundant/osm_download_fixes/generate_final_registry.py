import json
from pathlib import Path

registry_path = Path("configs/hierarchy/ward_registry.json")

blr_regions = {
    "South_Corridor": {
        "Basavanagudi": [106, 103, 105],
        "Jayanagar": [7, 8, 9, 10, 11],
        "BTM_Layout": [18, 19, 20, 21],
        "Koramangala_Adugodi": [22, 23, 24, 27],
        "HSR_Layout": [30, 70, 71, 72]
    }
}

relation_mapping = {
    106: 19883372, 103: 19883390, 105: 19883388,
    7: 19884598, 8: 19884593, 9: 19884551, 10: None, 11: 19883292,
    18: 19884537, 19: 19883486, 20: 19884562, 21: 19884538,
    22: 19883464, 23: 19883457, 24: 19884613, 27: 19883454,
    30: 19883281, 70: 19883476, 71: 19883431, 72: 19883285
}

registry = {
    "schema_version": "0.4.0",
    "description": "BBMP ward metadata registry with exact OSM relation mapping to handle overlapping ward numbers in OpenStreetMap.",
    "regions": {
        "South_Corridor": {
            "areas": list(blr_regions["South_Corridor"].keys())
        }
    },
    "areas": {},
    "wards": {}
}

import random
zone_types = ["residential", "mixed", "commercial", "arterial", "it_corridor"]

for region, areas in blr_regions.items():
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
                "parent_region": region,
                "neighbors": neighbors,
                "congestion_prior": random.choice(["low", "medium", "high"]),
                "priority_level": random.randint(1, 3),
                "hospital_sensitive": random.choice([True, False])
            }
            if relation_mapping.get(w_num):
                ward_entry["osm_relation_id"] = relation_mapping[w_num]
            else:
                # Default bbox for missing OSM relation (e.g., Ward 10 in Jayanagar)
                ward_entry["bbox"] = {"south": 12.92, "west": 77.58, "north": 12.94, "east": 77.60}
                
            registry["wards"][w_id] = ward_entry

with registry_path.open("w", encoding="utf-8") as f:
    json.dump(registry, f, indent=2)

print("Registry successfully updated!")
