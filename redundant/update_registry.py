import json
import random
from pathlib import Path

registry_path = Path("configs/hierarchy/ward_registry.json")

areas_map = {
    "HSR_Layout": [70, 71, 72],
    "BTM_Layout": [18, 19, 20, 21, 22, 23, 24],
    "Jayanagar": [7, 8, 9, 10, 11, 12, 13, 14],
    "Chickpet": [32, 33, 34],
    "Basavanagudi": [1, 2, 3, 4, 5, 6]
}

area_to_region = {
    "HSR_Layout": "South_BLR",
    "BTM_Layout": "South_BLR",
    "Jayanagar": "South_BLR",
    "Chickpet": "Central_BLR",
    "Basavanagudi": "North_BLR"
}

# Base schema structure
registry = {
    "schema_version": "0.3.0",
    "description": "BBMP ward metadata registry. Ward data is fetched via an Overpass QL boundary query.",
    "regions": {
        "South_BLR": {
            "areas": ["HSR_Layout", "BTM_Layout", "Jayanagar"]
        },
        "Central_BLR": {
            "areas": ["Chickpet"]
        },
        "North_BLR": {
            "areas": ["Basavanagudi"]
        }
    },
    "areas": {},
    "wards": {}
}

# Zone types for realistic variation
zone_types = ["residential", "mixed", "commercial", "arterial", "it_corridor"]

for area, ward_nums in areas_map.items():
    ward_ids = [f"ward_{str(n).zfill(3)}" for n in ward_nums]
    registry["areas"][area] = {"wards": ward_ids}
    
    region = area_to_region[area]
    
    for i, w_num in enumerate(ward_nums):
        w_id = f"ward_{str(w_num).zfill(3)}"
        
        # Determine neighbors roughly (adjacent in the list)
        neighbors = []
        if i > 0:
            neighbors.append(ward_nums[i-1])
        if i < len(ward_nums) - 1:
            neighbors.append(ward_nums[i+1])
            
        registry["wards"][w_id] = {
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

# Write out the updated registry
with registry_path.open("w", encoding="utf-8") as f:
    json.dump(registry, f, indent=2)

print(f"Updated {registry_path} with new ward mappings!")
