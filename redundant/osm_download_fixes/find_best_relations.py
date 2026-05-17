import urllib.request, urllib.parse, json, math

areas_map = {
    "HSR_Layout": {"wards": [70, 71, 72], "lat": 12.91, "lon": 77.63},
    "BTM_Layout": {"wards": [18, 19, 20, 21, 22, 23, 24], "lat": 12.91, "lon": 77.61},
    "Jayanagar": {"wards": [7, 8, 9, 10, 11, 12, 13, 14], "lat": 12.93, "lon": 77.58},
    "Chickpet": {"wards": [32, 33, 34], "lat": 12.97, "lon": 77.57},
    "Basavanagudi": {"wards": [1, 2, 3, 4, 5, 6], "lat": 13.10, "lon": 77.59} # North BLR (Yelahanka)
}

def distance(lat1, lon1, lat2, lon2):
    return math.sqrt((lat1-lat2)**2 + (lon1-lon2)**2)

best_relations = {}

for area, data in areas_map.items():
    print(f"\nProcessing {area}...")
    ward_nums = data["wards"]
    target_str = "|".join(map(str, ward_nums))
    
    query = f'[out:json];area["name"="Bengaluru"]->.blr;relation(area.blr)["boundary"="administrative"]["admin_level"="10"]["ward"~"^({target_str})$"];out center;'
    encoded = urllib.parse.urlencode({'data': query}).encode('utf-8')
    req = urllib.request.Request('https://overpass-api.de/api/interpreter', data=encoded, method='POST', headers={'User-Agent': 'HMRL-Traffic/1.0'})
    
    try:
        with urllib.request.urlopen(req) as res:
            res_data = json.loads(res.read())
            
            by_ward = {}
            for e in res_data.get('elements', []):
                w = int(e.get('tags', {}).get('ward'))
                if w not in by_ward:
                    by_ward[w] = []
                by_ward[w].append({
                    "id": e["id"], 
                    "name": e.get("tags", {}).get("name"),
                    "lat": e.get("center", {}).get("lat", 0),
                    "lon": e.get("center", {}).get("lon", 0)
                })
                
            for w in ward_nums:
                candidates = by_ward.get(w, [])
                if not candidates:
                    print(f"  Ward {w}: Not found!")
                    continue
                
                # Find closest
                closest = min(candidates, key=lambda c: distance(data["lat"], data["lon"], c["lat"], c["lon"]))
                print(f"  Ward {w} -> Selected {closest['name']} (ID: {closest['id']}) at dist {distance(data['lat'], data['lon'], closest['lat'], closest['lon']):.3f}")
                best_relations[f"ward_{str(w).zfill(3)}"] = closest['id']
                
    except Exception as e:
        print('Error:', e)

with open("best_relations.json", "w") as f:
    json.dump(best_relations, f, indent=2)
