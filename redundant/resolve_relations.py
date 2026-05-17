import urllib.request, urllib.parse, json, math

blr_regions = {
    "South_Corridor": {
        "Basavanagudi": [106, 103, 105],
        "Jayanagar": [7, 8, 9, 10, 11],
        "BTM_Layout": [18, 19, 20, 21],
        "Koramangala_Adugodi": [22, 23, 24, 27],
        "HSR_Layout": [30, 70, 71, 72]
    }
}

# The bounding box around South Corridor guarantees we get the right ward (not the duplicated ones in North BLR)
# Bbox: South 12.89, West 77.55, North 12.95, East 77.65
query = """[out:json];
relation["boundary"="administrative"]["admin_level"="10"](12.88, 77.55, 12.96, 77.66);
out center tags;"""

encoded = urllib.parse.urlencode({'data': query}).encode('utf-8')
req = urllib.request.Request('https://overpass-api.de/api/interpreter', data=encoded, method='POST', headers={'User-Agent': 'HMRL-Traffic/1.0'})

flat_targets = set()
for area, wards in blr_regions["South_Corridor"].items():
    flat_targets.update(wards)

try:
    print("Fetching OSM relations in South Corridor Bbox...")
    with urllib.request.urlopen(req, timeout=60) as res:
        data = json.loads(res.read())
        
        found_wards = {}
        for e in data.get('elements', []):
            ward_str = e.get('tags', {}).get('ward')
            if not ward_str:
                continue
            try:
                w = int(ward_str)
            except ValueError:
                continue
                
            if w in flat_targets:
                # If there are duplicates in the SAME bbox, we want the one closest to the area cluster.
                # But typically a tight bbox resolves the duplicates.
                if w not in found_wards:
                    found_wards[w] = []
                found_wards[w].append({
                    'id': e['id'],
                    'name': e.get('tags', {}).get('name', 'Unknown')
                })
                
        print("\nResolved Wards:")
        final_mapping = {}
        for area, w_list in blr_regions["South_Corridor"].items():
            print(f"\n{area}:")
            for w in w_list:
                cands = found_wards.get(w, [])
                if not cands:
                    print(f"  Ward {w} -> NOT FOUND")
                elif len(cands) == 1:
                    print(f"  Ward {w} -> {cands[0]['name']} (ID: {cands[0]['id']})")
                    final_mapping[w] = cands[0]
                else:
                    print(f"  Ward {w} -> MULTIPLE FOUND: {[c['name'] for c in cands]}. Picking first.")
                    print(f"    Selected: {cands[0]['name']} (ID: {cands[0]['id']})")
                    final_mapping[w] = cands[0]
                    
        with open("resolved_wards.json", "w") as f:
            json.dump(final_mapping, f, indent=2)

except Exception as e:
    print('Error:', e)
