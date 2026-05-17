import urllib.request, urllib.parse, json

# Bounding box for South Bengaluru covering our target areas
query = """[out:json];
relation["boundary"="administrative"]["admin_level"="10"](12.87, 77.52, 12.98, 77.65);
out center tags;"""

encoded = urllib.parse.urlencode({'data': query}).encode('utf-8')
req = urllib.request.Request('https://overpass-api.de/api/interpreter', data=encoded, method='POST', headers={'User-Agent': 'HMRL-Traffic/1.0'})

target_wards = {
    "Basavanagudi": [103, 104, 105, 106, 108],
    "Jayanagar": [7, 8, 9, 10, 11, 12, 13],
    "BTM_Layout": [18, 19, 20, 21, 22, 23],
    "Padmanabhanagar": [1, 2, 3, 4, 14, 15, 16],
    "Bommanahalli": [60, 62, 63, 64, 65]
}

flat_targets = set()
for w_list in target_wards.values():
    flat_targets.update(w_list)

try:
    print("Fetching OSM relations in South Bengaluru...")
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
                found_wards[w] = {
                    'id': e['id'],
                    'name': e.get('tags', {}).get('name', 'Unknown')
                }
                
        print("\nMatched Wards:")
        for area, w_list in target_wards.items():
            print(f"\n{area}:")
            for w in w_list:
                if w in found_wards:
                    print(f"  Ward {w} -> {found_wards[w]['name']} (ID: {found_wards[w]['id']})")
                else:
                    print(f"  Ward {w} -> NOT FOUND IN OSM BBOX")

except Exception as e:
    print('Error:', e)
