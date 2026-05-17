import urllib.request, urllib.parse, json
import math

# Bounding box for South Bengaluru (Jayanagar, BTM, Koramangala, HSR, Basavanagudi)
# Roughly 12.90 to 12.95 N, 77.56 to 77.64 E
query = """[out:json];
relation["boundary"="administrative"]["admin_level"="10"](12.89, 77.55, 12.96, 77.65);
out center;"""

print("Fetching wards in South Bengaluru bounding box...")
encoded = urllib.parse.urlencode({'data': query}).encode('utf-8')
req = urllib.request.Request('https://overpass-api.de/api/interpreter', data=encoded, method='POST', headers={'User-Agent': 'HMRL-Traffic/1.0'})

try:
    with urllib.request.urlopen(req, timeout=60) as res:
        data = json.loads(res.read())
        
        wards = []
        for e in data.get('elements', []):
            if e['type'] == 'relation':
                wards.append({
                    'id': e['id'],
                    'name': e.get('tags', {}).get('name', 'Unknown'),
                    'ward': e.get('tags', {}).get('ward', '0'),
                    'lat': e.get('center', {}).get('lat', 0),
                    'lon': e.get('center', {}).get('lon', 0)
                })
        
        print(f"Found {len(wards)} wards in the bounding box.")
        
        # Sort wards roughly by longitude (West to East) and then latitude
        wards.sort(key=lambda w: (w['lon'], w['lat']))
        
        for w in wards:
            print(f"ID: {w['id']}, Ward: {w['ward']}, Name: {w['name']}, Pos: ({w['lat']:.3f}, {w['lon']:.3f})")
            
        with open("south_wards.json", "w") as f:
            json.dump(wards, f, indent=2)

except Exception as e:
    print('Error:', e)
