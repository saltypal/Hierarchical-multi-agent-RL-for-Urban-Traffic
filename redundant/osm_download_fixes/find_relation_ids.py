import urllib.request, urllib.parse, json

target_wards = list(range(70, 73)) + list(range(18, 25)) + list(range(7, 15)) + list(range(32, 35)) + list(range(1, 7)) + list(range(100, 108))
target_str = "|".join(map(str, target_wards))

query = f'[out:json];area["name"="Bengaluru"]->.blr;relation(area.blr)["boundary"="administrative"]["admin_level"="10"]["ward"~"^({target_str})$"];out tags;'
encoded = urllib.parse.urlencode({'data': query}).encode('utf-8')
req = urllib.request.Request('https://overpass-api.de/api/interpreter', data=encoded, method='POST', headers={'User-Agent': 'HMRL-Traffic/1.0'})

try:
    with urllib.request.urlopen(req) as res:
        data = json.loads(res.read())
        
        # Group by ward number
        by_ward = {}
        for e in data.get('elements', []):
            w = int(e.get('tags', {}).get('ward'))
            if w not in by_ward:
                by_ward[w] = []
            by_ward[w].append({"id": e["id"], "name": e.get("tags", {}).get("name")})
            
        for w in sorted(by_ward.keys()):
            print(f"Ward {w}:")
            for item in by_ward[w]:
                print(f"  - {item['name']} (ID: {item['id']})")
except Exception as e:
    print('Error:', e)
