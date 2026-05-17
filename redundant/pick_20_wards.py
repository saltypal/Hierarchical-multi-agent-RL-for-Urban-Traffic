import urllib.request, urllib.parse, json

# Bounding box for Jayanagar/BTM/Basavanagudi/Padmanabhanagar/Bommanahalli
query = """[out:json];
relation["boundary"="administrative"]["admin_level"="10"](12.89, 77.56, 12.95, 77.62);
out center tags;"""

encoded = urllib.parse.urlencode({'data': query}).encode('utf-8')
req = urllib.request.Request('https://overpass-api.de/api/interpreter', data=encoded, method='POST', headers={'User-Agent': 'HMRL-Traffic/1.0'})

try:
    print("Fetching OSM relations in South Bengaluru...")
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
        
        wards.sort(key=lambda w: (w['lat'], w['lon']))
        for w in wards:
            print(f"Ward {w['ward']:3} | ID: {w['id']} | Name: {w['name']}")
            
except Exception as e:
    print('Error:', e)
