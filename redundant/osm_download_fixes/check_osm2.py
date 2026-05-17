import urllib.request, urllib.parse, json
query = '[out:json];area["name"="Bengaluru"]->.blr;relation(area.blr)["boundary"="administrative"]["admin_level"="10"]["ward"~"^(1|2|3|4|5|6)$"];out tags;'
encoded = urllib.parse.urlencode({'data': query}).encode('utf-8')
req = urllib.request.Request('https://overpass-api.de/api/interpreter', data=encoded, method='POST', headers={'User-Agent': 'HMRL-Traffic/1.0'})
try:
    with urllib.request.urlopen(req) as res:
        data = json.loads(res.read())
        print('Wards 1-6 in OSM:')
        elements = data.get('elements', [])
        for e in elements:
            print(f"  - Ward {e.get('tags', {}).get('ward')}: {e.get('tags', {}).get('name')}")
except Exception as e:
    print('Error:', e)
