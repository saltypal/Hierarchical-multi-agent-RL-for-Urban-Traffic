import urllib.request, urllib.parse, json

query = """[out:json];
relation["boundary"="administrative"]["admin_level"="10"]["ward"~"^(14|15|17)$"](12.85, 77.55, 12.96, 77.65);
out center tags;"""

encoded = urllib.parse.urlencode({'data': query}).encode('utf-8')
req = urllib.request.Request('https://overpass-api.de/api/interpreter', data=encoded, method='POST', headers={'User-Agent': 'HMRL-Traffic/1.0'})

try:
    with urllib.request.urlopen(req, timeout=60) as res:
        data = json.loads(res.read())
        for e in data.get('elements', []):
            print(f"Ward {e['tags'].get('ward')}: {e['tags'].get('name')} (ID: {e['id']})")
except Exception as e:
    print('Error:', e)
