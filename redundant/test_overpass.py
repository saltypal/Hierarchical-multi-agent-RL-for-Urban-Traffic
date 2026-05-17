import urllib.request, urllib.parse, json

query = """[out:json][timeout:120];
relation(19883372);map_to_area->.ward_area;
way(area.ward_area)["highway"];
out count;"""

encoded = urllib.parse.urlencode({'data': query}).encode('utf-8')
req = urllib.request.Request('https://overpass-api.de/api/interpreter', data=encoded, method='POST', headers={'User-Agent': 'HMRL-Traffic/1.0'})

try:
    with urllib.request.urlopen(req, timeout=60) as res:
        data = json.loads(res.read())
        print(data)
except Exception as e:
    print('Error:', e)
