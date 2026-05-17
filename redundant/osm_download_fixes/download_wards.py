import sys, time
from pathlib import Path
sys.path.insert(0, '.')
import src.preprocessing.osm_fetcher as fetcher

# Add 5 second delay to avoid HTTP 429 Too Many Requests
fetcher.REQUEST_DELAY_SECONDS = 5

print("Starting ward downloads. Please wait...")
results = fetcher.fetch_all_wards(Path('.'), force=True)

print('\n--- Download Summary ---')
success = 0
for r in results:
    if r['status'] == 'ok':
        success += 1
    else:
        print(f"FAILED: {r['ward_id']} - {r['status']}")
print(f'Successfully downloaded {success} out of {len(results)} wards.')
