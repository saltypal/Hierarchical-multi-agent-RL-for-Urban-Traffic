import sys
from pathlib import Path

PROJECT_ROOT = Path("d:/Bunker/BaseCamp/Hierarchical-multi-agent-RL-for-Urban-Traffic")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rl.sb3_ward_adapter import StableBaselinesWardEnv, WardAdapterConfig
import json

registry_path = PROJECT_ROOT / 'configs' / 'hierarchy' / 'ward_registry.json'
with registry_path.open('r') as f:
    registry = json.load(f)
    WARD_IDS = list(registry.get('wards', {}).keys())

config = WardAdapterConfig(
    ward_id=WARD_IDS,
    project_root=str(PROJECT_ROOT),
    scenario_id=["normal", "peak"],
    gui=False
)

env = StableBaselinesWardEnv(config)
print("Testing reset()...")
obs, info = env.reset()
print(f"Loaded ward: {env.current_ward_id}")

for i in range(5):
    print(f"Step {i}...")
    obs, reward, terminated, truncated, info = env.step(0)
    if terminated or truncated:
        break

print("Testing second reset() to see if it cleanly switches wards...")
obs, info = env.reset()
print(f"Loaded ward: {env.current_ward_id}")

env.close()
print("Test successful!")
