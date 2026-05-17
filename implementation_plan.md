# HMRL — Complete Implementation Plan (FINAL)

## Architecture

```
City   → NetworkX MCMF graph optimizer     (no training)
Area   → GNN traffic pressure forecaster   (supervised, PyTorch)
Ward   → PPO RL tactical controller        (SB3)
```

Multi-timescale: Ward=1s, Area=30s, City=120s/event.
Bidirectional: states flow up, predictions/constraints flow down.
GNN is a forecaster (not controller). Its predictions modify ward obs + reward.

---

## PHASE 1: Data Engineering & Maps

### 1a. Ward Registry
**File**: `configs/hierarchy/ward_registry.json`

Maps every ward ID to metadata. Bounding boxes enable Overpass API download.
```json
{"ward_001": {
    "label": "Basavanagudi W1", "zone_type": "residential",
    "parent_area": "Basavanagudi", "neighbors": [2, 3],
    "congestion_prior": "medium", "hospital_sensitive": false,
    "bbox": {"south": 12.935, "west": 77.565, "north": 12.945, "east": 77.575}
}}
```

### 1b. OSM Download
**File**: `src/preprocessing/osm_fetcher.py`
- `fetch_ward_osm(ward_id, bbox)` → calls Overpass API, saves `maps/raw_osm/ward_001.osm`
- `fetch_all_wards(registry)` → batch download

### 1c. Ward Processing
**File**: `src/preprocessing/ward_processor.py`
- `convert_osm_to_net(ward_id)` → runs `netconvert`, produces `maps/processed/ward_001/ward.net.xml`
- `extract_ward_metadata(ward_id)` → parses net.xml → `metadata.json` (edge count, junctions, signals, lane stats, road classifications). Reuses `summarize_net_xml()` from existing `map_pipeline.py`.
- `detect_ward_boundaries(ward_id)` → finds dead-end/perimeter edges → `boundaries.json` (ingress edges, egress edges, spawn candidates)
- `process_all_wards()` → batch orchestrate

### 1d. Directory Structure
```
maps/
    raw_osm/ward_001.osm, ward_002.osm, ...
    processed/ward_001/{ward.net.xml, metadata.json, boundaries.json}
    stitched/area/
    od_matrices/
    _legacy/              ← old koramangala/, bellandur/, etc.
```

### 1e. Notebook
`notebooks/01_osm_preprocessing.ipynb` — download, compile, validate, display summary table.

---

## PHASE 2: Traffic Generation

### 2a. Vehicle Profiles
**File**: `configs/vehicle_profiles.py`
7 SUMO vType definitions:
```python
VEHICLE_PROFILES = {
    "normal_car":  {"vClass":"passenger", "maxSpeed":16.67, "accel":2.6, "sigma":0.5, "minGap":2.5, "length":5.0},
    "aggressive":  {"vClass":"passenger", "maxSpeed":22.22, "accel":4.0, "sigma":0.9, "minGap":1.0, "length":5.0},
    "slow_driver": {"vClass":"passenger", "maxSpeed":11.11, "accel":1.5, "sigma":0.2, "minGap":4.0, "length":5.0},
    "bmtc_bus":    {"vClass":"bus",       "maxSpeed":11.11, "accel":1.0, "sigma":0.3, "minGap":3.0, "length":12.0},
    "truck":       {"vClass":"truck",     "maxSpeed":13.89, "accel":0.8, "sigma":0.3, "minGap":3.5, "length":10.0},
    "ambulance":   {"vClass":"emergency", "maxSpeed":22.22, "accel":3.5, "sigma":0.1, "minGap":1.5, "length":6.5},
    "govt_convoy": {"vClass":"authority", "maxSpeed":16.67, "accel":2.0, "sigma":0.1, "minGap":5.0, "length":5.5},
}
```

### 2b. Traffic Profiles
**File**: `configs/traffic_profiles.py`
Zone-semantic spawn priors:
```python
ZONE_PROFILES = {
    "commercial":         {"spawn_intensity": 1.8, "peak_multiplier": 2.5, "reward_bias": "throughput"},
    "residential":        {"spawn_intensity": 1.0, "peak_multiplier": 1.5, "reward_bias": "fairness"},
    "hospital_sensitive": {"spawn_intensity": 0.8, "peak_multiplier": 1.3, "reward_bias": "emergency"},
    # ... mixed, arterial, bottleneck, it_corridor
}
```

### 2c. Scenarios
**File**: `configs/scenarios.py`
10 scenarios: normal, peak_congestion, ambulance_emergency, vip_convoy, chaos_mode, traffic_surge, breakdown, blocked_road, asymmetric_overload, low_baseline. Each defines vehicle mix ratios, spawn multipliers, disturbance parameters.

### 2d. Traffic Generator
**File**: `src/traffic_generator.py`
```python
class TrafficGenerator:
    def generate_ward_routes(ward_id, scenario_id) -> Path:
        """Generate .rou.xml with vTypes from vehicle_profiles, OD from boundaries.json"""
    def generate_ward_sumocfg(ward_id) -> Path:
        """Create .sumocfg pointing to ward.net.xml + routes"""
    def step(self, sumo_env):
        """Per-tick: inject dynamic vehicles, breakdowns, incidents.
        Uses sumo_env API. NEVER imports traci."""
```

### 2e. Notebook
`notebooks/02_traffic_allocation.ipynb` — generate routes per ward, visualize vehicle mix and demand.

---

## PHASE 3: SUMO Abstraction + Topology

### 3a. SUMO Environment
**File**: `src/sumo_env.py` — **SOLE TraCI interface**
```python
class SumoEnv:
    # Lifecycle
    def start(self, config_path, gui=False): ...
    def stop(self): ...
    def step(self): ...
    def reset(self): ...
    # Vehicle control
    def add_vehicle(self, veh_id, route_id, vtype): ...
    def remove_vehicle(self, veh_id): ...
    def reroute_vehicle(self, veh_id): ...
    def set_vehicle_speed(self, veh_id, speed): ...
    # State queries
    def get_vehicle_ids(self) -> list[str]: ...
    def get_vehicle_speed(self, veh_id) -> float: ...
    def get_vehicle_waiting_time(self, veh_id) -> float: ...
    def get_edge_halting_count(self, edge_id) -> int: ...
    def get_arrived_count(self) -> int: ...
    # Ward-level aggregated state
    def get_ward_summary(self, ward_edge_ids) -> dict: ...
```

No other file imports `traci`.

### 3b. Topology
**File**: `src/topology.py`
```python
class Topology:
    def __init__(self, project_root):
        """Loads ward_registry.json, blr_regions.json, builds adjacency"""
    def build_ward_graph(self, area_id) -> nx.Graph
    def get_ward_neighbors(self, ward_id) -> list[str]
    def get_adjacency_matrix(self, area_id) -> np.ndarray  # for GNN
    def stitch_ward_maps(self, ward_ids) -> Path  # merged net.xml
    def get_edge_owner(self, edge_id) -> str  # ward_id
```

---

## PHASE 4: Ward RL Training

### 4a. Reward Calculator
**File**: `src/reward.py`
```python
class WardRewardCalculator:
    def __init__(self, zone_type, gnn_pressure=0.0):
        """Zone modulation: commercial→throughput, residential→fairness, hospital→ambulance"""
    
    def compute(self, sumo_env, ward_edges) -> float:
        """
        R = + w1*throughput + w2*trip_completion + w3*avg_speed
            + w4*ambulance_progress + w5*congestion_reduction
            - p1*wait_time - p2*queue_length - p3*spillback
            - p4*deadlock - p5*incident_duration - p6*ambulance_blocking
            - p7*unfairness
            - λ * gnn_pressure * outflow_to_neighbors   ← GNN penalty
        """
```

### 4b. Ward Adapter Refactor
**File**: `src/rl/sb3_ward_adapter.py` (MODIFY)

Key changes:
- Remove ALL `traci.*` calls → use `self.sumo_env: SumoEnv`
- Observation expanded to 12 dims: 10 local + `predicted_pressure` + `city_cap`
- During training: synthetic pressure injection (50% zero, 30% medium, 20% high)
- Uses `WardRewardCalculator` with GNN pressure penalty
- **Logs ward state snapshots every 30 ticks for GNN dataset** (free, no extra simulation)

```python
# Observation space
self.observation_space = spaces.Box(low=0, high=np.inf, shape=(12,), dtype=np.float32)

def _observation(self):
    local = self._local_state_10_dims()       # from sumo_env
    if self.training_mode:
        pressure = random_synthetic_pressure()  # curriculum learning
        cap = random.uniform(0.5, 1.0)
    else:
        pressure = self.gnn_prediction         # from area GNN
        cap = self.city_cap                    # from city solver
    return np.concatenate([local, [pressure, cap]])
```

### 4c. Multi-Algorithm Training
**File**: `src/rl/train.py` (NEW, replaces train_ppo.py)
```python
def train_ward(ward_id, algorithm="ppo", total_timesteps=10000, gui=False):
    """
    Trains PPO/A2C/DQN/SAC. During training:
    1. Synthetic pressure injection in obs
    2. Logs ward snapshots every 30 ticks → gnn_training_data.pt
    3. Saves model as models/{algorithm}/ward_{id}/model.pt + config.json
    """
```

### 4d. Notebook
`notebooks/04_ward_training.ipynb` — train wards, compare algorithms, plot reward curves. GNN dataset auto-generated.

---

## PHASE 5: GNN Area Forecaster + City Graph

### 5a. GNN Architecture
**File**: `src/controllers/area_controller.py`

```python
class WardPressureGNN(nn.Module):
    """2-layer Graph Convolutional Network. ~30 lines of PyTorch."""
    def __init__(self, in_features=7, hidden=32):
        self.W1 = nn.Linear(in_features, hidden)
        self.W2 = nn.Linear(hidden, 1)
    
    def forward(self, X, A_hat):
        # X: [N_wards, 7] — node features
        # A_hat: [N_wards, N_wards] — normalized adjacency (from topology)
        H = F.relu(A_hat @ self.W1(X))
        return torch.sigmoid(A_hat @ self.W2(H)).squeeze()  # [N_wards] pressure ∈ [0,1]

class AreaForecaster:
    def __init__(self, area_id, topology):
        self.A_hat = topology.get_adjacency_matrix(area_id)  # fixed
        self.model = WardPressureGNN()
    
    def train_offline(self, dataset_path):
        """Load gnn_training_data.pt, train MSE, save gnn_model.pt"""
    
    def predict(self, ward_states, city_cap) -> dict[str, float]:
        """Returns predicted pressure per ward"""
```

**GNN node features per ward** (7 dims):
congestion, Δcongestion, queue, Δqueue, avg_speed, throughput, incident_flag

**GNN training data** (auto-collected during Phase 4):
- X = ward graph state at time t
- Y = actual congestion per ward at time t+30s
- MSE loss, trains in minutes

### 5b. City Graph Optimizer
**File**: `src/controllers/city_controller.py`
```python
class CityController:
    def __init__(self, topology):
        """Build macro graph: nodes=areas, edges=corridors"""
    
    def solve(self, area_summaries) -> dict[str, float]:
        """
        NetworkX min-cost max-flow.
        Input: per-area {avg_congestion, throughput, incidents}
        Output: per-area inflow capacity cap ∈ [0.0, 1.0]
        """
```

Event triggers (run immediately instead of waiting 120s):
- Ambulance detected
- Any area congestion > 0.85
- Incident/blocked road
- VIP convoy injected

### 5c. Ward Inference Wrapper
**File**: `src/controllers/ward_agent.py`
```python
class WardAgent:
    def __init__(self, ward_id, model_path):
        """Load .pt state dict, build policy network"""
    def get_action(self, observation) -> int:
        """Returns WardAction index"""
```

---

## PHASE 6: Runtime Integration & Inference

### 6a. Runtime Orchestrator
**File**: `src/runtime.py` (REFACTOR — remove all training code)

```python
def run_simulation(scope, identifier, gui=False, scenario_id="normal"):
    topology = Topology(project_root)
    sumo_env = SumoEnv()
    traffic_gen = TrafficGenerator(topology, scenario_id)
    
    # Load trained models
    ward_agents = {wid: WardAgent(wid, f"models/ppo/{wid}/model.pt") for wid in ward_ids}
    area_forecasters = {aid: AreaForecaster(aid, topology) for aid in area_ids}
    city_controller = CityController(topology)
    
    # State buffers
    city_caps = {aid: 1.0 for aid in area_ids}      # initial: no restriction
    gnn_predictions = {wid: 0.0 for wid in ward_ids} # initial: no pressure
    
    sumo_env.start(sumocfg_path, gui=gui)
    
    for tick in range(max_ticks):
        traffic_gen.step(sumo_env)
        sumo_env.step()
        
        # === WARD: every tick ===
        for ward_id, agent in ward_agents.items():
            obs = build_ward_obs(sumo_env, ward_id, 
                                 gnn_predictions[ward_id], 
                                 city_caps[ward_to_area[ward_id]])
            action = agent.get_action(obs)
            apply_ward_action(sumo_env, ward_id, action)
        
        # === AREA: every 30 ticks ===
        if tick % 30 == 0:
            for area_id, forecaster in area_forecasters.items():
                ward_states = collect_ward_summaries(sumo_env, area_id)
                predictions = forecaster.predict(ward_states, city_caps[area_id])
                for wid, pressure in predictions.items():
                    gnn_predictions[wid] = pressure
        
        # === CITY: every 120 ticks or event ===
        if tick % 120 == 0 or detect_emergency(sumo_env):
            area_summaries = collect_area_summaries(sumo_env, area_ids)
            city_caps = city_controller.solve(area_summaries)
    
    sumo_env.stop()
```

### 6b. CLI
**File**: `hmrl.py` (MODIFY)
```bash
# Inference/simulation
python hmrl.py --scope area --id HSR_Layout --mode simulate --gui
python hmrl.py --scope ward --id ward_001 --mode simulate --gui

# Data generation mode (for logging)
python hmrl.py --scope area --id HSR_Layout --mode generate_data
```

Training happens ONLY in notebooks, not through hmrl.py.

---

## Complete File Manifest

### KEEP (no changes)
`src/rl/ward_actions.py`, `src/intelligence/directives.py`, `src/evaluation/metrics.py`, `configs/hierarchy/blr_regions.json`, `configs/preprocessing/metadata_schema.json`, `configs/preprocessing/area_road_config.json`

### MOVE
`Traffic/` → `Traffic_legacy/`
`maps/{koramangala,bellandur,hsr_layout,...}/` → `maps/_legacy/`

### CREATE
| File | Phase |
|------|-------|
| `configs/hierarchy/ward_registry.json` | 1 |
| `configs/traffic_profiles.py` | 2 |
| `configs/vehicle_profiles.py` | 2 |
| `configs/scenarios.py` | 2 |
| `src/preprocessing/osm_fetcher.py` | 1 |
| `src/preprocessing/ward_processor.py` | 1 |
| `src/sumo_env.py` | 3 |
| `src/topology.py` | 3 |
| `src/reward.py` | 4 |
| `src/traffic_generator.py` | 2 |
| `src/rl/train.py` | 4 |
| `src/controllers/__init__.py` | 5 |
| `src/controllers/area_controller.py` | 5 |
| `src/controllers/city_controller.py` | 5 |
| `src/controllers/ward_agent.py` | 4 |
| `notebooks/01_osm_preprocessing.ipynb` | 1 |
| `notebooks/02_traffic_allocation.ipynb` | 2 |
| `notebooks/04_ward_training.ipynb` | 4 |
| `notebooks/05_area_coordination.ipynb` | 5 |

### REFACTOR
`src/rl/sb3_ward_adapter.py` — remove traci, expand obs to 12 dims, add synthetic pressure
`src/runtime.py` — remove training code, add bidirectional multi-timescale loop
`hmrl.py` — simulation + data modes only
`src/hierarchy.py` — add ward_registry loader
`requirements.txt` — add networkx, requests

### DELETE
`src/rl/train_ppo.py` — replaced by `src/rl/train.py`

---

## Training Pipeline (One Pass)

```
Step 1: Train ward PPO (synthetic pressure injection)
        → auto-collects GNN training data during episodes
        → saves models/ppo/ward_001/model.pt
        → saves models/gnn/training_data.pt

Step 2: Train GNN offline on collected data (MSE, minutes)
        → saves models/gnn/area_model.pt

Step 3: Optional: fine-tune wards with live GNN predictions

Step 4: City solver — no training needed, runs at inference
```

## Model Storage
```
models/
    ppo/ward_001/{model.pt, config.json, training_log.csv}
    a2c/ward_001/...
    dqn/ward_001/...
    gnn/{area_model.pt, training_data.pt, config.json}
```
