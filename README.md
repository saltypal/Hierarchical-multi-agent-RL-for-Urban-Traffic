# Hierarchical Multi-Agent Reinforcement Learning (HMARL) for Urban Traffic Control

An advanced, scalable, and distributed traffic management framework that coordinates micro-level traffic signal adapters, meso-level graph neural network forecasters, and macro-level city route capacity balancers over realistic OpenStreetMap (OSM) road networks of Bengaluru, simulated inside Eclipse SUMO (Simulation of Urban MObility) and controlled via TraCI.

---

## System Architecture Overview

The core innovation of this system is resolving the curse of dimensionality in large-scale urban networks by utilizing a **three-tier hierarchical multi-agent architecture** that operates across progressive spatial and temporal scales:

```
                  ┌──────────────────────────────────────┐
                  │      City-Level Macro Controller     │  ◄── Ticks every 120s
                  │       (Linear Capacity caps solver)  │      (L3 - Macroscopic)
                  └──────────────────┬───────────────────┘
                                     │
                     Capacity Caps   │   Aggregated Saturation
                         (Top-down)  ▼   (Bottom-up)
                  ┌──────────────────────────────────────┐
                  │      Area-Level Meso Forecaster      │  ◄── Ticks every 60s
                  │        (STGCN / GCN Predictor)       │      (L2 - Mesoscopic)
                  └──────────────────┬───────────────────┘
                                     │
                  Pressure Signals   │   Aggregated Queues
                         (Top-down)  ▼   (Bottom-up)
                  ┌──────────────────────────────────────┐
                  │     Ward-Level Micro RL Agents       │  ◄── Ticks every 30s
                  │      (Discrete PPO/DQN Adapters)     │      (L1 - Microscopic)
                  └──────────────────┬───────────────────┘
                                     │
                       Semantic      │   Raw Vehicle States
                        Actions      ▼   (Every 1s)
                  ┌──────────────────────────────────────┐
                  │       Eclipse SUMO Simulator         │  ◄── Core simulation step
                  │         (Bangalore Wards)            │      (Continuous)
                  └──────────────────────────────────────┘
```

---

## The 3-Tier Hierarchical Layers

### 1. Ward Layer (L1 - Microscopic)
- **Role**: Directly controls traffic flow at localized ward regions and boundary intersections.
- **Agent Type**: Discrete [Proximal Policy Optimization (PPO)](file:///d:/Bunker/BaseCamp/Hierarchical-multi-agent-RL-for-Urban-Traffic/models/ppo) or [Deep Q-Network (DQN)](file:///d:/Bunker/BaseCamp/Hierarchical-multi-agent-RL-for-Urban-Traffic/models/dqn) agents.
- **Control Interval**: Action updates are evaluated every **30 simulation seconds** (Ward Tick).
- **Observation Space**: A 210-dimensional flattened array consisting of a 30-timestep sliding historical window (`WARD_TEMPORAL_WINDOW`) across 7 localized features:
  1. `congestion`: Delayed traffic ratio within the ward.
  2. `queue`: Number of halted vehicles (speed $< 0.1\text{ m/s}$).
  3. `avg_speed`: Average speed of active vehicles.
  4. `inflow`: Ingress rate of boundary vehicles.
  5. `outflow`: Throughput of exiting vehicles.
  6. `incident_flag`: Number of active breakdowns or road blockages.
  7. `ambulance_flag`: Active emergency vehicles needing priority.
  - *Plus top-down directives*: Graph spatial pressure forecasts and city-level capacity caps.
- **Action Space**: 10 high-level semantic actions defined in [ward_actions.py](file:///d:/Bunker/BaseCamp/Hierarchical-multi-agent-RL-for-Urban-Traffic/src/rl/ward_actions.py):
  - `NO_OP` (0): Relinquishes control to default SUMO actuated timings.
  - `REROUTE_HOTSPOT_GROUP` (1): Reroutes all vehicles on the ward's most congested edge.
  - `DEPRIORITIZE_MOST_CONGESTED_EDGE` (2): Artificially penalizes travel time weights of a congested lane to deflect traffic.
  - `CLEAR_AMBULANCE_PATH` (4): Clear corridors by force-rerouting surrounding vehicles ahead of active ambulances.
  - `HOLD_COMMERCIAL_INFLOW` (6) & `RELEASE_HELD_FLOW` (7): Temporary inflow throttling.
  - `REROUTE_AGGRESSIVE_DRIVERS` (8) & `REROUTE_HEAVY_VEHICLES` (9): Segregates traffic to redistribute local loads.

### 2. Area Layer (L2 - Mesoscopic)
- **Role**: Coordinates neighboring wards in an area (e.g. `HSR_Layout` or `BTM_Layout`) to prevent downstream deadlocks and maximize green wave alignment.
- **Model Type**: Spatio-Temporal Graph Convolutional Network (STGCN) combining Graph Convolutions with Gated Recurrent Units (GRU) for sequence learning, implemented in [area_controller.py](file:///d:/Bunker/BaseCamp/Hierarchical-multi-agent-RL-for-Urban-Traffic/src/controllers/area_controller.py).
- **Control Interval**: Predicts near-future ward pressure every **60 simulation seconds** (Area Tick).
- **Function**: Uses the spatial topology ( adjacency matrix normalized via $\hat A = D^{-1/2}(A+I)D^{-1/2}$) to forecast congestion patterns 30 seconds ahead. It integrates L3 capacity caps, shaping the pressure targets sent to lower-level ward agents.

### 3. City Layer (L3 - Macroscopic)
- **Role**: Global capacity allocator and path-routing optimizer.
- **Model Type**: Pure mathematical optimization and graph routing solver, implemented in [city_controller.py](file:///d:/Bunker/BaseCamp/Hierarchical-multi-agent-RL-for-Urban-Traffic/src/controllers/city_controller.py).
- **Control Interval**: Executes every **120 simulation seconds** (City Tick), or triggers instantly upon detecting a critical incident/extreme congestion alert.
- **Function**: Constructs a macro directed graph of inter-area connectivity. It calculates pressure-proportional balancing capacity caps ($c \in [0.2, 1.0]$) for each area. If an area suffers severe congestion or breakdowns, the L3 controller throttles neighboring inbound corridors to prevent cascading gridlocks.

---

## Traffic Demand & Incidents Pipeline

The traffic simulation engine implemented in [traffic_generator.py](file:///d:/Bunker/BaseCamp/Hierarchical-multi-agent-RL-for-Urban-Traffic/src/traffic_generator.py) uses a realistic hybrid model:

1. **Deterministic Pre-compilation (Phase 1)**:
   - Generates vehicles using a **70/30 spatial split**: 70% spawn stochastically at boundary ingress edges (weighted by lane widths and historical priors) while 30% represent local trip starts.
   - Trips are compiled into SUMO-native route files (`.rou.xml`), bypassing TraCI IPC socket overhead to maximize CPU performance during RL step training.
2. **Stochastic Real-time Disturbances (Phase 2)**:
   - At every single tick, rolls a probability check against the scenario's `breakdown_prob` (up to 8% in `chaos_mode`).
   - If triggered, halts a random active vehicle (`speed = 0.0`), creating a physical blockage. This tests the agents' capacity to reroute traffic around hotspots.
3. **Vehicle Profiles**: Incorporates heterogeneous Indian traffic conditions defined in [vehicle_profiles.py](file:///d:/Bunker/BaseCamp/Hierarchical-multi-agent-RL-for-Urban-Traffic/configs/vehicle_profiles.py), including cars, bikes, slow drivers, aggressive/rash drivers, heavy trucks, BMTC buses, and priority ambulances.

---

## Project Directory Structure

```
.
├── configs/                     # System configurations
│   ├── hierarchy/               # Area & ward linkage registries
│   ├── scenarios.py             # Traffic scenarios (normal, peak_congestion, chaos_mode)
│   └── vehicle_profiles.py      # Vehicle physics profiles & XML generation
│
├── src/                         # Core Python modules
│   ├── controllers/             # Hierarchical controller logic
│   │   ├── area_controller.py   # L2 STGCN forecaster
│   │   ├── city_controller.py   # L3 Macro Graph Solver
│   │   └── ward_agent.py        # L1 PPO / DQN wrapper
│   │
│   ├── rl/                      # Reinforcement learning components
│   │   ├── sb3_ward_adapter.py  # Gym/Gymnasium wrapper for SUMO env
│   │   └── ward_actions.py      # L1 semantic discrete action adapter
│   │
│   ├── runtime.py               # Main simulation orchestration loop
│   ├── sumo_env.py              # SUMO TraCI abstraction layer
│   └── topology.py              # Network parser & ward stitching utilities
│
├── dashboard/                   # Real-time Web UI Dashboard
│   ├── server.py                # Flask CORS-REST server
│   └── index.html               # CSS/JS dashboard interface
│
├── models/                      # Saved PyTorch checkpoints for RL and GNN models
│
├── maps/                        # GIS boundaries and network assets
│
├── evaluate.py                  # Structured evaluation framework
├── hmrl.py                      # Master inference script & entrypoint
└── requirements.txt             # Python dependency manifest
```

---

##  Installation & Environment Setup

### 1. Install SUMO
Ensure that Eclipse SUMO is installed on your operating system.
- **Windows**: Download the installer from the [SUMO Download Page](https://sumo.dlr.de/docs/Downloads.shtml) and run it.
- **Linux**: Install via apt:
  ```bash
  sudo apt-get install sumo sumo-tools sumo-doc
  ```
- **Set Environment Variable**: Make sure `SUMO_HOME` is set to your SUMO installation folder (e.g. `C:\Program Files (x86)\Eclipse\Sumo`).

### 2. Install Python Dependencies
Set up your virtual environment and install the required libraries:
```bash
pip install -r requirements.txt
```

---

##  Usage Guide

### 📂 1. Preprocessing and Map Stitching
Before running a simulation, ensure your ward map assets are parsed and preprocessed:
- Map directory target is resolved via the `HMRL_MAP_DIR` environment variable (defaults to `"processed"`).
- Wards are automatically stitched together by the `Topology` module during city-level executions.

### 🌐 2. Running a Simulation with GUI and Dashboard
To run a city-level simulation of HSR Layout and BTM Layout with pre-trained models, opening the SUMO GUI and spinning up the real-time metrics dashboard:

```bash
python hmrl.py --scope city --areas HSR_Layout BTM_Layout --scenario chaos_mode --algorithm ppo --gui --max-ticks 900
```

- **SUMO GUI**: A window will open where you can watch individual vehicle movements and traffic lights.
- **Dashboard UI**: Once the log prints `[dashboard] Starting server`, open your browser to **`http://localhost:5050`** to view real-time metrics, queue trends, area forecasts, and active actions.

### 📊 3. Running the Ablation Evaluation Framework
To run systematic benchmarks comparing progressively intelligent layers against a baseline (No RL) across all scenarios:

```bash
# Run a quick check across representative wards
python evaluate.py --mode quick --max-ticks 300

# Run a full hierarchical (Ward + Area + City) simulation evaluation
python evaluate.py --mode city --scenario chaos_mode --max-ticks 900
```
Upon completion, the framework writes detailed comparative plots and text files to **`results/evaluation/`**:
- `evaluation_report.txt`: Percentage improvement summaries.
- `congestion_and_speed.png`: Comparative time-series trends.
- `ambulance_and_routing.png`: Delay reductions for priority vehicles.

---

## 🧪 Mathematical Formulations

### L1 Reward Function (Microscopic Optimization)
The Ward RL agent's step reward is shaped to minimize congestion metrics locally:
$$Reward_t = - \left( w_1 \cdot \text{QueueLength}_t + w_2 \cdot \text{WaitingTime}_t + w_3 \cdot \text{IncidentDelay}_t + w_4 \cdot \text{AmbulanceDelay}_t \right)$$
- If the agent successfully clears pathways for emergency vehicles or reroutes traffic around blockages, the delays drop, giving a higher reward.

### L2 Graph Spatial Convolution (Mesoscopic Pressure)
Wards are treated as nodes in a graph $G = (V, E)$, with edges representing adjacent road connectivity. The Area GNN predicts the future ward pressure $P \in [0, 1]^N$ using:
$$H^{(l+1)} = \sigma \left( \hat{A} H^{(l)} W^{(l)} \right)$$
Where $\hat{A}$ is the normalized adjacency matrix, $H^{(l)}$ represents node features at layer $l$, and $W^{(l)}$ is the layer weight matrix.

### L3 Capacity Solver (Macroscopic Balancing)
The city controller determines capacity caps $C_k \in [0.2, 1.0]$ for each Area $k$:
$$C_k = 1.0 - \left( 0.6 \cdot \frac{Congestion_k}{\max_j Congestion_j} \right)$$
If a neighboring area $j$ has severe congestion, outbound capacities of adjacent areas are throttled to prevent upstream gridlock propagation:
$$C_{\text{neighbor}} \leftarrow \max(0.2, C_{\text{neighbor}} - 0.15 \cdot Congestion_j)$$
