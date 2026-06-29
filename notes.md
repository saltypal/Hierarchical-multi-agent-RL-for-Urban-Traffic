Traffic generation
The traffic generation engine ([traffic_generator.py](file:///d:/Bunker/BaseCamp/Hierarchical-multi-agent-RL-for-Urban-Traffic/src/traffic_generator.py)) operates as a **two-phase system**: first, it compiles a static probabilistic route demand file (`.rou.xml`), and second, it dynamically injects real-time incidents (breakdowns) during the active simulation loop.

Here is exactly how the entire process happens step-by-step:

---

### Phase 1: Compiling the Traffic Demand (`.rou.xml`)

Before the simulation starts, the `TrafficGenerator` calculates the volume, vehicle types, spawn points, and exit points for the duration of the scenario.

#### 1. Calculating Total Vehicle Count
The base traffic density is determined by the active scenario's **`spawn_multiplier`** (defined in `configs/scenarios.py`), boosted by a **`1.15` (15%) over-sampling factor** to compensate for un-routable isolated roads that SUMO safely discards:

$$\text{Total Vehicles} = 100 \times \text{number of wards} \times \text{spawn\_multiplier} \times 1.15$$

* *Example:* For `normal` traffic (`spawn_multiplier = 1.0`) on a single ward, it generates **115 vehicle trips**.
* *Example:* For `chaos_mode` traffic (`spawn_multiplier = 3.0`) on a single ward, it generates **345 vehicle trips**.

#### 2. Selecting Ingress & Egress Edges (The 70/30 Rule)
Vehicles are assigned starting points and destinations using a **probabilistic lane-weighted model** based on [boundaries.json](file:///d:/Bunker/BaseCamp/Hierarchical-multi-agent-RL-for-Urban-Traffic/maps/processed/ward_070/boundaries.json):

* **70% Boundary Inflow:** 70% of vehicles are assigned to spawn at the ward's entryways (`valid_ingress_edges`). The selection probability is **weighted by the lane count** of each entry edge. Wider main roads spawn more vehicles than narrow residential entryways.
* **30% Internal Trips:** 30% of vehicles are assigned to spawn at random drivable edges inside the ward network, simulating local trip initiation.
* **Egress Destinations:** Destinations are randomly sampled from `valid_egress_edges`, also weighted by lane counts, directing traffic naturally toward major exits.
* **Congestion Prior Multiplier:** If the ward has a high historical density rating (`medium` or `high` congestion prior in the registry), a **congestion bonus** is added to the entry lane weights, further magnifying boundary pressure.

#### 3. Sampling Vehicle Types & Depart Times
For each trip, the generator:
* **Samples a Vehicle Profile:** Randomly picks a vehicle class based on the scenario's vehicle mix ratios (e.g. 70% normal cars, 10% slow drivers, 8% BMTC buses, 5% heavy trucks, 1% emergency ambulances).
* **Assigns Physical Physics Profiles:** The vehicle's parameters (length, max speed, acceleration, deceleration, and driver imperfection) are loaded from [vehicle_profiles.py](file:///d:/Bunker/BaseCamp/Hierarchical-multi-agent-RL-for-Urban-Traffic/configs/vehicle_profiles.py).
* **Distributes Departures:** Spreads out departures by incrementing departure times by a random interval (`depart += rng.uniform(0.5, 3.0)`), creating a natural arrival flow rather than spawning everyone at once.

#### 4. Writing the SUMO Route File
All trips are compiled into XML format:
```xml
<trip id="ward_070_normal_car_12" type="normal_car" depart="15.4" from="290643499#1" to="645970024#12"/>
```
This is saved to `<ward_id>.rou.xml` alongside `<ward_id>.sumocfg`.

---

Yes, every time you initiate a simulation run, the TrafficGenerator connects the scenario definition to the ward boundaries, compiles a fresh, scenario-specific .rou.xml file, packages it inside a .sumocfg configuration, and feeds it directly into SUMO.

---

### Phase 2: Dynamic Real-Time Incidents (During Simulation)

Once the simulation starts, the `TrafficGenerator.step(sumo_env, tick)` method is called at **every simulation tick (1 second)** to inject active disturbances:

```python
def step(self, sumo_env: Any, tick: int) -> None:
    self._inject_breakdown(sumo_env, tick)
```

#### 1. Stochastic Breakdown Checks
Every second, the engine rolls a random number against the scenario's **`breakdown_prob`** (e.g., `0.02` / 2% probability in the `breakdown` scenario):

```python
if self.rng.random() < self.scenario["breakdown_prob"]:
    # A breakdown is triggered!
```

#### 2. Vehicle Halting
If a breakdown triggers:
1. It queries `SumoEnv` to get a list of all active vehicles currently driving in the network.
2. It randomly selects one vehicle (e.g., `ward_070_normal_car_12`).
3. It calls `sumo_env.set_vehicle_speed(broken, 0.0)`.
4. The vehicle immediately stops dead in its lane, functioning as a physical blockage. This forces following cars to decelerate, change lanes, or accumulate into realistic spillback congestion queues, testing the RL agent's ability to reroute traffic.

This is a highly insightful system design question. The decision to pre-compile the traffic demand into a `.rou.xml` file rather than spawning every vehicle dynamically at runtime via Python is a **deliberate architectural best practice** that resolves three major bottlenecks in reinforcement learning and micro-simulation:

---

### 1. The TraCI IPC/Socket Bottleneck (Performance)
The communication channel between Python and the SUMO C++ simulator is **TraCI** (Traffic Control Interface), which operates over a **local TCP socket**. 
* **The Cost of TraCI Calls:** Every time Python calls a TraCI command (like `traci.vehicle.add()`), it has to serialize the data, send it over the TCP loopback interface, wait for the SUMO C++ engine to process it, and receive a response.
* **Overhead at Scale:** In an urban network with hundreds of active vehicles spawning constantly, performing these socket roundtrips for every single vehicle insertion slows the simulation speed to a crawl.
* **The Solution:** By pre-compiling the trips into a `.rou.xml` file, **SUMO’s native C++ engine manages vehicle loading and spawning internally**. Python never has to execute socket commands for spawns, preserving 100% of the CPU cycles for high-frequency RL policy execution and GNN prediction rollouts.

---

### 2. Scientific Reproducibility & Fair Benchmarking
In Reinforcement Learning, to scientifically prove that one model (e.g., PPO) is superior to another (e.g., DQN or a legacy static controller), **every agent must be evaluated under the exact same traffic conditions**.
* If traffic were generated dynamically on the fly with fresh random seeds every second, the reward signals would be highly noisy and inconsistent (e.g., PPO might get evaluated on a lighter random traffic wave, whereas DQN gets evaluated on an extremely unlucky congestion surge).
* By saving the stochastically sampled trips into a `.rou.xml` file, **we create a deterministic demand benchmark**. Every RL algorithm is tested against the exact same vehicle arrival sequences, depart times, and routing demands, ensuring that performance differences are 100% due to the agents' control capabilities.

---

### 3. Exploiting SUMO's Native C++ Router
When a vehicle trip is defined in a `.rou.xml` file:
* SUMO's highly-optimized internal C++ routing engine (using Dijkstra or A* algorithms) computes the shortest path between the origin and destination instantly at spawn.
* If we spawned vehicles completely dynamically at runtime in Python, we would have to manually compute valid, contiguous paths in Python (using NetworkX) and explicitly feed the entire list of edges for every single vehicle trip via TraCI, which introduces major CPU computational overhead.

---

### Summary: A Hybrid Dynamic Architecture
The framework is actually a **hybrid dynamic system**:
1. **Deterministic Spawning (`.rou.xml`):** Spawning schedules, vehicle types, and initial shortest paths are pre-compiled to bypass socket overhead and ensure a fair benchmark.
2. **Dynamic Control (TraCI/RL):** Once spawned, the traffic flow is **highly dynamic**. The RL ward agents actively override the default simulator flow at every single second by altering traffic signal phases, and the GNN dynamically penalizes congestion, causing vehicles to reroute in real time based on active queue pile-ups.








-----


Stitches multiple arbitrary areas together, deploying a multi-agent ward network under the coordination of the global City Coordinator which executes a Linear Programming (LP) optimization solver to restrict boundary inflows and block gridlock propagating between constituencies.



Searched for "WARD_TEMPORAL_WINDOW"
Searched for "WARD_"
Viewed runtime.py:1-50
Viewed temporal_features.py:1-91
Searched for "_build_ward_observation"
Searched for "_build_ward_observation"
Searched for "class SumoEnv"
Searched for "gym"
Viewed sb3_ward_adapter.py:1-444

The RL Ward Agent observes a **flattened temporal stack (sliding historical window)** of localized traffic metrics. 

Instead of a single "static" snapshot, the agent sees the trend over time, which helps it learn how queues form, clear, or propagate.

Here are the exact specifications of the RL observation space:

### 1. Shape and Structure
* **Total Dimension:** **210 dimensions** (`WARD_TEMPORAL_WINDOW` * `WARD_FEATURE_DIM`).
* **Temporal Window:** **30 timesteps** (the sliding history queue).
* **Feature Dimensions:** **7 features** collected at each timestep frame.
* **Format:** The 30 historical frames are stacked chronologically, padded (if the simulation just started), and flattened into a single 1D tensor of shape `(210,)` before being passed to the RL model (e.g. SB3 PPO).

---

### 2. The 7 Features Observed at Each Timestep
Every second, the environment captures a 7-dimensional frame ([`src/controllers/temporal_features.py`](file:///d:/Bunker/BaseCamp/Hierarchical-multi-agent-RL-for-Urban-Traffic/src/controllers/temporal_features.py#L18-L28)):

| Feature | Feature Name | Description |
| :--- | :--- | :--- |
| **1** | `congestion` | **Congestion score:** A normalized ratio representing the intensity of congestion/delays across all edges within the ward. |
| **2** | `queue` | **Queue length:** The total count of halting vehicles (vehicles moving slower than $0.1\text{ m/s}$) inside the ward. |
| **3** | `avg_speed` | **Average speed:** The average current speed of all vehicles active in the ward network. |
| **4** | `inflow` | **Traffic inflow:** The rate/count of vehicles entering the ward's edges. |
| **5** | `outflow` | **Traffic outflow:** The throughput rate (number of vehicles successfully exiting the ward). |
| **6** | `incident_flag` | **Incident presence:** A scalar indicating active breakdowns or blocked roads in the ward (Phase 2 incidents). |
| **7** | `ambulance_flag` | **Emergency presence:** A scalar indicating if an emergency vehicle (ambulance) is active in the ward and needs prioritization. |

---

### 3. Temporal Graph & Hierarchy (Observations from upper layers)
When running within the full hierarchical system ([`src/runtime.py`](file:///d:/Bunker/BaseCamp/Hierarchical-multi-agent-RL-for-Urban-Traffic/src/runtime.py)), the GNN area forecaster and city controller predictions are dynamically integrated to influence the RL environment:
* **GNN Pressure:** The GNN's predicted ward-level pressure forecast acts as an upper-layer directive (injected into training synthetically or calculated in real-time during inference by the `AreaForecaster`).
* **City Capacity Caps:** Throttling boundaries computed by the global city controller limit entryways and propagate down to influence observations and shape reward calculation.


Searched for "class WardAction"
Viewed ward_actions.py:1-29

In this hierarchical RL framework, the reinforcement learning (RL) agents operate at the **Ward level**. Rather than manipulating individual, low-level signal timers directly, the agents use a discrete, **semantic action space** consisting of **10 high-level traffic control policies**. 

These actions are defined in [`src/rl/ward_actions.py`](file:///d:/Bunker/BaseCamp/Hierarchical-multi-agent-RL-for-Urban-Traffic/src/rl/ward_actions.py) and applied inside the simulation runtime ([`src/runtime.py`](file:///d:/Bunker/BaseCamp/Hierarchical-multi-agent-RL-for-Urban-Traffic/src/runtime.py#L153-L233)).

Here is the exact action catalog and how they affect the traffic simulation:

| Value | Action Name | How it Works under the Hood |
| :--- | :--- | :--- |
| **0** | `NO_OP` | **No operation:** Does nothing. Hand control back to default SUMO physics/timings. |
| **1** | `REROUTE_HOTSPOT_GROUP` | **Local hotspot bypass:** Finds the single edge in the ward with the highest number of halting vehicles and calls `sumo_env.reroute_vehicle()` on all vehicles on that edge, forcing them to find alternate routes immediately. |
| **2** | `DEPRIORITIZE_MOST_CONGESTED_EDGE` | **Congestion avoidance routing:** Finds the most congested edge, artificially sets its travel time weight to `9999.0` (making it extremely undesirable in SUMO's internal router), and forces all vehicles on it to reroute. |
| **3** | `PRIORITIZE_ALTERNATE_EDGE` | Functions similarly to `REROUTE_HOTSPOT_GROUP` to trigger alternative path selection. |
| **4** | `CLEAR_AMBULANCE_PATH` | **Emergency vehicle prioritization:** Identifies any active ambulances in the ward, checks their next 3 upcoming edges, and forces all *non-emergency* vehicles on those edges to reroute, clearing a corridor. |
| **5** | `INCIDENT_REROUTE` | **Incident response:** Selects the ward's key inflow edges and clears them of traffic by rerouting all vehicles currently on them. |
| **6** | `HOLD_COMMERCIAL_INFLOW` | **Inflow throttling:** Artificially holds traffic by reducing the speed of all active vehicles on the ward's edges to `0.0` (holding them in place). |
| **7** | `RELEASE_HELD_FLOW` | **Throttling release:** Restores the speed control of all stochastically held vehicles back to SUMO (`speed = -1.0`), releasing them back into normal flow. |
| **8** | `REROUTE_AGGRESSIVE_DRIVERS` | **Disorder management:** Filters all active vehicles in the ward for aggressive/rash drivers and triggers a rerouting event on them to distribute their impact. |
| **9** | `REROUTE_HEAVY_VEHICLES` | **Heavy vehicle routing:** Filters all active vehicles for trucks and BMTC buses, and forces them to reroute to prevent heavy vehicles from blockading narrow lanes. |


**Slide: Title**  
- **Title:** Spatio‑Temporal GCN (STGCN) — Area Pressure Forecaster  
- **One‑line:** Predicts per‑ward near‑future pressure by combining spatial graph convolution with temporal recurrence.

**Slide: Why STGCN?**  
- **Spatial Coupling:** traffic effects propagate along road/ward links (use maps → adjacency).  
- **Temporal Dynamics:** congestion evolves over minutes — need memory across timesteps.  
- **Parameter Efficiency:** shared weights generalize across wards and topologies.

**Slide: High‑level Architecture**  
- **Input:** `X_seq ∈ ℝ^{T×N×F}` (T=sequence length, N=wards, F=8 features).  
- **Spatial (per‑timestep):** GCN projection: $H_t = \mathrm{ReLU}(\hat A\,X_t\,W_g)$.  
- **Temporal:** GRU over each node’s spatial sequence → final state $h_T^i$.  
- **Head:** $P_i = \sigma(W_o h_T^i)$ → pressure ∈ [0,1] per ward.

**Slide: Core Math**  
- **Adjacency normalization:** $ \hat A = D^{-1/2}(A+I)D^{-1/2} $  
- **Spatial conv (per t):** $ H_t = \mathrm{ReLU}(\hat A\,X_t\,W_g) $  
- **Output:** $ P_i = \sigma\big(W_o\,\mathrm{GRU}([H_{1,i},\dots,H_{T,i}])\big) $

**Slide: Dataflow (mermaid)**

```mermaid
flowchart LR
  SUMO[SUMO simulator\ndata each tick]
  FB[Feature Builder\nX_t ∈ R^{N×F}]
  A[Topology → A_hat\n(normalized adjacency)]
  GCN[Spatial GCN per t\nH_t = ReLU(A_hat · X_t · W_g)]
  NodeSeq[Node-wise sequences\n(H_1..H_T)_i]
  GRU[GRU per node\nprocess sequence → h_T^i]
  Head[Linear + Sigmoid\n→ P ∈ [0,1]^N]
  Ward[Ward Agents\nreward shaping & obs]
  SUMO --> FB
  FB --> GCN
  A --> GCN
  GCN --> NodeSeq
  NodeSeq --> GRU
  GRU --> Head
  Head --> Ward
  Ward --> SUMO
```

**Slide: Training**  
- **Labels:** supervised targets = observed congestion at +Δ seconds (collected during RL runs).  
- **Windowing:** sliding windows of length `T` → sequences for STGCN.  
- **Loss / Opt:** MSE loss, Adam optimizer.  
- **Save:** `area_model_stgcn.pt` (weights only).

**Slide: Runtime & Integration**  
- **Ingest cadence:** append frame each tick; pad history until T frames.  
- **Predict cadence:** predictions used at `AREA_INTERVAL` ticks (default 60).  
- **Use of P:** shapes WardAgent reward/observation to discourage inflow into high‑pressure wards.

**Slide: Practical Tips / Pitfalls**  
- **Normalize features:** scale speeds, counts, congestion to stable ranges.  
- **Adjacency:** compute from maps; fallback to identity if node counts mismatch.  
- **Padding:** pad history with earliest frame (code does this).  
- **Debugging:** log `X_seq`, `H_t`, and `P` for a small area to validate behavior.  
- **Batching & device:** check tensor shapes `[B,T,N,F]` and run on GPU when available.

**Slide: Quick demo snippet**  
```python
# minimal forward (pseudo)
T, N, F, H = 10, 9, 8, 32
x_seq = torch.rand(T, N, F)              # [T,N,F]
a_hat = torch.tensor(adj, dtype=torch.float32)  # [N,N]
model = SpatioTemporalGCN(in_features=F, hidden_dim=H, seq_len=T)
pred = model(x_seq, a_hat)               # [N]
print(pred.shape)  # -> torch.Size([N])
```

**Slide: Key takeaways**  
- **STGCN = spatial + temporal:** GCN captures neighborhood effects; GRU captures evolution.  
- **Maps matter:** adjacency encodes real connectivity so pressure predictions respect topology.  
- **Use cases:** reward shaping, early congestion warning, guiding ward‑level RL decisions.