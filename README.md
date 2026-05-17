# Hierarchical-multi-agent-RL-for-Urban-Traffic
A hierarchical multi-agent reinforcement learning (HMARL) framework for large-scale urban traffic orchestration using SUMO + TraCI over real Bangalore road networks.

## Current Runnable Entrypoints

- `python hmrl.py --mode preprocess --scenario-id wednesday_peak_techpark_maps_v1`
  - scaffolds `maps/` area folders
  - writes per-area preprocessing reports and SUMO command plans
  - generates configurable OD matrix artifacts
- `python hmrl.py --mode train_ppo --scope ward --sumo-config Traffic/sumo/config.sumocfg`
  - runs SB3 PPO with the SUMO-backed ward adapter (requires `stable-baselines3` + `gymnasium`)
- `python hmrl.py --mode train_dqn --scope ward --sumo-config Traffic/sumo/config.sumocfg --gui`
  - runs the legacy hierarchical DQN loop


# Hierarchical Multi-Agent Reinforcement Learning for Urban Traffic Optimization Using SUMO and TraCI

## Project Overview

This project proposes a scalable and adaptive urban traffic management framework using Reinforcement Learning (RL), SUMO (Simulation of Urban Mobility), and the TraCI API over realistic Bangalore road networks.

The core idea is to model the city traffic system as a hierarchical network of intelligent Roadside Units (RSUs), where each RSU is responsible for managing traffic flow at a specific spatial level. Instead of using static traffic signal systems or predefined routing strategies, the system dynamically learns optimal traffic orchestration policies using RL algorithms such as DQN, PPO, and Actor-Critic methods.

The framework is designed to operate on real Bangalore map segments extracted from OpenStreetMap (OSM), enabling experimentation over heterogeneous urban structures such as structured residential layouts, dense commercial regions, highways, roundabouts, and bottleneck-heavy junctions.

The project aims to study how different RL algorithms perform under varying traffic topologies, road morphologies, traffic densities, and demographic driving behaviors, while also exploring hierarchical coordination between multiple traffic control agents.

---

# Core Motivation

Traditional traffic management systems suffer from several limitations:

* Static signal timings
* Lack of adaptive congestion handling
* No coordination between neighboring intersections
* Inability to scale dynamically to real-time urban traffic patterns
* Poor handling of heterogeneous Indian traffic conditions

Most existing RL-based traffic optimization systems are also limited because they:

* focus only on a single junction,
* use simplified road networks,
* ignore large-scale coordination,
* and do not evaluate algorithm suitability across different urban environments.

This project addresses these limitations by introducing:

* hierarchical traffic intelligence,
* distributed multi-agent coordination,
* and adaptive RL policy selection based on urban topology.

---

# System Architecture

The proposed architecture follows a hierarchical multi-agent structure.

```text id="40l5aj"
Vehicle Layer
↓
Local RSU Layer
↓
Area RSU Layer
↓
Regional RSU Layer
↓
City-Level Super RSU
```

Each layer operates at a different spatial abstraction level and has different responsibilities.

---

# 1. Vehicle Layer

Each vehicle in the SUMO simulation acts as a dynamic traffic entity.

Using the TraCI API, every vehicle continuously communicates information such as:

* position,
* speed,
* acceleration,
* waiting time,
* lane occupancy,
* route information,
* destination,
* congestion state.

Vehicles themselves are not initially treated as independent RL agents. Instead, they function as observable entities controlled indirectly by higher-level RSUs.

---

# 2. Local RSU Layer

A Local RSU manages a specific traffic region such as:

* a single intersection,
* a roundabout,
* a corridor,
* or a small road segment.

This layer performs fine-grained traffic control operations.

## Responsibilities

* Traffic signal phase control
* Adaptive green-time allocation
* Queue balancing
* Local congestion reduction
* Lane prioritization
* Emergency vehicle handling

## State Space

The RL agent observes:

* queue length,
* waiting time,
* average vehicle speed,
* vehicle density,
* lane occupancy,
* incoming and outgoing traffic rates.

## Action Space

The agent performs actions such as:

* switching signal phases,
* extending or reducing green duration,
* enabling lane priorities,
* applying local rerouting policies.

---

# 3. Area RSU Layer

Multiple Local RSUs are coordinated by an Area RSU.

An Area RSU represents a larger urban locality such as:

* HSR Layout,
* Koramangala,
* Indiranagar,
* Bellandur.

The Area RSU prevents neighboring intersections from making conflicting local decisions.

For example:

* one junction should not aggressively clear traffic if it causes downstream deadlock in adjacent intersections.

## Responsibilities

* Inter-junction coordination
* Area-wide traffic balancing
* Congestion propagation control
* Coordinated signal timing
* Corridor optimization

---

# 4. Regional RSU Layer

Regional RSUs coordinate multiple Area RSUs.

Examples:

* Bangalore South,
* East Bangalore,
* Central Bangalore.

This layer performs macro-scale traffic orchestration.

## Responsibilities

* Traffic distribution between areas
* Peak-hour balancing
* Route pressure redistribution
* Regional congestion minimization

The Regional RSU operates using aggregated traffic states rather than low-level junction states.

---

# 5. Super Master RSU

The highest level in the hierarchy is the city-level Super Master RSU.

This layer does not directly control traffic signals. Instead, it acts as a strategic meta-controller.

## Responsibilities

* Global traffic optimization
* City-wide congestion monitoring
* Emergency response coordination
* Reward shaping for lower-level agents
* Large-scale rerouting policies
* Dynamic policy adaptation

The Super Master RSU enables scalable intelligent traffic management for the entire simulated Bangalore city.


City RSU
│
├── Regional Area RSUs
│      ├── Area RSUs
│      │      ├── Local Area RSUs
│      │      │      ├── Junctions
│      │      │      └── Vehicles

---

# Realistic Bangalore Traffic Modeling

Unlike synthetic road networks, this project uses realistic Bangalore maps extracted from OpenStreetMap.

Different Bangalore areas are selected to represent different urban morphologies and traffic behaviors.

| Area                 | Characteristics           |
| -------------------- | ------------------------- |
| HSR Layout           | Structured grid roads     |
| Koramangala          | Mixed urban traffic       |
| Bellandur            | IT corridor congestion    |
| Silk Board           | Chaotic merging traffic   |
| Electronic City      | Highway + urban hybrid    |
| Indiranagar          | Dense commercial traffic  |
| Tin Factory Junction | Bottleneck-heavy topology |

Each area is converted into a normalized SUMO simulation environment using Python preprocessing scripts.

The preprocessing pipeline includes:

* map extraction,
* junction normalization,
* lane standardization,
* route generation,
* ID normalization,
* traffic flow generation.

---

# Reinforcement Learning Framework

The project evaluates multiple RL algorithms across different traffic environments.

The main objective is not only traffic optimization, but also determining which RL algorithm performs best under specific urban conditions.

---

# RL Algorithms Used

## 1. DQN (Deep Q-Network)

Best suited for:

* discrete traffic signal control,
* small intersections,
* simple junctions.

### Suitable Environments

* 3-way intersections
* 4-way intersections
* structured grids

---

## 2. PPO (Proximal Policy Optimization)

Best suited for:

* large-scale dynamic environments,
* continuous adaptation,
* heterogeneous traffic conditions,
* scalable coordination.

### Suitable Environments

* dense urban areas,
* variable traffic patterns,
* large road networks,
* multi-agent systems.

---

## 3. Actor-Critic Methods (A2C/A3C)

Best suited for:

* hierarchical coordination,
* multi-agent collaboration,
* real-time adaptive decision making.

### Suitable Environments

* regional coordination,
* RSU communication,
* large-scale orchestration.

---

# Hierarchical RL Strategy

Different RL algorithms are assigned to different hierarchy levels.

| Layer            | Recommended RL Model      |
| ---------------- | ------------------------- |
| Local RSU        | DQN / Double DQN          |
| Area RSU         | PPO                       |
| Regional RSU     | Actor-Critic              |
| Super Master RSU | PPO + Actor-Critic Hybrid |

This enables:

* efficient local control,
* stable regional coordination,
* scalable city-level optimization.

---

# Experimental Study

The project performs comparative analysis across:

* road topologies,
* traffic densities,
* demographic patterns,
* RL algorithms,
* hierarchical coordination strategies.

---

# Topology-Based Evaluation

The algorithms are tested on:

* 3-way junctions,
* 4-way intersections,
* roundabouts,
* arterial corridors,
* grid-based layouts,
* mixed urban structures.

---

# Traffic Conditions

Experiments include:

* low-density traffic,
* medium-density traffic,
* high-density traffic,
* peak-hour bursts,
* stochastic congestion conditions.

---

# Indian Traffic Demography Modeling

The simulation incorporates heterogeneous Indian traffic conditions:

* cars,
* bikes,
* buses,
* trucks,
* autos.

Behavioral parameters include:

* lane discipline,
* overtaking aggressiveness,
* acceleration variability,
* congestion response patterns.

This significantly improves realism compared to traditional traffic RL studies.

---

# Evaluation Metrics

The framework is evaluated using both traffic metrics and RL performance metrics.

## Traffic Metrics

* Average waiting time
* Queue length
* Throughput
* Average travel time
* Average speed
* Fuel consumption
* CO₂ emissions
* Stop frequency

## RL Metrics

* Cumulative reward
* Convergence stability
* Training efficiency
* Scalability
* Reward variance
* Policy robustness

---

# Expected Outcomes

The project aims to:

* reduce congestion,
* improve traffic flow,
* minimize waiting time,
* reduce fuel consumption,
* improve regional coordination,
* identify optimal RL strategies for different urban structures,
* and demonstrate scalable hierarchical traffic intelligence.

---

# Key Research Contribution

The primary contribution of this work is:

> Adaptive RL policy selection and hierarchical multi-agent coordination for heterogeneous urban traffic environments.

Instead of assuming a single RL model works universally, the project investigates:

* which RL models work best,
* under which traffic conditions,
* for which road structures,
* and at what hierarchy level.

This makes the work significantly more research-oriented and scalable than conventional traffic signal optimization systems.

---

# Final Refined One-Paragraph Version

> This project proposes a hierarchical multi-agent reinforcement learning framework for large-scale urban traffic optimization using SUMO and the TraCI API over realistic Bangalore road networks. The system models traffic control as a distributed RSU-based architecture, where local RSUs manage individual intersections and road segments, while higher-level RSUs coordinate area-level, regional, and city-wide traffic flow. Real Bangalore regions such as HSR Layout, Bellandur, and Koramangala are extracted from OpenStreetMap and converted into standardized SUMO simulation environments. Reinforcement learning algorithms including DQN, PPO, and Actor-Critic methods are trained and benchmarked across heterogeneous traffic topologies, varying congestion conditions, and Indian demographic traffic behaviors. The project aims to determine the optimal RL strategy for different urban morphologies while minimizing congestion, queue length, waiting time, fuel consumption, and emissions through scalable hierarchical coordination and adaptive traffic intelligence.
======================================
also, the source and destinations, or routes, are they defined already in osm maps when converted? can i automate the defining of routes and source destiantions as houses/buildings/offices and can i define the frequency or how the traffic is from google maps for every area on a specific day? so i have decided that i will consider only one day (wednesday because its the peak), set the routes majorly towards techparks now tell me a way we can accomplish this.
So here is my proposed system design:

I want you to take the current code as base inspiration and modularize it completely and make use of it.

we will have different folders and python filesfor different things, including notebooks

So our system includes multiple agents at different levels
--------------------------
Build the preprocessing and map-generation pipeline for a hierarchical reinforcement learning based Bengaluru traffic optimization project using SUMO and TraCI.

The project structure should contain a root folder called:

maps/

   Inside the maps folder, create subfolders for the top 10 selected Bengaluru areas from different traffic and demographic regions. Example areas may include:

* HSR Layout
* Koramangala
* Bellandur
* Electronic City
* Indiranagar
* Whitefield
* Silk Board
* Tin Factory
* JP Nagar
* Hebbal

Each area folder should contain:

* Raw OpenStreetMap (.osm) files (keep placeholder, I will add it soon)
* SUMO network files (.net.xml) (Shall be added later)
* Route files (.rou.xml) (Shall be added later)
* Trip files (.trips.xml) (Shall be added later)
* Additional grid/intersection metadata XML files if needed (Shall be added later)
* Preprocessed and normalized network outputs (Shall be added later)

The OSM maps extracted initially will not be perfectly simulation-ready. The goal is to automatically preprocess and standardize them using scripts instead of manually modifying maps using SUMO NetEdit as much as possible.

Inside the root maps folder, create a Jupyter Notebook named:

map_preprocessing_pipeline.ipynb

This notebook should automatically iterate through all area folders and perform the following pipeline for each Bengaluru region:

1. Detect and load the raw OSM map file.
2. Convert the OSM map into SUMO-compatible network files using SUMO tools such as:
   * netconvert
   * polyconvert
   * randomTrips.py
   * duarouter
3. Automatically normalize and standardize:
   * edge IDs
   * node IDs
   * lane naming
   * traffic light naming
   * junction labeling
   * route identifiers
4. Ensure that edge naming follows a uniform convention across all Bengaluru regions so RL state extraction becomes consistent.
5. Automatically clean and preprocess problematic map structures such as:
   * disconnected roads
   * duplicate edges
   * malformed junctions
   * invalid lane connections
   * overly complex intersections
   * missing traffic signals
6. Use only scripts and automated preprocessing as much as possible instead of manual NetEdit corrections.
7. Generate realistic SUMO network outputs:
   * .net.xml
   * .edg.xml
   * .nod.xml
   * .con.xml
   * .poly.xml
8. Automatically classify junctions into categories such as:
   * 3-way intersections
   * 4-way intersections
   * roundabouts
   * arterial merges
   * corridor junctions
9. Create metadata JSON or CSV files for every area containing:
   * number of junctions
   * number of edges
   * average lane count
   * traffic signal count
   * road hierarchy statistics
   * junction classifications
10. Generate visualization outputs for each area:
* rendered road graph
* junction overlays
* cluster overlays
* traffic corridor highlighting
11. Automatically divide each area into RL clusters.
A cluster represents:
* 2–5 nearby traffic-dependent junctions controlled by one RL agent.
Cluster generation should consider:
* spatial proximity,
* shared arterial roads,
* congestion dependency,
* corridor continuity.
12. Export cluster metadata for later RL training.
13. Prepare route-generation placeholders for future traffic demand simulation.
Future route generation should support:
* residential-to-tech-park traffic,
* office rush-hour flows,
* Wednesday peak-hour traffic modeling,
* probabilistic origin-destination generation.
14. The notebook should be modular and reusable for adding future Bengaluru regions.
15. The entire preprocessing pipeline should be designed for scalable hierarchical reinforcement learning experimentation over realistic Bengaluru traffic environments using SUMO and TraCI.

===============================================

Now let's discuss about Src folder
inside SRC folder, we should have code that manages all 
Your src/ folder is:

simulation orchestration,
RL environments,
RSU coordination,
training,
evaluation,
Traffic intelligence.

