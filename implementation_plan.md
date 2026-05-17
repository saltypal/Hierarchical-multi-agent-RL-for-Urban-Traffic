# Codebase Cleanup and Secondary Map Architecture Plan

This plan outlines the steps to clean up obsolete scripts, confirm routing strategies, create the optimized `Processed_Map_2` (primary/secondary roads only), and execute a full retraining cycle for all RL models on the new map.

## 1. Open Questions / Confirmations
* **OD Matrix Confirmation:** I checked the codebase, and we are **NOT** using the legacy OD Matrix generation anymore! The RL simulation purely uses the dynamic boundary-pressure generators defined in `TrafficGenerator` (spawn bounds, ingress/egress probabilities). The only matrices being used are the Adjacency Matrices for the GNN.

## 2. Codebase Cleanup
I will create a new `redundant/` folder at the root of the project and move all outdated bootstrapping and testing scripts there to keep your root directory pristine.
* **Files to move:** `pick_20_wards.py`, `preprocess_all.py`, `resolve_relations.py`, `resolved_wards.json`, `test_env.py`, `test_overpass.py`, `test_traci.py`, `update_final_4_areas.py`, `update_registry.py`.
* **Directories to move:** `osm_download_fixes/`

## 3. Processed_Map_2 Architecture Pipeline
Instead of hardcoding `"processed"` everywhere, I will introduce an environment variable `HMRL_MAP_DIR` (defaulting to `"processed"`) across the stack so we can hot-swap map versions seamlessly without breaking the codebase.

#### [MODIFY] `src/preprocessing/ward_processor.py`
* Modify `process_ward` to accept `output_dir_name` and an optional list of `extra_netconvert_args` (so we can pass the edge filters for Map 2).

#### [MODIFY] `src/traffic_generator.py`, `src/topology.py`, `src/rl/sb3_ward_adapter.py`, `src/runtime.py`
* Read `os.getenv("HMRL_MAP_DIR", "processed")` instead of hardcoding the `"processed"` subfolder for network files.

#### [NEW] `build_map2.py`
* A script that loops through the 16 wards and runs `ward_processor.py` but saves to `Processed_Map_2`.
* It will pass the strict filter: `--keep-edges.by-type highway.primary,highway.primary_link,highway.secondary,highway.secondary_link,highway.trunk,highway.trunk_link,highway.motorway,highway.motorway_link`.
* This eliminates tertiary, residential, and footways, creating a highly optimized macroscopic map.

## 4. Retraining All Models on Processed_Map_2
#### [NEW] `retrain_map2.py`
* A master training script that automatically sets `HMRL_MAP_DIR="Processed_Map_2"`.
* Iteratively triggers `train_global_agent` for `PPO`, `A2C`, and `DQN` sequentially on the new map.
* Generates a new GNN dataset from the Map 2 topology.

## User Review Required
Please review the list of redundant files I'm moving. If you want to keep any of them in the root, let me know. If the `Processed_Map_2` architecture looks good, I will begin execution immediately!
