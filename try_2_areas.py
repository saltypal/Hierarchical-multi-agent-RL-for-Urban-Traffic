import sys
import os
import argparse
import logging
from pathlib import Path
import json
import numpy as np
import time
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Project Root Setup
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sumo_env import SumoEnv
from src.topology import Topology
from src.traffic_generator import TrafficGenerator, _weighted_choice, _normalize_edge_list
from src.controllers.city_controller import CityController
from src.controllers.area_controller import AreaForecaster
from src.controllers.ward_agent import WardAgent
from configs.vehicle_profiles import build_all_vtypes_xml

# Control timescales
WARD_INTERVAL = 1
AREA_INTERVAL = 30
CITY_INTERVAL = 120

def parse_args():
    parser = argparse.ArgumentParser(description="Coordinating 2 stitched areas in real-time.")
    parser.add_argument(
        "--ticks",
        type=int,
        default=200,
        help="Maximum simulation duration in ticks/seconds."
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Run simulation with SUMO GUI."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Initialize Topology
    topology = Topology(PROJECT_ROOT)
    
    # Define the 2 target areas to coordinate
    target_areas = ["HSR_Layout", "BTM_Layout"]
    logger.info(f"Targeting areas: {target_areas}")
    
    # Extract all constituent wards
    ward_ids = []
    for aid in target_areas:
        ward_ids.extend(topology.get_area_wards(aid))
    
    logger.info(f"Stitching wards: {ward_ids}")
    
    # Build ward -> area lookup
    ward_to_area = {}
    for aid in target_areas:
        for wid in topology.get_area_wards(aid):
            ward_to_area[wid] = aid
            
    # 2. Stitch maps using SUMO netconvert
    stitched_net = topology.stitch_ward_maps(ward_ids)
    output_dir = stitched_net.parent
    
    # 3. Generate Stitched Traffic routes across both areas
    # We combine boundary ingress/egress edges for both areas
    ward_set = set(ward_ids)
    area_ingress = []
    area_egress = []
    
    for wid in ward_ids:
        neighbors = topology.get_ward_neighbors(wid)
        has_external_face = any(n not in ward_set for n in neighbors) or len(neighbors) == 0
        if has_external_face:
            bounds = topology.get_ward_boundaries(wid)
            area_ingress.extend(_normalize_edge_list(bounds.get("valid_ingress_edges", [])))
            area_egress.extend(_normalize_edge_list(bounds.get("valid_egress_edges", [])))
            
    if not area_ingress or not area_egress:
        for wid in ward_ids:
            bounds = topology.get_ward_boundaries(wid)
            area_ingress.extend(_normalize_edge_list(bounds.get("valid_ingress_edges", [])))
            area_egress.extend(_normalize_edge_list(bounds.get("valid_egress_edges", [])))
            
    # Generate route file
    total_count = int(100 * len(ward_ids) * 1.15)  # over-sampling factor
    route_path = output_dir / "stitched_two_areas.rou.xml"
    
    # Generate vehicle mix generator
    temp_gen = TrafficGenerator(PROJECT_ROOT, ward_ids[0])
    
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">',
        "",
        build_all_vtypes_xml(),
        "",
    ]
    
    depart = 0.0
    for i in range(total_count):
        vtype = temp_gen._sample_vehicle_type()
        origin = _weighted_choice(area_ingress, temp_gen.rng)
        destination = _weighted_choice(area_egress, temp_gen.rng)
        veh_id = f"two_areas_{vtype}_{i}"
        depart += temp_gen.rng.uniform(0.5, 3.0)
        
        orig_id = origin["edge_id"] if isinstance(origin, dict) else origin
        dest_id = destination["edge_id"] if isinstance(destination, dict) else destination
        lines.append(f'    <trip id="{veh_id}" type="{vtype}" depart="{depart:.1f}" from="{orig_id}" to="{dest_id}"/>')
        
    lines.append("</routes>")
    route_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Generated route file: {route_path}")
    
    # 4. Generate sumocfg
    cfg_path = output_dir / "stitched_two_areas.sumocfg"
    cfg_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="{stitched_net}"/>
        <route-files value="{route_path}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="{args.ticks}"/>
    </time>
    <processing>
        <ignore-route-errors value="true"/>
    </processing>
</configuration>
"""
    cfg_path.write_text(cfg_content, encoding="utf-8")
    logger.info(f"Generated sumocfg file: {cfg_path}")
    
    # 5. Initialize Controllers and Agents
    sumo_env = SumoEnv()
    
    # Load global model if available
    global_model = PROJECT_ROOT / "models" / "ppo" / "global_agent" / "model.pt"
    ward_agents = {}
    for wid in ward_ids:
        # Fallback to ward local if global agent is missing
        model_path = global_model if global_model.exists() else PROJECT_ROOT / "models" / "ppo" / wid / "model.pt"
        ward_agents[wid] = WardAgent(wid, model_path)
        
    area_forecasters = {}
    for aid in target_areas:
        model_dir = PROJECT_ROOT / "models" / "gnn"
        area_forecasters[aid] = AreaForecaster(aid, topology, model_dir)
        
    city_controller = CityController(topology)
    
    # State buffers
    city_caps = {aid: 1.0 for aid in target_areas}
    gnn_predictions = {wid: 0.0 for wid in ward_ids}
    
    # Metrics
    total_arrived = 0
    speed_samples = []
    
    # 6. Start Simulation
    logger.info("Initializing multi-area simulation...")
    sumo_env.start(str(cfg_path), gui=args.gui)
    
    start_time = time.time()
    
    try:
        for tick in range(args.ticks):
            sumo_env.step()
            
            # Check completed
            if sumo_env.get_min_expected_number() <= 0 and tick > 10:
                logger.info("Simulation empty, terminating.")
                break
                
            # WARD LAYER
            ward_boundaries = {}
            for wid in ward_ids:
                bounds = topology.get_ward_boundaries(wid)
                ward_edges = bounds.get("spawn_candidates", [])
                ward_boundaries[wid] = ward_edges
                
                summary = sumo_env.get_ward_summary(ward_edges)
                obs = np.array([
                    summary["throughput"],
                    summary["queue"],
                    summary["queue"] / max(summary["throughput"], 1),
                    summary["congestion"],
                    summary["congestion"],
                    summary["avg_speed"],
                    0.0,
                    summary["ambulance_flag"],
                    0.0,
                    0.0,
                    gnn_predictions.get(wid, 0.0),
                    city_caps.get(ward_to_area.get(wid, ""), 1.0)
                ], dtype=np.float32)
                
                action = ward_agents[wid].get_action(obs)
                speed_samples.append(summary["avg_speed"])
                
            # AREA LAYER
            if tick % AREA_INTERVAL == 0 and tick > 0:
                for aid in target_areas:
                    area_ward_ids = topology.get_area_wards(aid)
                    ward_summaries = {}
                    for wid in area_ward_ids:
                        edges = ward_boundaries.get(wid, [])
                        ward_summaries[wid] = sumo_env.get_ward_summary(edges)
                        
                    predictions = area_forecasters[aid].predict(
                        ward_summaries, city_caps.get(aid, 1.0)
                    )
                    for wid, pressure in predictions.items():
                        gnn_predictions[wid] = pressure
                        
                logger.info(f"[Tick {tick}] Area GNN updated pressure directives.")
                
            # CITY LAYER
            if tick % CITY_INTERVAL == 0 and tick > 0:
                area_summaries = {}
                for aid in target_areas:
                    area_ward_ids = topology.get_area_wards(aid)
                    congestions = []
                    for wid in area_ward_ids:
                        edges = ward_boundaries.get(wid, [])
                        summary = sumo_env.get_ward_summary(edges)
                        congestions.append(summary["congestion"])
                        
                    area_summaries[aid] = {
                        "avg_congestion": np.mean(congestions) if congestions else 0.0,
                        "total_throughput": float(len(sumo_env.get_vehicle_ids())),
                        "incident_severity": 0.0
                    }
                    
                city_caps = city_controller.solve(area_summaries)
                logger.info(f"[Tick {tick}] City Controller updated area capacities: {city_caps}")
                
            total_arrived += sumo_env.get_arrived_count()
            
    except Exception as exc:
        logger.error(f"Simulation crashed: {exc}")
    finally:
        sumo_env.stop()
        
    elapsed = time.time() - start_time
    print("\n======================================================")
    print("      Stitched Multi-Area Coordination Complete       ")
    print("======================================================")
    print(f"Total Ticks: {tick + 1}")
    print(f"Total Vehicles Arrived: {total_arrived}")
    print(f"Average Vehicle Speed: {np.mean(speed_samples):.2f} m/s")
    print(f"Real-world Execution Time: {elapsed:.2f} seconds")
    print("======================================================")

if __name__ == "__main__":
    main()
