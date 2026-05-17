"""Bidirectional multi-timescale simulation orchestrator.

This module is the kernel of the HMRL system. It owns the main simulation
loop and coordinates data flow between all three control layers.

**No training logic here.** Training happens exclusively in notebooks
and ``src/rl/train.py``.

Control timescales:
    Ward RL agent  : every 1 simulation second
    Area GNN       : every 30 simulation seconds
    City graph     : every 120 simulation seconds OR event-triggered
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.sumo_env import SumoEnv
from src.topology import Topology
from src.traffic_generator import TrafficGenerator
from src.controllers.city_controller import CityController
from src.controllers.area_controller import AreaForecaster
from src.controllers.ward_agent import WardAgent
from src.evaluation.metrics import SimulationMetrics

try:
    from dashboard.server import update_state as _dash_update, start_background as _dash_start
    _HAS_DASHBOARD = True
except ImportError:
    _HAS_DASHBOARD = False
    def _dash_update(*a, **kw): pass
    def _dash_start(*a, **kw): pass

logger = logging.getLogger(__name__)

# Control timescales (in simulation ticks / seconds)
WARD_INTERVAL = 1
AREA_INTERVAL = 30
CITY_INTERVAL = 120


def run_simulation(
    scope: str,
    identifier: str,
    project_root: Path,
    gui: bool = False,
    scenario_id: str = "normal",
    max_ticks: int = 3600,
    algorithm: str = "ppo",
    dashboard: bool = True,
    gui_delay_ms: float = 150.0,
) -> dict[str, Any]:
    """Execute a full hierarchical simulation with all three control layers.

    Args:
        scope: Simulation scope (``"ward"``, ``"area"``, ``"city"``).
        identifier: Scope identifier (e.g. ward ID or area ID).
        project_root: Project root directory.
        gui: Whether to show SUMO GUI.
        scenario_id: Traffic scenario to use.
        max_ticks: Maximum simulation duration in seconds.
        algorithm: RL algorithm used for ward agents.
        dashboard: Whether to start the live dashboard server.
        gui_delay_ms: Default delay (in ms) to apply in SUMO GUI mode.

    Returns:
        Simulation result dictionary with metrics and metadata.
    """
    logger.info(
        "Starting simulation: scope=%s, id=%s, scenario=%s, ticks=%d",
        scope, identifier, scenario_id, max_ticks,
    )

    # ---- Start dashboard server (background thread) ----
    if dashboard and _HAS_DASHBOARD:
        _dash_start(port=5050)
        _dash_update({"scope": scope, "identifier": identifier, "scenario": scenario_id})

    # ---- Resolve topology ----
    topology = Topology(project_root)

    if scope == "ward":
        ward_ids = [identifier]
        area_ids = [topology.get_ward_area(identifier)]
    elif scope == "area":
        ward_ids = topology.get_area_wards(identifier)
        area_ids = [identifier]
    elif scope == "city":
        area_ids = topology.get_all_area_ids()
        ward_ids = []
        for aid in area_ids:
            ward_ids.extend(topology.get_area_wards(aid))
    else:
        raise ValueError(f"Unknown scope: {scope}")

    # ---- Build ward → area lookup ----
    ward_to_area: dict[str, str] = {}
    for wid in ward_ids:
        ward_to_area[wid] = topology.get_ward_area(wid)

    # ---- Initialise SUMO ----
    sumo_env = SumoEnv()

    # Use first ward's sumocfg for now (multi-ward stitching is Phase 5+)
    primary_ward = ward_ids[0]
    sumocfg = project_root / "maps" / "processed" / primary_ward / "ward.sumocfg"
    if not sumocfg.exists():
        raise FileNotFoundError(f"No sumocfg found: {sumocfg}")

    # ---- Initialise traffic generator ----
    traffic_gen = TrafficGenerator(
        project_root, primary_ward, scenario_id=scenario_id,
    )

    # ---- Load ward agents ----
    ward_agents: dict[str, WardAgent] = {}
    for wid in ward_ids:
        model_path = project_root / "models" / algorithm / wid / "model.pt"
        ward_agents[wid] = WardAgent(wid, model_path)

    # ---- Initialise area forecasters ----
    area_forecasters: dict[str, AreaForecaster] = {}
    for aid in area_ids:
        model_dir = project_root / "models" / "gnn"
        area_forecasters[aid] = AreaForecaster(aid, topology, model_dir)

    # ---- Initialise city controller ----
    city_controller = CityController(topology)

    # ---- State buffers ----
    city_caps: dict[str, float] = {aid: 1.0 for aid in area_ids}
    gnn_predictions: dict[str, float] = {wid: 0.0 for wid in ward_ids}

    # ---- Metrics tracking ----
    total_arrived = 0
    total_wait = 0.0
    speed_samples: list[float] = []

    # ---- Main simulation loop ----
    sumo_env.start(str(sumocfg), gui=gui, delay_ms=gui_delay_ms)
    logger.info("Simulation started with %d wards, %d areas", len(ward_ids), len(area_ids))

    start_time = time.time()

    try:
        for tick in range(max_ticks):
            # Step simulation
            traffic_gen.step(sumo_env, tick)
            sumo_env.step()

            # Check if simulation is complete
            if sumo_env.get_min_expected_number() <= 0 and tick > 10:
                logger.info("No more vehicles expected, ending at tick %d", tick)
                break

            # ---- WARD LAYER: every tick ----
            ward_boundaries = {}
            ward_summaries_dash: dict[str, dict] = {}
            ward_actions_dash: dict[str, str] = {}
            for wid in ward_ids:
                boundaries = topology.get_ward_boundaries(wid)
                ward_edges = boundaries.get("spawn_candidates", [])
                ward_boundaries[wid] = ward_edges

                # Build 12-dim observation
                summary = sumo_env.get_ward_summary(ward_edges)
                obs = np.array([
                    summary["throughput"],      # 0: vehicle count
                    summary["queue"],           # 1: queue count
                    summary["queue"] / max(summary["throughput"], 1),  # 2: queue ratio
                    summary["congestion"],      # 3: wait proxy
                    summary["congestion"],      # 4: max wait proxy
                    summary["avg_speed"],       # 5: avg speed
                    0.0,                        # 6: avg route length (placeholder)
                    summary["ambulance_flag"],  # 7: ambulance count
                    0.0,                        # 8: aggressive count (placeholder)
                    0.0,                        # 9: heavy vehicle count (placeholder)
                    gnn_predictions.get(wid, 0.0),   # 10: area directive
                    city_caps.get(ward_to_area.get(wid, ""), 1.0),  # 11: city cap
                ], dtype=np.float32)

                action = ward_agents[wid].get_action(obs)

                # Track for dashboard and metrics
                speed_samples.append(summary["avg_speed"])
                ward_summaries_dash[wid] = {
                    **summary,
                    "pressure": gnn_predictions.get(wid, 0.0),
                }
                ward_actions_dash[wid] = f"action_{action}"

            # Push live state to dashboard every tick
            _dash_update({
                "tick": tick,
                "elapsed": time.time() - start_time,
                "total_arrived": total_arrived,
                "ward_states": ward_summaries_dash,
                "ward_actions": ward_actions_dash,
                "area_predictions": gnn_predictions,
                "city_caps": city_caps,
                "metrics": {
                    "avg_speed": float(np.mean(speed_samples[-30:])) if speed_samples else 0.0,
                    "total_vehicles": len(sumo_env.get_vehicle_ids()),
                    "total_queue": int(sum(ws["queue"] for ws in ward_summaries_dash.values())),
                    "throughput": total_arrived,
                },
            })

            # ---- AREA LAYER: every 30 ticks ----
            if tick % AREA_INTERVAL == 0 and tick > 0:
                for aid in area_ids:
                    area_ward_ids = topology.get_area_wards(aid)
                    ward_summaries = {}
                    for wid in area_ward_ids:
                        edges = ward_boundaries.get(wid, [])
                        ward_summaries[wid] = sumo_env.get_ward_summary(edges)

                    predictions = area_forecasters[aid].predict(
                        ward_summaries, city_caps.get(aid, 1.0),
                    )
                    for wid, pressure in predictions.items():
                        gnn_predictions[wid] = pressure

                    logger.debug(
                        "Area %s predictions: %s",
                        aid,
                        {k: f"{v:.2f}" for k, v in predictions.items()},
                    )

            # ---- CITY LAYER: every 120 ticks or event-triggered ----
            area_summaries: dict[str, dict[str, float]] = {}
            if tick % CITY_INTERVAL == 0 and tick > 0:
                for aid in area_ids:
                    area_ward_ids = topology.get_area_wards(aid)
                    congestions = []
                    for wid in area_ward_ids:
                        edges = ward_boundaries.get(wid, [])
                        summary = sumo_env.get_ward_summary(edges)
                        congestions.append(summary["congestion"])

                    area_summaries[aid] = {
                        "avg_congestion": np.mean(congestions) if congestions else 0.0,
                        "total_throughput": float(len(sumo_env.get_vehicle_ids())),
                        "incident_severity": 0.0,
                    }

                city_caps = city_controller.solve(area_summaries)

            # Event-triggered city check
            elif area_summaries or city_controller.should_trigger(area_summaries):
                city_caps = city_controller.solve(area_summaries)

            # Track arrivals
            total_arrived += sumo_env.get_arrived_count()

    except Exception as exc:
        logger.error("Simulation error at tick %d: %s", tick, exc)
        raise
    finally:
        sumo_env.stop()

    elapsed = time.time() - start_time

    # ---- Compile results ----
    results = {
        "scope": scope,
        "identifier": identifier,
        "scenario_id": scenario_id,
        "algorithm": algorithm,
        "total_ticks": tick + 1,
        "total_arrived": total_arrived,
        "avg_speed": float(np.mean(speed_samples)) if speed_samples else 0.0,
        "elapsed_seconds": elapsed,
        "city_caps_final": city_caps,
        "gnn_predictions_final": gnn_predictions,
    }

    # Save to results
    results_dir = project_root / "results" / "inference"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"{scope}_{identifier}_{scenario_id}.json"
    with results_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)

    logger.info("Simulation complete: %d ticks, %d arrived, saved → %s",
                tick + 1, total_arrived, results_path)

    return results
