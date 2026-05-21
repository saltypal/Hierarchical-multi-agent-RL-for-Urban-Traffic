"""Bidirectional multi-timescale simulation orchestrator.

This module owns the live simulation loop and coordinates data flow
between the ward RL layer, the area GNN layer, and the city controller.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from src.controllers.area_controller import AreaForecaster
from src.controllers.city_controller import CityController
from src.controllers.temporal_features import (
    WARD_FEATURE_DIM,
    WARD_TEMPORAL_WINDOW,
    build_ward_feature_frame,
    pad_history,
)
from src.controllers.ward_agent import WardAgent
from src.rl.ward_actions import WardAction
from src.sumo_env import SumoEnv
from src.topology import Topology
from src.traffic_generator import TrafficGenerator, _normalize_edge_list, _weighted_choice

try:
    from dashboard.server import update_state as _dash_update, start_background as _dash_start
    _HAS_DASHBOARD = True
except ImportError:
    _HAS_DASHBOARD = False

    def _dash_update(*a, **kw):  # type: ignore[return-type]
        pass

    def _dash_start(*a, **kw):  # type: ignore[return-type]
        pass

logger = logging.getLogger(__name__)

WARD_INTERVAL = 30
AREA_INTERVAL = 60
CITY_INTERVAL = 120


def _load_registry(project_root: Path) -> dict[str, Any]:
    path = project_root / "configs" / "hierarchy" / "ward_registry.json"
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _default_area_ids(project_root: Path, count: int = 2) -> list[str]:
    registry = _load_registry(project_root)
    area_ids = list(registry.get("areas", {}).keys())
    return area_ids[:count]


def _selected_ward_ids(project_root: Path, area_ids: list[str]) -> list[str]:
    registry = _load_registry(project_root)
    wards = registry.get("wards", {})
    selected: list[str] = []
    for area_id in area_ids:
        area_wards = [
            wid for wid, meta in wards.items()
            if meta.get("parent_area") == area_id
        ]
        selected.extend(sorted(area_wards))
    return selected


def _resolve_ward_model_path(project_root: Path, algorithm: str, ward_id: str) -> Path:
    model_root = project_root / "models" / algorithm
    candidates = [
        model_root / ward_id / "model.pt",
        model_root / "global_agent" / "model.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[-1]


def _build_ward_vehicle_ids(sumo_env: SumoEnv, ward_edges: list[str]) -> list[str]:
    vehicle_ids: list[str] = []
    for edge_id in ward_edges:
        vehicle_ids.extend(sumo_env.get_edge_vehicle_ids(edge_id))
    return sorted(set(vehicle_ids))


def _vehicle_matches(sumo_env: SumoEnv, vehicle_id: str, tokens: list[str]) -> bool:
    vtype = sumo_env.get_vehicle_type(vehicle_id)
    haystack = f"{vehicle_id} {vtype}".lower()
    return any(token in haystack for token in tokens)


def _most_congested_edge(sumo_env: SumoEnv, ward_edges: list[str]) -> str | None:
    # Prefer ward-specific edges; fall back to ALL sim edges if empty
    # (matches training behaviour in sb3_ward_adapter._most_congested_edge)
    candidate_edges = ward_edges if ward_edges else sumo_env.get_edge_ids()
    if not candidate_edges:
        return None
    return max(candidate_edges, key=sumo_env.get_edge_halting_count)


def _prime_edge_traveltimes(sumo_env: SumoEnv, ward_edges: list[str]) -> None:
    """Set edge travel times from measured speeds so rerouteTraveltime() works.

    By default SUMO uses free-flow speeds for rerouting, meaning rerouteTraveltime()
    finds the same path every time. This primes the edge cost table with current
    measured speeds so congested edges get higher travel times and vehicles are
    actually redirected to genuinely faster routes.
    """
    all_edges = ward_edges if ward_edges else sumo_env.get_edge_ids()
    for edge_id in all_edges:
        veh_ids = sumo_env.get_edge_vehicle_ids(edge_id)
        if not veh_ids:
            continue
        speeds = [sumo_env.get_vehicle_speed(v) for v in veh_ids]
        avg_speed = max(sum(speeds) / len(speeds), 0.1)
        # Higher cost = more congested; use 100m/avg_speed as proxy travel time
        cost = 100.0 / avg_speed
        sumo_env.adapt_edge_traveltime(edge_id, cost)
        _adapted_edges.setdefault("_prime", set()).add(edge_id)


def _build_ward_observation(
    sumo_env: SumoEnv,
    ward_edges: list[str],
    gnn_pressure: float,
    city_cap: float,
) -> np.ndarray:
    vehicle_ids = _build_ward_vehicle_ids(sumo_env, ward_edges)
    vehicle_count = len(vehicle_ids)

    if vehicle_count == 0:
        return np.zeros(12, dtype=np.float32)

    speeds = [sumo_env.get_vehicle_speed(v) for v in vehicle_ids]
    waits = [sumo_env.get_vehicle_waiting_time(v) for v in vehicle_ids]
    route_lengths = [len(sumo_env.get_vehicle_route(v)) for v in vehicle_ids]
    queue_count = sum(1 for speed in speeds if speed < 0.1)

    ambulance_count = sum(
        1 for v in vehicle_ids if _vehicle_matches(sumo_env, v, ["ambulance"])
    )
    aggressive_count = sum(
        1 for v in vehicle_ids if _vehicle_matches(sumo_env, v, ["aggressive", "rash"])
    )
    heavy_count = sum(
        1 for v in vehicle_ids if _vehicle_matches(
            sumo_env, v, ["truck", "bus", "bmtc", "heavy"]
        )
    )

    return np.asarray([
        float(vehicle_count),
        float(queue_count),
        float(queue_count / vehicle_count),
        float(sum(waits)),
        float(max(waits)),
        float(sum(speeds) / vehicle_count),
        float(sum(route_lengths) / vehicle_count),
        float(ambulance_count),
        float(aggressive_count),
        float(heavy_count),
        float(gnn_pressure),
        float(city_cap),
    ], dtype=np.float32)


# Track edges that have had their traveltime adapted so we can reset them
_adapted_edges: dict[str, set[str]] = {}  # ward_id -> set of edge_ids


def _reset_adapted_edges(sumo_env: SumoEnv, ward_id: str) -> None:
    """Reset previously adapted edge travel times to allow SUMO to recalculate."""
    adapted = _adapted_edges.get(ward_id, set())
    for edge_id in adapted:
        try:
            sumo_env.adapt_edge_traveltime(edge_id, -1.0)  # -1 = reset to default
        except Exception:
            pass
    _adapted_edges[ward_id] = set()


def _apply_ward_action(
    sumo_env: SumoEnv,
    action_idx: int,
    ward_edges: list[str],
    held_vehicle_ids: set[str],
    ward_id: str = "",
) -> None:
    action = WardAction(int(action_idx))

    if action == WardAction.NO_OP:
        return

    if action in (
        WardAction.REROUTE_HOTSPOT_GROUP,
        WardAction.PRIORITIZE_ALTERNATE_EDGE,
    ):
        _prime_edge_traveltimes(sumo_env, ward_edges)
        edge_id = _most_congested_edge(sumo_env, ward_edges)
        if edge_id is None:
            return
        for v in sumo_env.get_edge_vehicle_ids(edge_id):
            sumo_env.reroute_vehicle(v)
        return

    if action == WardAction.DEPRIORITIZE_MOST_CONGESTED_EDGE:
        edge_id = _most_congested_edge(sumo_env, ward_edges)
        if edge_id is None:
            return
        _prime_edge_traveltimes(sumo_env, ward_edges)
        # Use a moderate penalty (300s) instead of 9999 to avoid permanent gridlock
        sumo_env.adapt_edge_traveltime(edge_id, 300.0)
        _adapted_edges.setdefault(ward_id, set()).add(edge_id)
        for v in sumo_env.get_edge_vehicle_ids(edge_id):
            sumo_env.reroute_vehicle(v)
        return

    if action == WardAction.CLEAR_AMBULANCE_PATH:
        vehicle_ids = _build_ward_vehicle_ids(sumo_env, ward_edges)
        ambulance_ids = [
            v for v in vehicle_ids if _vehicle_matches(sumo_env, v, ["ambulance"])
        ]
        if not ambulance_ids:
            return
        _prime_edge_traveltimes(sumo_env, ward_edges)
        protected_edges: set[str] = set()
        for amb_id in ambulance_ids:
            route = sumo_env.get_vehicle_route(amb_id)
            idx = sumo_env.get_vehicle_route_index(amb_id)
            protected_edges.update(route[idx: idx + 3])
        for edge_id in protected_edges:
            for v in sumo_env.get_edge_vehicle_ids(edge_id):
                if v not in ambulance_ids:
                    sumo_env.reroute_vehicle(v)
        return

    if action == WardAction.INCIDENT_REROUTE:
        _prime_edge_traveltimes(sumo_env, ward_edges)
        for edge_id in ward_edges[:2]:
            for v in sumo_env.get_edge_vehicle_ids(edge_id):
                sumo_env.reroute_vehicle(v)
        return

    if action == WardAction.HOLD_COMMERCIAL_INFLOW:
        # Only hold on a small subset of edges (boundary-like) to avoid freezing entire ward
        hold_edges = ward_edges[:3]
        for edge_id in hold_edges:
            for v in sumo_env.get_edge_vehicle_ids(edge_id):
                sumo_env.set_vehicle_speed(v, 0.0)
                held_vehicle_ids.add(v)
        return

    if action == WardAction.RELEASE_HELD_FLOW:
        current_vehicles = set(sumo_env.get_vehicle_ids())
        for v in list(held_vehicle_ids):
            if v in current_vehicles:
                sumo_env.set_vehicle_speed(v, -1.0)
            held_vehicle_ids.discard(v)
        return

    if action == WardAction.REROUTE_AGGRESSIVE_DRIVERS:
        _prime_edge_traveltimes(sumo_env, ward_edges)
        for v in _build_ward_vehicle_ids(sumo_env, ward_edges):
            if _vehicle_matches(sumo_env, v, ["aggressive", "rash"]):
                sumo_env.reroute_vehicle(v)
        return

    if action == WardAction.REROUTE_HEAVY_VEHICLES:
        _prime_edge_traveltimes(sumo_env, ward_edges)
        for v in _build_ward_vehicle_ids(sumo_env, ward_edges):
            if _vehicle_matches(sumo_env, v, ["truck", "bus", "bmtc", "heavy"]):
                sumo_env.reroute_vehicle(v)


def _write_sumocfg(output_dir: Path, net_name: str, route_name: str, cfg_name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    net_path = output_dir / net_name
    route_path = output_dir / route_name
    cfg_path = output_dir / cfg_name

    cfg_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="{net_path}"/>
        <route-files value="{route_path}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="3600"/>
    </time>
    <processing>
        <ignore-route-errors value="true"/>
    </processing>
</configuration>
"""
    cfg_path.write_text(cfg_content, encoding="utf-8")
    return cfg_path


def _generate_deployment_assets(
    project_root: Path,
    topology: Topology,
    traffic_gen: TrafficGenerator,
    selected_ward_ids: list[str],
    deployment_label: str,
) -> Path:
    stitched_net = topology.stitch_ward_maps(selected_ward_ids, output_name=deployment_label)
    output_dir = stitched_net.parent
    route_path = output_dir / f"{deployment_label}.rou.xml"

    ward_set = set(selected_ward_ids)
    ingress_edges: list[dict[str, Any]] = []
    egress_edges: list[dict[str, Any]] = []
    internal_edges: list[dict[str, Any]] = []

    for wid in selected_ward_ids:
        bounds = topology.get_ward_boundaries(wid)
        neighbors = topology.get_ward_neighbors(wid)
        is_boundary_ward = any(n not in ward_set for n in neighbors) or len(neighbors) == 0
        if is_boundary_ward:
            ingress_edges.extend(_normalize_edge_list(bounds.get("valid_ingress_edges", [])))
            egress_edges.extend(_normalize_edge_list(bounds.get("valid_egress_edges", [])))
        internal_edges.extend(_normalize_edge_list(bounds.get("internal_edges", [])))

    if not ingress_edges or not egress_edges:
        for wid in selected_ward_ids:
            bounds = topology.get_ward_boundaries(wid)
            ingress_edges.extend(_normalize_edge_list(bounds.get("valid_ingress_edges", [])))
            egress_edges.extend(_normalize_edge_list(bounds.get("valid_egress_edges", [])))

    # Parse stitched net file to find all valid edges actually present in the network
    valid_edge_ids = set()
    if stitched_net.exists():
        try:
            with stitched_net.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if "<edge id=" in line:
                        start = line.find('id="') + 4
                        end = line.find('"', start)
                        if start > 3 and end > start:
                            eid = line[start:end]
                            if not eid.startswith(":"):
                                valid_edge_ids.add(eid)
        except Exception as e:
            logger.warning("Could not parse stitched net for valid edges: %s", e)

    # Filter ingress, egress, and internal edges to only keep those present in valid_edge_ids
    if valid_edge_ids:
        ingress_edges = [
            e for e in ingress_edges 
            if (e["edge_id"] if isinstance(e, dict) else e) in valid_edge_ids
        ]
        egress_edges = [
            e for e in egress_edges 
            if (e["edge_id"] if isinstance(e, dict) else e) in valid_edge_ids
        ]
        internal_edges = [
            e for e in internal_edges 
            if (e["edge_id"] if isinstance(e, dict) else e) in valid_edge_ids
        ]

    total_count = int(
        100 * len(selected_ward_ids) * traffic_gen.scenario["spawn_multiplier"] * 1.15
    )
    boundary_fraction = traffic_gen.scenario.get("boundary_spawn_fraction", 0.7)

    from configs.vehicle_profiles import build_all_vtypes_xml

    lines: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"'
        ' xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">',
        "",
        build_all_vtypes_xml(),
        "",
    ]

    depart = 0.0
    for i in range(total_count):
        vtype = traffic_gen._sample_vehicle_type()
        if traffic_gen.rng.random() < boundary_fraction and ingress_edges:
            origin = _weighted_choice(ingress_edges, traffic_gen.rng)
        elif internal_edges:
            origin = _weighted_choice(internal_edges, traffic_gen.rng)
        elif ingress_edges:
            origin = _weighted_choice(ingress_edges, traffic_gen.rng)
        else:
            origin = "unknown"

        destination = _weighted_choice(egress_edges, traffic_gen.rng)
        veh_id = f"{deployment_label}_{vtype}_{i}"
        depart += traffic_gen.rng.uniform(0.5, 3.0)

        orig_id = origin["edge_id"] if isinstance(origin, dict) else origin
        dest_id = destination["edge_id"] if isinstance(destination, dict) else destination

        lines.append(
            f'    <trip id="{veh_id}" type="{vtype}" depart="{depart:.1f}"'
            f' from="{orig_id}" to="{dest_id}"/>'
        )

    lines.append("</routes>")
    route_path.write_text("\n".join(lines), encoding="utf-8")
    return _write_sumocfg(output_dir, f"{deployment_label}.net.xml", f"{deployment_label}.rou.xml", f"{deployment_label}.sumocfg")


def run_simulation(
    scope: str,
    identifier: str,
    project_root: Path,
    gui: bool = False,
    scenario_id: str = "normal",
    max_ticks: int = 3600,
    algorithm: str = "dqn",
    dashboard: bool = True,
    gui_delay_ms: float = 150.0,
    area_ids: list[str] | None = None,
    use_rl: bool = True,
    use_area: bool = True,
    use_city: bool = True,
    collect_tick_records: bool = False,
    persist_results: bool = True,
) -> dict[str, Any]:
    """Execute a hierarchical simulation with ward, area, and city layers."""
    logger.info(
        "Starting simulation: scope=%s, id=%s, scenario=%s, ticks=%d",
        scope, identifier, scenario_id, max_ticks,
    )

    if dashboard and _HAS_DASHBOARD:
        _dash_start(port=5050)
        _dash_update({"scope": scope, "identifier": identifier, "scenario": scenario_id})

    topology = Topology(project_root)
    algorithm = algorithm.lower()

    if scope == "ward":
        selected_area_ids = [topology.get_ward_area(identifier)]
        selected_ward_ids = [identifier]
    elif scope == "area":
        selected_area_ids = [identifier]
        selected_ward_ids = topology.get_area_wards(identifier)
    elif scope == "city":
        selected_area_ids = area_ids or _default_area_ids(project_root, count=2)
        selected_ward_ids = _selected_ward_ids(project_root, selected_area_ids)
    else:
        raise ValueError(f"Unknown scope: {scope}")

    if not selected_ward_ids:
        raise ValueError("No wards selected for simulation")

    ward_to_area = {
        wid: topology.get_ward_area(wid)
        for wid in selected_ward_ids
    }

    traffic_gen = TrafficGenerator(
        project_root,
        selected_ward_ids[0],
        scenario_id=scenario_id,
    )
    sumo_env = SumoEnv()

    if len(selected_ward_ids) == 1 and scope == "ward":
        map_dir = os.getenv("HMRL_MAP_DIR", "processed")
        sumocfg = project_root / "maps" / map_dir / selected_ward_ids[0] / "ward.sumocfg"
        if not sumocfg.exists():
            raise FileNotFoundError(f"No sumocfg found: {sumocfg}")
    else:
        deployment_label = "__".join(selected_area_ids)
        sumocfg = _generate_deployment_assets(
            project_root,
            topology,
            traffic_gen,
            selected_ward_ids,
            deployment_label,
        )

    ward_agents: dict[str, WardAgent] = {}
    if use_rl:
        for wid in selected_ward_ids:
            model_path = _resolve_ward_model_path(project_root, algorithm, wid)
            ward_agents[wid] = WardAgent(wid, model_path)

    area_forecasters: dict[str, AreaForecaster] = {}
    if use_area:
        for aid in selected_area_ids:
            area_forecasters[aid] = AreaForecaster(
                aid,
                topology,
                project_root / "models" / "gnn",
            )

    city_controller = CityController(topology, selected_area_ids) if use_city else None

    # Use ALL ward edges (internal + ingress) for summary & action targeting.
    # Using only spawn_candidates means queries return empty vehicle lists
    # because vehicles drive on interior edges, not entry edges — this caused
    # identical metrics between Baseline and RL (bug: actions hit zero vehicles).
    def _all_ward_edges(wid: str) -> list[dict]:
        bounds = topology.get_ward_boundaries(wid)
        combined: list = []
        combined.extend(_normalize_edge_list(bounds.get("internal_edges", [])))
        combined.extend(_normalize_edge_list(bounds.get("valid_ingress_edges", [])))
        combined.extend(_normalize_edge_list(bounds.get("valid_egress_edges", [])))
        if not combined:
            # absolute fallback: use spawn_candidates
            combined = _normalize_edge_list(bounds.get("spawn_candidates", []))
        # deduplicate while preserving order
        seen: set[str] = set()
        deduped: list[dict] = []
        for e in combined:
            eid = e["edge_id"] if isinstance(e, dict) else str(e)
            if eid not in seen:
                seen.add(eid)
                deduped.append(e if isinstance(e, dict) else {"edge_id": eid})
        return deduped

    ward_edges = {wid: _all_ward_edges(wid) for wid in selected_ward_ids}
    ward_hold_state: dict[str, set[str]] = {wid: set() for wid in selected_ward_ids}
    ward_pressure: dict[str, float] = {wid: 0.0 for wid in selected_ward_ids}
    ward_city_cap: dict[str, float] = {
        wid: 1.0 for wid in selected_ward_ids
    }
    last_ward_obs: dict[str, np.ndarray] = {
        wid: np.zeros(WARD_TEMPORAL_WINDOW * WARD_FEATURE_DIM, dtype=np.float32)
        for wid in selected_ward_ids
    }
    ward_histories: dict[str, deque[np.ndarray]] = {
        wid: deque(maxlen=WARD_TEMPORAL_WINDOW) for wid in selected_ward_ids
    }
    last_ward_action_names: dict[str, str] = {
        wid: WardAction.NO_OP.name for wid in selected_ward_ids
    }
    last_ward_state: dict[str, dict[str, float]] = {
        wid: {} for wid in selected_ward_ids
    }
    last_actions: dict[str, int] = {
        wid: WardAction.NO_OP.value for wid in selected_ward_ids
    }
    last_action_tick: dict[str, int | None] = {wid: None for wid in selected_ward_ids}
    reroute_count = 0

    total_arrived = 0
    speed_samples: list[float] = []
    queue_samples: list[float] = []
    congestion_samples: list[float] = []
    throughput_samples: list[float] = []
    trend_metrics: dict[str, list[float]] = {
        "avg_speed": [],
        "congestion_score": [],
        "queue_length": [],
        "throughput": [],
    }
    city_caps: dict[str, float] = {aid: 1.0 for aid in selected_area_ids}
    area_predictions: dict[str, float] = {
        wid: 0.0 for wid in selected_ward_ids
    }
    tick_records: list[dict[str, Any]] = []
    ward_tick_records: list[dict[str, Any]] = []
    active_vehicle_start: dict[str, int] = {}
    active_vehicle_type: dict[str, str] = {}
    completed_travel_times: list[float] = []
    completed_ambulance_times: list[float] = []
    incident_delay_accumulator = 0.0
    ambulance_delay_accumulator = 0.0
    last_area_tick: int | None = None
    last_city_tick: int | None = None
    last_ward_tick: int | None = None

    sumo_env.start(str(sumocfg), gui=gui, delay_ms=gui_delay_ms)
    logger.info(
        "Simulation started with %d wards across %d areas",
        len(selected_ward_ids), len(selected_area_ids),
    )

    for wid in selected_ward_ids:
        edges = [e["edge_id"] for e in ward_edges[wid]]
        summary = sumo_env.get_ward_summary(edges)
        frame = build_ward_feature_frame(summary)
        for _ in range(WARD_TEMPORAL_WINDOW):
            ward_histories[wid].append(frame.copy())
        last_ward_state[wid] = summary

    start_time = time.time()
    tick = 0

    try:
        for tick in range(max_ticks):
            traffic_gen.step(sumo_env, tick)
            sumo_env.step()
            current_vehicle_ids = set(sumo_env.get_vehicle_ids())
            previous_vehicle_ids = set(active_vehicle_start)

            for veh_id in current_vehicle_ids - previous_vehicle_ids:
                active_vehicle_start[veh_id] = tick
                active_vehicle_type[veh_id] = sumo_env.get_vehicle_type(veh_id).lower()

            for veh_id in previous_vehicle_ids - current_vehicle_ids:
                start_tick = active_vehicle_start.pop(veh_id, tick)
                travel_time = float(max(0, tick - start_tick))
                completed_travel_times.append(travel_time)
                veh_type = active_vehicle_type.pop(veh_id, "")
                if "ambulance" in veh_type:
                    completed_ambulance_times.append(travel_time)

            for wid in selected_ward_ids:
                edges = [e["edge_id"] for e in ward_edges[wid]]
                summary = sumo_env.get_ward_summary(edges)
                last_ward_state[wid] = summary
                ward_histories[wid].append(build_ward_feature_frame(summary))

            if sumo_env.get_min_expected_number() <= 0 and tick > 10:
                logger.info("No more vehicles expected, ending at tick %d", tick)
                break

            if tick % WARD_INTERVAL == 0:
                last_ward_tick = tick
                for wid in selected_ward_ids:
                    observation = pad_history(
                        ward_histories[wid],
                        WARD_TEMPORAL_WINDOW,
                        WARD_FEATURE_DIM,
                    ).reshape(-1)
                    last_ward_obs[wid] = observation
                    action = WardAction.NO_OP.value
                    if use_rl:
                        action = ward_agents[wid].get_action(observation)
                    last_actions[wid] = action
                    last_ward_action_names[wid] = WardAction(action).name
                    last_action_tick[wid] = tick
                    if use_rl:
                        edges = [e["edge_id"] for e in ward_edges[wid]]
                        # Reset previous edge travel time adaptations before applying new action
                        _reset_adapted_edges(sumo_env, wid)
                        _apply_ward_action(
                            sumo_env,
                            action,
                            edges,
                            ward_hold_state[wid],
                            ward_id=wid,
                        )
                        if WardAction(action) != WardAction.NO_OP:
                            reroute_count += 1
                    speed_samples.append(summary.get("avg_speed", 0.0))
                    queue_samples.append(summary.get("queue", 0.0))
                    congestion_samples.append(summary.get("congestion", 0.0))

            if use_area:
                for aid in selected_area_ids:
                    area_ward_ids = topology.get_area_wards(aid)
                    area_summary = {
                        wid: last_ward_state.get(wid, {})
                        for wid in area_ward_ids
                        if wid in selected_ward_ids
                    }
                    area_actions = {
                        wid: last_ward_action_names.get(wid, WardAction.NO_OP.name)
                        for wid in area_ward_ids
                        if wid in selected_ward_ids
                    }
                    area_forecasters[aid].ingest_tick(
                        area_summary,
                        ward_actions=area_actions,
                        city_cap=city_caps.get(aid, 1.0),
                    )

                if tick % AREA_INTERVAL == 0 and tick > 0:
                    last_area_tick = tick
                    for aid in selected_area_ids:
                        predictions = area_forecasters[aid].predict(
                            None,
                            city_cap=city_caps.get(aid, 1.0),
                            ingest=False,
                        )
                        for wid, pressure in predictions.items():
                            ward_pressure[wid] = pressure
                            area_predictions[wid] = pressure

            if use_city and city_controller is not None and tick % CITY_INTERVAL == 0 and tick > 0:
                last_city_tick = tick
                area_summaries: dict[str, dict[str, float]] = {}
                for aid in selected_area_ids:
                    area_ward_ids = topology.get_area_wards(aid)
                    area_congestion = [
                        last_ward_state.get(wid, {}).get("congestion", 0.0)
                        for wid in area_ward_ids
                        if wid in selected_ward_ids
                    ]
                    area_summaries[aid] = {
                        "avg_congestion": float(np.mean(area_congestion)) if area_congestion else 0.0,
                        "total_throughput": float(sum(
                            last_ward_state.get(wid, {}).get("throughput", 0.0)
                            for wid in area_ward_ids
                            if wid in selected_ward_ids
                        )),
                        "incident_severity": float(max(
                            last_ward_state.get(wid, {}).get("incident_flag", 0.0)
                            for wid in area_ward_ids
                            if wid in selected_ward_ids
                        )) if area_congestion else 0.0,
                    }

                city_caps = city_controller.solve(area_summaries)
                for wid in selected_ward_ids:
                    area_id = ward_to_area[wid]
                    ward_city_cap[wid] = city_caps.get(area_id, 1.0)

            ward_state_values = list(last_ward_state.values())
            avg_speed = float(np.mean([
                state.get("avg_speed", 0.0) for state in ward_state_values
            ])) if ward_state_values else 0.0
            avg_congestion = float(np.mean([
                state.get("congestion", 0.0) for state in ward_state_values
            ])) if ward_state_values else 0.0
            total_queue = float(sum(
                state.get("queue", 0.0) for state in ward_state_values
            ))
            avg_waiting_time = float(np.mean([
                state.get("waiting_time", 0.0) for state in ward_state_values
            ])) if ward_state_values else 0.0
            active_incident_delay = float(sum(
                state.get("waiting_time", 0.0)
                for state in ward_state_values
                if state.get("incident_flag", 0.0) > 0.0
            ))
            active_ambulance_delay = float(sum(
                state.get("waiting_time", 0.0)
                for state in ward_state_values
                if state.get("ambulance_flag", 0.0) > 0.0
            ))
            incident_delay_accumulator += active_incident_delay
            ambulance_delay_accumulator += active_ambulance_delay
            throughput_samples.append(float(total_arrived))

            for key, value in (
                ("avg_speed", avg_speed),
                ("congestion_score", avg_congestion),
                ("queue_length", total_queue),
                ("throughput", float(total_arrived)),
            ):
                trend_metrics[key].append(value)
                if len(trend_metrics[key]) > 60:
                    trend_metrics[key].pop(0)

            if collect_tick_records:
                tick_records.append({
                    "time": tick,
                    "avg_speed": avg_speed,
                    "congestion_score": avg_congestion,
                    "queue_length": total_queue,
                    "throughput": float(total_arrived),
                    "trip_completion": float(total_arrived),
                    "travel_time": float(np.mean(completed_travel_times)) if completed_travel_times else 0.0,
                    "waiting_time": avg_waiting_time,
                    "ambulance_delay": active_ambulance_delay,
                    "incident_delay": active_incident_delay,
                    "reroute_count": reroute_count,
                })
                for wid, state in last_ward_state.items():
                    action_name = last_ward_action_names.get(wid, WardAction.NO_OP.name)
                    ward_tick_records.append({
                        "time": tick,
                        "ward_id": wid,
                        "scenario_id": scenario_id,
                        "avg_speed": float(state.get("avg_speed", 0.0)),
                        "congestion_score": float(state.get("congestion", 0.0)),
                        "queue_length": float(state.get("queue", 0.0)),
                        "throughput": float(state.get("throughput", 0.0)),
                        "inflow": float(state.get("inflow", 0.0)),
                        "outflow": float(state.get("outflow", 0.0)),
                        "waiting_time": float(state.get("waiting_time", 0.0)),
                        "incident_flag": float(state.get("incident_flag", 0.0)),
                        "ambulance_flag": float(state.get("ambulance_flag", 0.0)),
                        "pressure": float(ward_pressure.get(wid, 0.0)),
                        "city_cap": float(ward_city_cap.get(wid, 1.0)),
                        "ppo_action": action_name,
                        "ppo_action_id": int(WardAction[action_name]),
                        "area_directive": float(ward_pressure.get(wid, 0.0)),
                        "city_directive": float(ward_city_cap.get(ward_to_area[wid], 1.0)),
                    })

            _dash_update({
                "tick": tick,
                "elapsed": time.time() - start_time,
                "total_arrived": total_arrived,
                "controller_status": {
                    "use_rl": use_rl,
                    "use_area": use_area,
                    "use_city": use_city,
                    "ward_interval": WARD_INTERVAL,
                    "area_interval": AREA_INTERVAL,
                    "city_interval": CITY_INTERVAL,
                    "last_ward_tick": last_ward_tick,
                    "last_area_tick": last_area_tick,
                    "last_city_tick": last_city_tick,
                },
                "ward_states": {
                    wid: {
                        **last_ward_state.get(wid, {}),
                        "pressure": ward_pressure.get(wid, 0.0),
                        "city_cap": ward_city_cap.get(wid, 1.0),
                        "last_action_tick": last_action_tick.get(wid),
                    }
                    for wid in selected_ward_ids
                },
                "ward_actions": {
                    wid: WardAction(last_actions[wid]).name
                    for wid in selected_ward_ids
                },
                "area_predictions": area_predictions,
                "city_caps": city_caps,
                "trend_metrics": trend_metrics,
                "metrics": {
                    "avg_speed": avg_speed,
                    "total_vehicles": len(current_vehicle_ids),
                    "total_queue": int(total_queue),
                    "throughput": total_arrived,
                    "congestion_score": avg_congestion,
                },
            })

            total_arrived += sumo_env.get_arrived_count()

    except Exception as exc:
        logger.error("Simulation error at tick %d: %s", tick, exc)
        raise
    finally:
        sumo_env.stop()

    elapsed = time.time() - start_time
    results = {
        "scope": scope,
        "identifier": identifier,
        "scenario_id": scenario_id,
        "algorithm": algorithm,
        "selected_areas": selected_area_ids,
        "selected_wards": selected_ward_ids,
        "total_ticks": tick + 1,
        "total_arrived": total_arrived,
        "avg_speed": float(np.mean(speed_samples)) if speed_samples else 0.0,
        "avg_congestion": float(np.mean(congestion_samples)) if congestion_samples else 0.0,
        "avg_queue": float(np.mean(queue_samples)) if queue_samples else 0.0,
        "avg_waiting_time": float(np.mean([
            state.get("waiting_time", 0.0) for state in last_ward_state.values()
        ])) if last_ward_state else 0.0,
        "avg_travel_time": float(np.mean(completed_travel_times)) if completed_travel_times else 0.0,
        "avg_ambulance_travel_time": float(np.mean(completed_ambulance_times)) if completed_ambulance_times else 0.0,
        "incident_delay": incident_delay_accumulator,
        "ambulance_delay": ambulance_delay_accumulator,
        "reroute_count": reroute_count,
        "elapsed_seconds": elapsed,
        "city_caps_final": city_caps,
        "gnn_predictions_final": area_predictions,
        "controller_status": {
            "use_rl": use_rl,
            "use_area": use_area,
            "use_city": use_city,
            "ward_interval": WARD_INTERVAL,
            "area_interval": AREA_INTERVAL,
            "city_interval": CITY_INTERVAL,
            "last_ward_tick": last_ward_tick,
            "last_area_tick": last_area_tick,
            "last_city_tick": last_city_tick,
        },
    }
    if collect_tick_records:
        results["tick_records"] = tick_records
        results["ward_tick_records"] = ward_tick_records

    results_path: Path | None = None
    if persist_results:
        results_dir = project_root / "results" / "inference"
        results_dir.mkdir(parents=True, exist_ok=True)
        results_path = results_dir / f"{scope}_{identifier}_{scenario_id}.json"
        with results_path.open("w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, default=str)

    logger.info(
        "Simulation complete: %d ticks, %d arrived, saved → %s",
        tick + 1, total_arrived, results_path,
    )
    return results
