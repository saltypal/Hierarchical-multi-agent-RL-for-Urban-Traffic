"""Dynamic traffic generation engine.

Responsible for route file creation, vehicle spawning, breakdown injection,
and incident simulation. Uses ``SumoEnv`` for all simulator interaction —
never imports ``traci`` directly.

Traffic generation modes:

**Training (single ward):**
    70% of vehicles spawn at boundary ingress edges (weighted by lane count)
    to simulate inter-ward pressure. 30% spawn at random internal edges.
    Destinations are random egress edges, weighted by lane count.

**Evaluation (stitched area):**
    Vehicles spawn at the area's external boundary edges and route freely
    across all wards in the stitched network. Tests whether independently-
    trained ward agents can coordinate.
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

from configs.vehicle_profiles import VEHICLE_PROFILES, build_all_vtypes_xml
from configs.scenarios import get_scenario
from configs.traffic_profiles import get_zone_profile

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Edge list helpers (backward-compatible with old flat-string format)
# ------------------------------------------------------------------

def _normalize_edge_list(raw: list) -> list[dict[str, Any]]:
    """Convert edge entries to the new dict format.

    Handles both the old format (list of strings) and the new format
    (list of dicts with ``edge_id`` and ``lanes``).
    """
    if not raw:
        return []
    if isinstance(raw[0], str):
        return [{"edge_id": eid, "lanes": 1} for eid in raw]
    return raw


def _weighted_choice(
    edges: list[dict[str, Any]],
    rng: random.Random,
) -> str:
    """Pick an edge ID weighted by lane count."""
    if not edges:
        return "unknown"
    ids = [e["edge_id"] for e in edges]
    weights = [e["lanes"] for e in edges]
    return rng.choices(ids, weights=weights, k=1)[0]


class TrafficGenerator:
    """Stochastic traffic ecosystem engine.

    Creates route files, spawns vehicles dynamically, and injects
    disturbances according to scenario configurations.
    """

    def __init__(
        self,
        project_root: Path,
        ward_id: str,
        scenario_id: str = "normal",
        seed: int = 42,
    ) -> None:
        self.project_root = project_root
        self.ward_id = ward_id
        self.scenario = get_scenario(scenario_id)
        self.rng = random.Random(seed)
        self._spawn_counter = 0
        self._breakdown_injected = False

        # Load ward assets
        self._boundaries = self._load_boundaries()
        self._zone_profile = self._load_zone_profile()

    # ------------------------------------------------------------------
    # Asset loading
    # ------------------------------------------------------------------

    def _load_boundaries(self) -> dict[str, Any]:
        path = (
            self.project_root / "maps" / os.getenv("HMRL_MAP_DIR", "processed")
            / self.ward_id / "boundaries.json"
        )
        if not path.exists():
            logger.warning("No boundaries.json for %s", self.ward_id)
            return {
                "spawn_candidates": [],
                "valid_egress_edges": [],
                "internal_edges": [],
            }
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def _load_zone_profile(self) -> dict[str, Any]:
        registry_path = (
            self.project_root / "configs" / "hierarchy" / "ward_registry.json"
        )
        with registry_path.open("r", encoding="utf-8") as fh:
            registry = json.load(fh)
        ward_meta = registry.get("wards", {}).get(self.ward_id, {})
        zone_type = ward_meta.get("zone_type", "mixed")

        # Apply congestion_prior bonus to boundary pressure
        congestion = ward_meta.get("congestion_prior", "medium")
        self._congestion_bonus = {"low": 0.0, "medium": 0.05, "high": 0.1}.get(
            congestion, 0.0
        )

        return get_zone_profile(zone_type)

    # ------------------------------------------------------------------
    # Route file generation (offline, pre-simulation)
    # ------------------------------------------------------------------

    def generate_ward_routes(self, num_vehicles: int | None = None) -> Path:
        """Generate a ``.rou.xml`` file for a single ward (training mode).

        Uses boundary pressure spawning: the majority of vehicles enter
        at boundary ingress edges (weighted by lane count) to simulate
        inter-ward traffic pressure. A smaller fraction originates from
        internal edges.

        Args:
            num_vehicles: Optional override for vehicle count.

        Returns:
            Path to the generated route file.
        """
        output_dir = self.project_root / "maps" / os.getenv("HMRL_MAP_DIR", "processed") / self.ward_id
        output_dir.mkdir(parents=True, exist_ok=True)
        route_path = output_dir / "ward.rou.xml"

        ingress_edges = _normalize_edge_list(
            self._boundaries.get("spawn_candidates", [])
        )
        egress_edges = _normalize_edge_list(
            self._boundaries.get("valid_egress_edges", [])
        )
        internal_edges = _normalize_edge_list(
            self._boundaries.get("internal_edges", [])
        )

        if not ingress_edges or not egress_edges:
            logger.warning(
                "Insufficient boundary edges for %s: %d spawn, %d egress",
                self.ward_id, len(ingress_edges), len(egress_edges),
            )

        # Compute vehicle count from scenario + zone profile
        if num_vehicles is not None:
            base_count = num_vehicles
        else:
            base_count = int(
                100
                * self._zone_profile["spawn_intensity"]
                * self.scenario["spawn_multiplier"]
                * 1.15  # 15% over-sampling for disconnected trip compensation
            )

        # Boundary spawn fraction from scenario config
        boundary_fraction = self.scenario.get("boundary_spawn_fraction", 0.7)
        # Apply congestion prior bonus (high-congestion wards get more boundary pressure)
        boundary_fraction = min(1.0, boundary_fraction + self._congestion_bonus)

        lines: list[str] = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"'
            ' xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">',
            "",
            build_all_vtypes_xml(),
            "",
        ]

        # Generate vehicle trips
        depart = 0.0
        for i in range(base_count):
            vtype = self._sample_vehicle_type()

            # Decide origin: boundary (ingress) or internal
            if self.rng.random() < boundary_fraction and ingress_edges:
                origin = _weighted_choice(ingress_edges, self.rng)
            elif internal_edges:
                origin = _weighted_choice(internal_edges, self.rng)
            elif ingress_edges:
                origin = _weighted_choice(ingress_edges, self.rng)
            else:
                origin = "unknown"

            # Destination: always egress (weighted by lanes)
            destination = _weighted_choice(egress_edges, self.rng)

            veh_id = f"{self.ward_id}_{vtype}_{i}"
            depart += self.rng.uniform(0.5, 3.0)

            # Ensure we extract the edge_id string if origin/destination is a dictionary
            orig_id = origin["edge_id"] if isinstance(origin, dict) else origin
            dest_id = destination["edge_id"] if isinstance(destination, dict) else destination

            lines.append(
                f'    <trip id="{veh_id}" type="{vtype}" depart="{depart:.1f}"'
                f' from="{orig_id}" to="{dest_id}"/>'
            )

        lines.append("</routes>")

        route_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(
            "Generated %d trips (%.0f%% boundary) → %s",
            base_count, boundary_fraction * 100, route_path,
        )
        return route_path

    def generate_area_routes(
        self,
        area_id: str,
        topology: Any,
        num_vehicles: int | None = None,
    ) -> Path:
        """Generate routes for a stitched multi-ward area (evaluation mode).

        Vehicles spawn at the area's external boundary edges and SUMO
        routes them freely across all constituent wards.

        Args:
            area_id: Area identifier (e.g. ``"Basavanagudi"``).
            topology: ``Topology`` instance for ward/area lookups.
            num_vehicles: Optional override for vehicle count.

        Returns:
            Path to the generated ``.rou.xml`` file.
        """
        ward_ids = topology.get_area_wards(area_id)
        if not ward_ids:
            raise ValueError(f"No wards found for area: {area_id}")

        # Stitch ward maps
        stitched_net = topology.stitch_ward_maps(ward_ids)
        output_dir = stitched_net.parent
        route_path = output_dir / f"{area_id}.rou.xml"

        # Collect area's external boundary edges
        # An external ingress edge is one that belongs to a perimeter ward
        # and doesn't connect to a sibling ward
        ward_set = set(ward_ids)
        area_ingress: list[dict[str, Any]] = []
        area_egress: list[dict[str, Any]] = []

        for wid in ward_ids:
            neighbors = topology.get_ward_neighbors(wid)
            has_external_face = any(n not in ward_set for n in neighbors) or len(neighbors) == 0

            if has_external_face:
                bounds = topology.get_ward_boundaries(wid)
                area_ingress.extend(
                    _normalize_edge_list(bounds.get("valid_ingress_edges", []))
                )
                area_egress.extend(
                    _normalize_edge_list(bounds.get("valid_egress_edges", []))
                )

        # Fallback: if filtering left nothing, use all ward boundaries
        if not area_ingress or not area_egress:
            for wid in ward_ids:
                bounds = topology.get_ward_boundaries(wid)
                area_ingress.extend(
                    _normalize_edge_list(bounds.get("valid_ingress_edges", []))
                )
                area_egress.extend(
                    _normalize_edge_list(bounds.get("valid_egress_edges", []))
                )

        # Compute vehicle count
        if num_vehicles is not None:
            total_count = num_vehicles
        else:
            total_count = int(
                100
                * len(ward_ids)
                * self.scenario["spawn_multiplier"]
                * 1.15  # 15% over-sampling for disconnected trip compensation
            )

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
            vtype = self._sample_vehicle_type()
            origin = _weighted_choice(area_ingress, self.rng)
            destination = _weighted_choice(area_egress, self.rng)
            veh_id = f"{area_id}_{vtype}_{i}"
            depart += self.rng.uniform(0.5, 3.0)

            orig_id = origin["edge_id"] if isinstance(origin, dict) else origin
            dest_id = destination["edge_id"] if isinstance(destination, dict) else destination

            lines.append(
                f'    <trip id="{veh_id}" type="{vtype}" depart="{depart:.1f}"'
                f' from="{orig_id}" to="{dest_id}"/>'
            )

        lines.append("</routes>")
        route_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(
            "Generated %d area-level trips for %s (%d wards) → %s",
            total_count, area_id, len(ward_ids), route_path,
        )
        return route_path

    def generate_ward_sumocfg(self) -> Path:
        """Create a ``.sumocfg`` file for the ward.

        Returns:
            Path to the generated configuration file.
        """
        processed_dir = self.project_root / "maps" / os.getenv("HMRL_MAP_DIR", "processed") / self.ward_id
        net_path = processed_dir / "ward.net.xml"
        route_path = processed_dir / "ward.rou.xml"
        cfg_path = processed_dir / "ward.sumocfg"

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
        logger.info("Generated sumocfg → %s", cfg_path)
        return cfg_path

    # ------------------------------------------------------------------
    # Dynamic per-tick operations (called during simulation)
    # ------------------------------------------------------------------

    def step(self, sumo_env: Any, tick: int) -> None:
        """Per-tick traffic operations: dynamic spawning, breakdowns, incidents.

        Args:
            sumo_env: ``SumoEnv`` instance (sole simulator interface).
            tick: Current simulation tick number.
        """
        self._inject_breakdown(sumo_env, tick)

    def _inject_breakdown(self, sumo_env: Any, tick: int) -> None:
        """Randomly stop a vehicle to simulate a breakdown."""
        if self.scenario["breakdown_prob"] <= 0:
            return

        if self.rng.random() < self.scenario["breakdown_prob"]:
            vehicles = sumo_env.get_vehicle_ids()
            if vehicles:
                broken = self.rng.choice(vehicles)
                sumo_env.set_vehicle_speed(broken, 0.0)
                logger.info("Breakdown injected: %s at tick %d", broken, tick)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sample_vehicle_type(self) -> str:
        """Sample a vehicle type according to scenario mix ratios."""
        mix = self.scenario["vehicle_mix"]
        types = list(mix.keys())
        weights = list(mix.values())
        return self.rng.choices(types, weights=weights, k=1)[0]
