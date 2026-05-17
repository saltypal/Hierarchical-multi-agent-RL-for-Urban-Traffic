"""Dynamic traffic generation engine.

Responsible for route file creation, vehicle spawning, breakdown injection,
and incident simulation. Uses ``SumoEnv`` for all simulator interaction —
never imports ``traci`` directly.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

from configs.vehicle_profiles import VEHICLE_PROFILES, build_all_vtypes_xml
from configs.scenarios import get_scenario
from configs.traffic_profiles import get_zone_profile

logger = logging.getLogger(__name__)


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
            self.project_root / "maps" / "processed"
            / self.ward_id / "boundaries.json"
        )
        if not path.exists():
            logger.warning("No boundaries.json for %s", self.ward_id)
            return {"spawn_candidates": [], "valid_egress_edges": []}
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def _load_zone_profile(self) -> dict[str, Any]:
        registry_path = (
            self.project_root / "configs" / "hierarchy" / "ward_registry.json"
        )
        with registry_path.open("r", encoding="utf-8") as fh:
            registry = json.load(fh)
        zone_type = registry.get("wards", {}).get(self.ward_id, {}).get("zone_type", "mixed")
        return get_zone_profile(zone_type)

    # ------------------------------------------------------------------
    # Route file generation (offline, pre-simulation)
    # ------------------------------------------------------------------

    def generate_ward_routes(self, num_vehicles: int | None = None) -> Path:
        """Generate a ``.rou.xml`` file for the ward.

        Uses boundary edges as origins/destinations, vehicle profiles
        for vType definitions, and scenario mix ratios for demand.

        Args:
            num_vehicles: Optional override for the number of vehicles to generate.

        Returns:
            Path to the generated route file.
        """
        output_dir = self.project_root / "maps" / "processed" / self.ward_id
        output_dir.mkdir(parents=True, exist_ok=True)
        route_path = output_dir / "ward.rou.xml"

        spawn_edges = self._boundaries.get("spawn_candidates", [])
        egress_edges = self._boundaries.get("valid_egress_edges", [])

        if not spawn_edges or not egress_edges:
            logger.warning(
                "Insufficient boundary edges for %s: %d spawn, %d egress",
                self.ward_id, len(spawn_edges), len(egress_edges),
            )

        # Compute vehicle counts from scenario
        if num_vehicles is not None:
            base_count = num_vehicles
        else:
            base_count = int(
                100
                * self._zone_profile["spawn_intensity"]
                * self.scenario["spawn_multiplier"]
            )

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
            origin = self.rng.choice(spawn_edges) if spawn_edges else "unknown"
            destination = self.rng.choice(egress_edges) if egress_edges else "unknown"
            veh_id = f"{self.ward_id}_{vtype}_{i}"
            depart += self.rng.uniform(0.5, 3.0)

            lines.append(
                f'    <trip id="{veh_id}" type="{vtype}" depart="{depart:.1f}"'
                f' from="{origin}" to="{destination}"/>'
            )

        lines.append("</routes>")

        route_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Generated %d trips → %s", base_count, route_path)
        return route_path

    def generate_ward_sumocfg(self) -> Path:
        """Create a ``.sumocfg`` file for the ward.

        Returns:
            Path to the generated configuration file.
        """
        processed_dir = self.project_root / "maps" / "processed" / self.ward_id
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
