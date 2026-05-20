"""SUMO/TraCI abstraction layer.

This is the **sole module** in the entire codebase permitted to import or
call ``traci``.  Every other module that needs simulator interaction must
route through ``SumoEnv``.
"""

from __future__ import annotations

import socket
import logging
from pathlib import Path
from typing import Any

import traci

logger = logging.getLogger(__name__)


class SumoEnv:
    """Simulation operating-system abstraction over SUMO/TraCI.

    Provides a clean API for lifecycle management, vehicle control,
    state queries, and ward-level metric aggregation.
    """

    def __init__(self) -> None:
        self._started = False
        self._config_path: str | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, config_path: str, gui: bool = False, delay_ms: float = 150.0) -> None:
        """Start a new SUMO simulation."""
        # Clean up any leftover/dangling TraCI connections from previous runs/cells
        try:
            traci.close()
        except Exception:
            pass

        binary = "sumo-gui" if gui else "sumo"
        self._config_path = config_path
        cmd = [binary, "-c", config_path, "--no-warnings", "--no-step-log"]
        if gui:
            gui_cfg = Path(__file__).parent.parent / "configs" / "gui-settings.xml"
            if gui_cfg.exists():
                cmd.extend(["--gui-settings-file", str(gui_cfg)])
                
        # Find a free local TCP port dynamically to prevent port collisions
        free_port = 8813
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', 0))
                free_port = s.getsockname()[1]
        except Exception:
            pass

        try:
            traci.start(cmd, port=free_port)
        except Exception as e:
            if "already active" in str(e) or "default" in str(e):
                logger.warning("TraCI connection default already active. Forcing close and retrying...")
                try:
                    traci.close()
                except Exception:
                    pass
                traci.start(cmd, port=free_port)
            else:
                raise e

        self._started = True
        logger.info("SUMO started on port %d: %s (gui=%s)", free_port, config_path, gui)

        if gui:
            try:
                # Set default simulation delay to see cars moving cleanly
                traci.gui.setDelay("View #0", delay_ms)
                logger.info("SUMO GUI delay set to %.1f ms", delay_ms)
            except Exception as e:
                logger.warning("Could not set GUI delay: %s", e)

    def stop(self) -> None:
        """Shut down the running SUMO instance."""
        if self._started:
            traci.close()
            self._started = False
            logger.info("SUMO stopped")

    def step(self) -> None:
        """Advance the simulation by one timestep."""
        traci.simulationStep()

    def reset(self) -> None:
        """Reload the current configuration without restarting the process."""
        if self._config_path is None:
            raise RuntimeError("Cannot reset: no config loaded")
        traci.load(["-c", self._config_path])
        logger.info("SUMO reset: %s", self._config_path)

    def load_config(self, config_path: str) -> None:
        """Load a different SUMO configuration."""
        self._config_path = config_path
        traci.load(["-c", config_path])
        logger.info("SUMO config loaded: %s", config_path)

    @property
    def is_running(self) -> bool:
        return self._started

    # ------------------------------------------------------------------
    # Vehicle control
    # ------------------------------------------------------------------

    def add_vehicle(
        self,
        veh_id: str,
        route_id: str,
        vtype: str = "normal_car",
        depart_time: float | None = None,
    ) -> None:
        """Insert a vehicle into the simulation."""
        try:
            traci.vehicle.add(veh_id, route_id, typeID=vtype)
            if depart_time is not None:
                traci.vehicle.setDepart(veh_id, depart_time)
        except traci.TraCIException as exc:
            logger.warning("Failed to add vehicle %s: %s", veh_id, exc)

    def remove_vehicle(self, veh_id: str) -> None:
        """Remove a vehicle from the simulation."""
        try:
            traci.vehicle.remove(veh_id)
        except traci.TraCIException as exc:
            logger.warning("Failed to remove vehicle %s: %s", veh_id, exc)

    def reroute_vehicle(self, veh_id: str) -> None:
        """Reroute a vehicle using current travel times."""
        try:
            traci.vehicle.rerouteTraveltime(veh_id)
        except traci.TraCIException:
            pass

    def set_vehicle_speed(self, veh_id: str, speed: float) -> None:
        """Set vehicle speed. Use -1 to hand control back to SUMO."""
        try:
            traci.vehicle.setSpeed(veh_id, speed)
        except traci.TraCIException:
            pass

    # ------------------------------------------------------------------
    # Vehicle state queries
    # ------------------------------------------------------------------

    def get_vehicle_ids(self) -> list[str]:
        return list(traci.vehicle.getIDList())

    def get_vehicle_speed(self, veh_id: str) -> float:
        try:
            return traci.vehicle.getSpeed(veh_id)
        except traci.TraCIException:
            return 0.0

    def get_vehicle_waiting_time(self, veh_id: str) -> float:
        try:
            return traci.vehicle.getWaitingTime(veh_id)
        except traci.TraCIException:
            return 0.0

    def get_vehicle_type(self, veh_id: str) -> str:
        try:
            return traci.vehicle.getTypeID(veh_id)
        except traci.TraCIException:
            return ""

    def get_vehicle_route(self, veh_id: str) -> list[str]:
        try:
            return list(traci.vehicle.getRoute(veh_id))
        except traci.TraCIException:
            return []

    def get_vehicle_route_index(self, veh_id: str) -> int:
        try:
            return traci.vehicle.getRouteIndex(veh_id)
        except traci.TraCIException:
            return 0

    # ------------------------------------------------------------------
    # Edge state queries
    # ------------------------------------------------------------------

    def get_edge_ids(self) -> list[str]:
        return [
            eid for eid in traci.edge.getIDList()
            if not eid.startswith(":")
        ]

    def get_edge_vehicle_ids(self, edge_id: str) -> list[str]:
        try:
            return list(traci.edge.getLastStepVehicleIDs(edge_id))
        except traci.TraCIException:
            return []

    def get_edge_halting_count(self, edge_id: str) -> int:
        try:
            return traci.edge.getLastStepHaltingNumber(edge_id)
        except traci.TraCIException:
            return 0

    def adapt_edge_traveltime(self, edge_id: str, time: float) -> None:
        try:
            traci.edge.adaptTraveltime(edge_id, time)
        except traci.TraCIException:
            pass

    # ------------------------------------------------------------------
    # Simulation state queries
    # ------------------------------------------------------------------

    def get_time(self) -> float:
        return traci.simulation.getTime()

    def get_arrived_count(self) -> int:
        return traci.simulation.getArrivedNumber()

    def get_departed_count(self) -> int:
        return traci.simulation.getDepartedNumber()

    def get_min_expected_number(self) -> int:
        return traci.simulation.getMinExpectedNumber()

    # ------------------------------------------------------------------
    # Junction queries
    # ------------------------------------------------------------------

    def get_junction_ids(self) -> list[str]:
        return [
            jid for jid in traci.junction.getIDList()
            if not jid.startswith(":")
        ]

    def get_junction_incoming_edges(self, junction_id: str) -> list[str]:
        try:
            return list(traci.junction.getIncomingEdges(junction_id))
        except traci.TraCIException:
            return []

    # ------------------------------------------------------------------
    # Ward-level aggregated state
    # ------------------------------------------------------------------

    def get_ward_summary(self, ward_edge_ids: list[str]) -> dict[str, float]:
        """Compute aggregated traffic metrics for a set of edges (one ward)."""
        vehicle_ids: list[str] = []
        for edge_id in ward_edge_ids:
            vehicle_ids.extend(self.get_edge_vehicle_ids(edge_id))

        vehicle_count = len(vehicle_ids)
        if vehicle_count == 0:
            return {
                "congestion": 0.0,
                "queue": 0.0,
                "avg_speed": 0.0,
                "throughput": 0.0,
                "inflow": 0.0,
                "outflow": 0.0,
                "waiting_time": 0.0,
                "max_waiting_time": 0.0,
                "vehicle_count": 0.0,
                "incident_flag": 0.0,
                "ambulance_flag": 0.0,
            }

        speeds = [self.get_vehicle_speed(v) for v in vehicle_ids]
        waits = [self.get_vehicle_waiting_time(v) for v in vehicle_ids]
        queue_count = sum(1 for s in speeds if s < 0.1)
        avg_speed = sum(speeds) / vehicle_count
        ambulance_count = sum(
            1 for v in vehicle_ids
            if "ambulance" in self.get_vehicle_type(v).lower()
        )

        return {
            "congestion": queue_count / max(vehicle_count, 1),
            "queue": float(queue_count),
            "avg_speed": avg_speed,
            "throughput": float(vehicle_count),
            "inflow": float(self.get_departed_count()),
            "outflow": float(self.get_arrived_count()),
            "waiting_time": float(sum(waits) / vehicle_count),
            "max_waiting_time": float(max(waits)),
            "vehicle_count": float(vehicle_count),
            "incident_flag": 1.0 if max(waits) > 120.0 else 0.0,
            "ambulance_flag": float(min(ambulance_count, 1)),
        }
