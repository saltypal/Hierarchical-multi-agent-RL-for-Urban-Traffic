"""Stable-Baselines3 adapter for ward-level RL training.

Routes ALL simulator interaction through ``SumoEnv`` — no direct
``traci`` imports. Observation space is 12 dimensions: 10 local
traffic features + 1 GNN predicted pressure + 1 city capacity cap.

During training mode, synthetic pressure values are injected to
teach the agent to respond to upper-layer signals before the GNN
exists. GNN data snapshots are collected every 30 steps for free.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.sumo_env import SumoEnv
from src.reward import WardRewardCalculator

from .ward_actions import WardAction

try:
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError as exc:  # pragma: no cover
    gym = None
    spaces = None
    _GYM_IMPORT_ERROR = exc
else:
    _GYM_IMPORT_ERROR = None

_GymBase = gym.Env if gym is not None else object


@dataclass
class WardAdapterConfig:
    """Runtime settings for a ward-level RL training adapter."""

    ward_id: str | list[str] = "ward_001"
    project_root: str = "."
    gui: bool = False  # NEVER True during training
    scenario_id: str | list[str] = "normal"
    training_mode: bool = True
    decision_interval_steps: int = 5
    max_simulation_steps: int = 360


class StableBaselinesWardEnv(_GymBase):
    """SB3-compatible environment for ward RL training.

    All vehicle motion, routing, waiting times, and network physics
    come from SUMO via ``SumoEnv``. This adapter only exposes the
    reset/step contract that SB3 expects.

    Observation (12 dims):
        [0]  vehicle_count
        [1]  queue_count
        [2]  queue_ratio
        [3]  total_wait_proxy
        [4]  max_wait_proxy
        [5]  avg_speed
        [6]  avg_route_length
        [7]  ambulance_count
        [8]  aggressive_count
        [9]  heavy_vehicle_count
        [10] predicted_pressure  (from GNN or synthetic during training)
        [11] city_capacity_cap   (from city solver or synthetic during training)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: dict[str, Any] | WardAdapterConfig) -> None:
        _require_gymnasium()

        if isinstance(config, dict):
            self.config = WardAdapterConfig(**{
                k: v for k, v in config.items()
                if k in WardAdapterConfig.__dataclass_fields__
            })
        else:
            self.config = config

        self.sumo_env = SumoEnv()
        self.current_step = 0
        self.held_vehicle_ids: set[str] = set()
        self.last_invalid_action = False
        self._rng = random.Random(42)

        if isinstance(self.config.ward_id, str):
            self.ward_ids = [self.config.ward_id]
        else:
            self.ward_ids = self.config.ward_id

        if isinstance(self.config.scenario_id, str):
            self.scenario_ids = [self.config.scenario_id]
        else:
            self.scenario_ids = self.config.scenario_id

        self.current_ward_id = self.ward_ids[0]
        self.current_scenario_id = self.scenario_ids[0]

        # Load ward assets for the initial ward
        self._project_root = Path(self.config.project_root)
        self._ward_edges = self._load_ward_edges(self.current_ward_id)
        self._zone_type = self._load_zone_type(self.current_ward_id)

        # Reward calculator
        self.reward_calc = WardRewardCalculator(self._zone_type)

        # GNN data collection buffer
        self._gnn_buffer: list[dict[str, Any]] = []
        self._step_counter = 0

        # Current upper-layer signals (set externally during inference)
        self.current_gnn_prediction: float = 0.0
        self.current_city_cap: float = 1.0

        # Action and observation spaces
        self.action_space = spaces.Discrete(len(WardAction))
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(12,), dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Asset loading
    # ------------------------------------------------------------------

    def _load_ward_edges(self, ward_id: str) -> list[str]:
        """Load spawn candidate edges from boundaries.json."""
        path = (
            self._project_root / "maps" / "processed"
            / ward_id / "boundaries.json"
        )
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as fh:
            boundaries = json.load(fh)
        candidates = boundaries.get("spawn_candidates", [])
        # Extract just the edge_id string from the dictionary
        return [
            c["edge_id"] if isinstance(c, dict) else str(c)
            for c in candidates
        ]

    def _load_zone_type(self, ward_id: str) -> str:
        """Load zone type from ward registry."""
        path = self._project_root / "configs" / "hierarchy" / "ward_registry.json"
        if not path.exists():
            return "mixed"
        with path.open("r", encoding="utf-8") as fh:
            registry = json.load(fh)
        return registry.get("wards", {}).get(ward_id, {}).get("zone_type", "mixed")

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0
        self.held_vehicle_ids.clear()
        self.last_invalid_action = False

        # Randomize ward and scenario for Multi-Ward Training
        self.current_ward_id = self._rng.choice(self.ward_ids)
        self.current_scenario_id = self._rng.choice(self.scenario_ids)

        # Update environment dynamics to the newly selected ward
        self._ward_edges = self._load_ward_edges(self.current_ward_id)
        self._zone_type = self._load_zone_type(self.current_ward_id)
        self.reward_calc = WardRewardCalculator(self._zone_type)
        self.reward_calc.reset()

        sumocfg = (
            self._project_root / "maps" / "processed"
            / self.current_ward_id / "ward.sumocfg"
        )

        if self.sumo_env.is_running:
            self.sumo_env.load_config(str(sumocfg))
        else:
            self.sumo_env.start(str(sumocfg), gui=self.config.gui)

        return self._observation(), {"sumo_time": self.sumo_env.get_time()}

    def step(
        self, action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.last_invalid_action = False
        self._apply_action(WardAction(int(action)))

        for _ in range(self.config.decision_interval_steps):
            self.sumo_env.step()
            self.current_step += 1
            self._step_counter += 1
            if self.sumo_env.get_min_expected_number() == 0:
                break

        observation = self._observation()
        pressure = self._get_pressure()
        arrived = self.sumo_env.get_arrived_count()

        ward_state = self.sumo_env.get_ward_summary(self._ward_edges)
        reward = self.reward_calc.compute(
            ward_state, arrived, gnn_pressure=pressure,
            invalid_action=self.last_invalid_action,
        )

        terminated = self.sumo_env.get_min_expected_number() == 0
        truncated = self.current_step >= self.config.max_simulation_steps

        # Collect GNN data every 30 steps (free)
        if self._step_counter % 30 == 0:
            self._collect_gnn_snapshot(ward_state)

        info = {
            "sumo_time": self.sumo_env.get_time(),
            "action_name": WardAction(int(action)).name,
            "invalid_action": self.last_invalid_action,
        }

        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        self.sumo_env.stop()

    # ------------------------------------------------------------------
    # Observation building
    # ------------------------------------------------------------------

    def _observation(self) -> np.ndarray:
        vehicle_ids = self.sumo_env.get_vehicle_ids()
        vehicle_count = len(vehicle_ids)

        if vehicle_count == 0:
            return np.zeros(12, dtype=np.float32)

        speeds = [self.sumo_env.get_vehicle_speed(v) for v in vehicle_ids]
        waits = [self.sumo_env.get_vehicle_waiting_time(v) for v in vehicle_ids]
        route_lengths = [len(self.sumo_env.get_vehicle_route(v)) for v in vehicle_ids]
        queue_count = sum(1 for s in speeds if s < 0.1)

        ambulance_count = sum(
            1 for v in vehicle_ids if self._vehicle_matches(v, ["ambulance"])
        )
        aggressive_count = sum(
            1 for v in vehicle_ids if self._vehicle_matches(v, ["aggressive", "rash"])
        )
        heavy_count = sum(
            1 for v in vehicle_ids if self._vehicle_matches(v, ["truck", "bus", "bmtc", "heavy"])
        )

        pressure = self._get_pressure()
        cap = self._get_city_cap()

        return np.asarray([
            float(vehicle_count),                          # 0
            float(queue_count),                            # 1
            float(queue_count / vehicle_count),            # 2
            float(sum(waits)),                             # 3
            float(max(waits)),                             # 4
            float(sum(speeds) / vehicle_count),            # 5
            float(sum(route_lengths) / vehicle_count),     # 6
            float(ambulance_count),                        # 7
            float(aggressive_count),                       # 8
            float(heavy_count),                            # 9
            pressure,                                      # 10: GNN / synthetic
            cap,                                           # 11: city cap / synthetic
        ], dtype=np.float32)

    def _get_pressure(self) -> float:
        """Get GNN pressure: real during inference, synthetic during training."""
        if self.config.training_mode:
            return self._rng.choices(
                [0.0, self._rng.uniform(0.3, 0.6), self._rng.uniform(0.7, 0.95)],
                weights=[0.5, 0.3, 0.2],
            )[0]
        return self.current_gnn_prediction

    def _get_city_cap(self) -> float:
        """Get city cap: real during inference, synthetic during training."""
        if self.config.training_mode:
            return self._rng.uniform(0.5, 1.0)
        return self.current_city_cap

    # ------------------------------------------------------------------
    # Action application (all through SumoEnv)
    # ------------------------------------------------------------------

    def _apply_action(self, action: WardAction) -> None:
        if action == WardAction.NO_OP:
            return
        elif action == WardAction.REROUTE_HOTSPOT_GROUP:
            self._reroute_vehicles_on_edge(self._most_congested_edge())
        elif action == WardAction.DEPRIORITIZE_MOST_CONGESTED_EDGE:
            self._deprioritize_congested_edge()
        elif action == WardAction.PRIORITIZE_ALTERNATE_EDGE:
            self._reroute_vehicles_on_edge(self._most_congested_edge())
        elif action == WardAction.CLEAR_AMBULANCE_PATH:
            self._clear_ambulance_path()
        elif action == WardAction.INCIDENT_REROUTE:
            self._incident_reroute()
        elif action == WardAction.HOLD_COMMERCIAL_INFLOW:
            self._hold_inflow()
        elif action == WardAction.RELEASE_HELD_FLOW:
            self._release_held_flow()
        elif action == WardAction.REROUTE_AGGRESSIVE_DRIVERS:
            self._reroute_matching_vehicles(["aggressive", "rash"])
        elif action == WardAction.REROUTE_HEAVY_VEHICLES:
            self._reroute_matching_vehicles(["truck", "bus", "bmtc", "heavy"])

    def _most_congested_edge(self) -> str | None:
        edge_ids = self.sumo_env.get_edge_ids()
        if not edge_ids:
            self.last_invalid_action = True
            return None
        return max(edge_ids, key=self.sumo_env.get_edge_halting_count)

    def _reroute_vehicles_on_edge(self, edge_id: str | None) -> None:
        if edge_id is None:
            self.last_invalid_action = True
            return
        vehicle_ids = self.sumo_env.get_edge_vehicle_ids(edge_id)
        if not vehicle_ids:
            self.last_invalid_action = True
            return
        for v in vehicle_ids:
            self.sumo_env.reroute_vehicle(v)

    def _deprioritize_congested_edge(self) -> None:
        edge_id = self._most_congested_edge()
        if edge_id is None:
            self.last_invalid_action = True
            return
        self.sumo_env.adapt_edge_traveltime(edge_id, 9999.0)
        self._reroute_vehicles_on_edge(edge_id)

    def _clear_ambulance_path(self) -> None:
        vehicle_ids = self.sumo_env.get_vehicle_ids()
        ambulance_ids = [
            v for v in vehicle_ids if self._vehicle_matches(v, ["ambulance"])
        ]
        if not ambulance_ids:
            self.last_invalid_action = True
            return
        protected_edges: set[str] = set()
        for amb_id in ambulance_ids:
            route = self.sumo_env.get_vehicle_route(amb_id)
            idx = self.sumo_env.get_vehicle_route_index(amb_id)
            protected_edges.update(route[idx: idx + 3])
        for edge_id in protected_edges:
            for v in self.sumo_env.get_edge_vehicle_ids(edge_id):
                if v not in ambulance_ids:
                    self.sumo_env.reroute_vehicle(v)

    def _incident_reroute(self) -> None:
        # Reroute vehicles on blocked edges if any
        for edge_id in self._ward_edges[:2]:
            self._reroute_vehicles_on_edge(edge_id)

    def _hold_inflow(self) -> None:
        for edge_id in self._ward_edges:
            for v in self.sumo_env.get_edge_vehicle_ids(edge_id):
                self.sumo_env.set_vehicle_speed(v, 0.0)
                self.held_vehicle_ids.add(v)

    def _release_held_flow(self) -> None:
        current_vehicles = set(self.sumo_env.get_vehicle_ids())
        for v in list(self.held_vehicle_ids):
            if v in current_vehicles:
                self.sumo_env.set_vehicle_speed(v, -1.0)
            self.held_vehicle_ids.discard(v)

    def _reroute_matching_vehicles(self, tokens: list[str]) -> None:
        matched = [
            v for v in self.sumo_env.get_vehicle_ids()
            if self._vehicle_matches(v, tokens)
        ]
        if not matched:
            self.last_invalid_action = True
            return
        for v in matched:
            self.sumo_env.reroute_vehicle(v)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _vehicle_matches(self, vehicle_id: str, tokens: list[str]) -> bool:
        vtype = self.sumo_env.get_vehicle_type(vehicle_id)
        haystack = f"{vehicle_id} {vtype}".lower()
        return any(t in haystack for t in tokens)

    # ------------------------------------------------------------------
    # GNN data collection
    # ------------------------------------------------------------------

    def _collect_gnn_snapshot(self, ward_state: dict[str, float]) -> None:
        """Store a ward state snapshot for later GNN training."""
        self._gnn_buffer.append({
            "step": self._step_counter,
            "features": np.array([
                ward_state.get("congestion", 0.0),
                0.0,  # delta congestion (computed post-hoc)
                ward_state.get("queue", 0.0),
                0.0,  # delta queue (computed post-hoc)
                ward_state.get("avg_speed", 0.0),
                ward_state.get("throughput", 0.0),
                ward_state.get("incident_flag", 0.0),
                self.current_city_cap,
            ], dtype=np.float32),
            "congestion": ward_state.get("congestion", 0.0),
        })

    def get_gnn_snapshot(self) -> dict[str, Any] | None:
        """Return the latest GNN snapshot (called by GNNDataCollector callback)."""
        if self._gnn_buffer:
            return self._gnn_buffer[-1]
        return None


def _require_gymnasium() -> None:
    if _GYM_IMPORT_ERROR is not None:
        raise RuntimeError(
            "gymnasium is required for the SB3 ward adapter."
        ) from _GYM_IMPORT_ERROR
