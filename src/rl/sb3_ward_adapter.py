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
import os
import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.sumo_env import SumoEnv
from src.reward import WardRewardCalculator
from src.controllers.temporal_features import (
    WARD_FEATURE_DIM,
    WARD_TEMPORAL_WINDOW,
    build_ward_feature_frame,
    pad_history,
)

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

OBSERVATION_DIM = WARD_TEMPORAL_WINDOW * WARD_FEATURE_DIM
ACTION_DIM = 10


@dataclass
class WardAdapterConfig:
    """Runtime settings for a ward-level RL training adapter."""

    ward_id: str | list[str] = "ward_001"
    project_root: str = "."
    gui: bool = False  # NEVER True during training
    scenario_id: str | list[str] = "normal"
    training_mode: bool = True
    decision_interval_steps: int = WARD_TEMPORAL_WINDOW
    max_simulation_steps: int = 360


class StableBaselinesWardEnv(_GymBase):
    """SB3-compatible environment for ward RL training.

    All vehicle motion, routing, waiting times, and network physics
    come from SUMO via ``SumoEnv``. This adapter only exposes the
    reset/step contract that SB3 expects.

    Observation:
        Flattened ``[30, 7]`` temporal stack:
        congestion, queue, avg_speed, inflow, outflow, incident_flag, ambulance_flag
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
        self._ward_history: deque[np.ndarray] = deque(maxlen=WARD_TEMPORAL_WINDOW)
        self._temporal_trace: list[dict[str, Any]] = []

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
            low=0.0, high=np.inf, shape=(OBSERVATION_DIM,), dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Asset loading
    # ------------------------------------------------------------------

    def _load_ward_edges(self, ward_id: str) -> list[str]:
        """Load ALL ward-internal edges from boundaries.json.

        Previously only loaded spawn_candidates (boundary entry edges), which
        meant get_ward_summary returned near-zero readings because vehicles
        drive on internal edges, not entry edges. This caused the GNN dataset
        and RL observations to be nearly zero vectors.

        Now loads internal_edges + valid_ingress_edges + valid_egress_edges so
        observations reflect real traffic state, matching runtime.py behaviour.
        """
        path = (
            self._project_root / "maps" / os.getenv("HMRL_MAP_DIR", "processed")
            / ward_id / "boundaries.json"
        )
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as fh:
            boundaries = json.load(fh)

        combined: list = []
        for key in ("internal_edges", "valid_ingress_edges", "valid_egress_edges"):
            for e in boundaries.get(key, []):
                combined.append(e["edge_id"] if isinstance(e, dict) else str(e))

        if not combined:
            # Absolute fallback: use spawn_candidates
            for e in boundaries.get("spawn_candidates", []):
                combined.append(e["edge_id"] if isinstance(e, dict) else str(e))

        # Deduplicate while preserving order
        seen: set[str] = set()
        result: list[str] = []
        for eid in combined:
            if eid not in seen:
                seen.add(eid)
                result.append(eid)
        return result

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
        self._ward_history.clear()
        self._temporal_trace.clear()

        sumocfg = (
            self._project_root / "maps" / os.getenv("HMRL_MAP_DIR", "processed")
            / self.current_ward_id / "ward.sumocfg"
        )

        if self.sumo_env.is_running:
            self.sumo_env.stop()
        self.sumo_env.start(str(sumocfg), gui=self.config.gui)
        self._prime_history()

        return self._observation(), {"sumo_time": self.sumo_env.get_time()}

    def step(
        self, action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.last_invalid_action = False
        self._apply_action(WardAction(int(action)))

        accumulated_arrived = 0
        accumulated_departed = 0
        for _ in range(self.config.decision_interval_steps):
            self.sumo_env.step()
            self.current_step += 1
            self._step_counter += 1
            self._record_current_state(action)
            accumulated_arrived += self.sumo_env.get_arrived_count()
            accumulated_departed += self.sumo_env.get_departed_count()
            if self.sumo_env.get_min_expected_number() == 0:
                break

        observation = self._observation()
        pressure = self._get_pressure()

        ward_state = self.sumo_env.get_ward_summary(self._ward_edges)
        # Override inflow and outflow with the aggregated values over the decision interval
        ward_state["inflow"] = float(accumulated_departed)
        ward_state["outflow"] = float(accumulated_arrived)

        reward = self.reward_calc.compute(
            ward_state, accumulated_arrived, gnn_pressure=pressure,
            invalid_action=self.last_invalid_action,
        )

        terminated = self.sumo_env.get_min_expected_number() == 0
        truncated = self.current_step >= self.config.max_simulation_steps

        # Collect GNN data every temporal window for backward compatibility.
        if self._step_counter % WARD_TEMPORAL_WINDOW == 0:
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
        return pad_history(
            self._ward_history,
            WARD_TEMPORAL_WINDOW,
            WARD_FEATURE_DIM,
        ).reshape(-1)

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

    def _prime_history(self) -> None:
        summary = self.sumo_env.get_ward_summary(self._ward_edges)
        frame = build_ward_feature_frame(summary)
        for _ in range(WARD_TEMPORAL_WINDOW):
            self._ward_history.append(frame.copy())
            self._temporal_trace.append(self._build_trace_entry(summary, WardAction.NO_OP))

    def _record_current_state(self, action: int) -> None:
        summary = self.sumo_env.get_ward_summary(self._ward_edges)
        frame = build_ward_feature_frame(summary)
        self._ward_history.append(frame)
        self._temporal_trace.append(self._build_trace_entry(summary, WardAction(int(action))))

    def _build_trace_entry(self, summary: dict[str, float], action: WardAction) -> dict[str, Any]:
        return {
            "time": self.current_step,
            "step": self._step_counter,
            "ward_id": self.current_ward_id,
            "scenario_id": self.current_scenario_id,
            "congestion_score": float(summary.get("congestion", 0.0)),
            "queue_length": float(summary.get("queue", 0.0)),
            "avg_speed": float(summary.get("avg_speed", 0.0)),
            "inflow": float(summary.get("inflow", 0.0)),
            "outflow": float(summary.get("outflow", 0.0)),
            "incident_flag": float(summary.get("incident_flag", 0.0)),
            "ambulance_flag": float(summary.get("ambulance_flag", 0.0)),
            "ppo_action": action.name,
            "area_directive": float(self.current_gnn_prediction),
            "city_directive": float(self.current_city_cap),
        }

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
        # Use moderate penalty (300s) to avoid permanent gridlock
        self.sumo_env.adapt_edge_traveltime(edge_id, 300.0)
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
        # Only hold on boundary-like subset to avoid freezing entire ward
        hold_edges = self._ward_edges[:3]
        for edge_id in hold_edges:
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
            "features": build_ward_feature_frame(ward_state),
            "congestion": ward_state.get("congestion", 0.0),
            "temporal_trace": list(self._temporal_trace[-WARD_TEMPORAL_WINDOW:]),
        })

    def get_gnn_snapshot(self) -> dict[str, Any] | None:
        """Return the latest GNN snapshot (called by GNNDataCollector callback)."""
        if self._gnn_buffer:
            return self._gnn_buffer[-1]
        return None

    def get_temporal_trace(self) -> list[dict[str, Any]]:
        """Return the raw temporal trace for dataset builders."""
        return list(self._temporal_trace)


def _require_gymnasium() -> None:
    if _GYM_IMPORT_ERROR is not None:
        raise RuntimeError(
            "gymnasium is required for the SB3 ward adapter."
        ) from _GYM_IMPORT_ERROR
