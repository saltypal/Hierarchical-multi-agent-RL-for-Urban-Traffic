"""Shared temporal feature helpers for ward and area models."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from src.rl.ward_actions import WardAction

WARD_TEMPORAL_WINDOW = 30
WARD_FEATURE_DIM = 7
AREA_TEMPORAL_WINDOW = 60
AREA_FEATURE_DIM = 11


def build_ward_feature_frame(summary: dict[str, float]) -> np.ndarray:
    """Build a single ward timestep frame for PPO stacking."""
    return np.asarray([
        float(summary.get("congestion", 0.0)),
        float(summary.get("queue", 0.0)),
        float(summary.get("avg_speed", 0.0)),
        float(summary.get("inflow", 0.0)),
        float(summary.get("outflow", 0.0)),
        float(summary.get("incident_flag", 0.0)),
        float(summary.get("ambulance_flag", 0.0)),
    ], dtype=np.float32)


def encode_action_name(action_name: str | None) -> int:
    """Map a ward action name to a stable integer id."""
    if not action_name:
        return int(WardAction.NO_OP)
    try:
        return int(WardAction[action_name])
    except KeyError:
        return int(WardAction.NO_OP)


def build_area_feature_frame(
    summary: dict[str, float],
    action_name: str | None,
    recent_actions: deque[int] | None = None,
) -> np.ndarray:
    """Build a temporal graph node frame with intervention history."""
    action_id = encode_action_name(action_name)
    recent_actions = recent_actions or deque(maxlen=AREA_TEMPORAL_WINDOW)
    recent_reroutes = sum(
        1 for item in recent_actions
        if item in {
            int(WardAction.REROUTE_HOTSPOT_GROUP),
            int(WardAction.PRIORITIZE_ALTERNATE_EDGE),
            int(WardAction.DEPRIORITIZE_MOST_CONGESTED_EDGE),
            int(WardAction.INCIDENT_REROUTE),
            int(WardAction.REROUTE_AGGRESSIVE_DRIVERS),
            int(WardAction.REROUTE_HEAVY_VEHICLES),
        }
    )
    emergency_activation = 1.0 if action_id in {
        int(WardAction.CLEAR_AMBULANCE_PATH),
        int(WardAction.INCIDENT_REROUTE),
    } else 0.0
    incident_activation = 1.0 if action_id == int(WardAction.INCIDENT_REROUTE) else float(summary.get("incident_flag", 0.0))

    return np.asarray([
        float(summary.get("congestion", 0.0)),
        float(summary.get("queue", 0.0)),
        float(summary.get("avg_speed", 0.0)),
        float(summary.get("inflow", 0.0)),
        float(summary.get("outflow", 0.0)),
        float(summary.get("incident_flag", 0.0)),
        float(summary.get("ambulance_flag", 0.0)),
        float(action_id),
        float(recent_reroutes),
        float(emergency_activation),
        float(incident_activation),
    ], dtype=np.float32)


def pad_history(history: deque[np.ndarray], window: int, feature_dim: int) -> np.ndarray:
    """Left-pad a deque of frames into a fixed-size tensor."""
    if not history:
        return np.zeros((window, feature_dim), dtype=np.float32)

    frames = list(history)[-window:]
    if len(frames) < window:
        pad = [frames[0]] * (window - len(frames))
        frames = pad + frames
    return np.stack(frames, axis=0).astype(np.float32, copy=False)
