"""Zone-aware multi-component ward reward calculator.

Computes a PPO-stable, normalized 5-component reward signal.
Components:
    1. Efficiency (Throughput & Speed)
    2. Congestion Penalty
    3. Improvement Shaping
    4. Context Modifier (Ambulances & Incidents)
    5. Hierarchical Penalty

Final reward is clipped to [-10, 10] to ensure stable policy gradients.
"""

from __future__ import annotations

import numpy as np

# ------------------------------------------------------------------
# Default reward weights
# ------------------------------------------------------------------

DEFAULT_WEIGHTS: dict[str, float] = {
    "w_throughput": 2.0,
    "w_speed": 1.5,
    "p_queue": 2.5,
    "p_congestion": 2.0,
    "w_delta_queue": 1.5,
    "w_delta_congestion": 1.5,
    "w_ambulance_speed": 3.0,
    "p_ambulance_blocking": 3.0,
    "p_incident_queue": 2.0,
    "w_incident_recovery": 2.0,
    "p_hierarchy": 1.5,
    "p_invalid": 1.0,
}

# ------------------------------------------------------------------
# Zone-specific modulation overrides
# ------------------------------------------------------------------
ZONE_MODULATION: dict[str, dict[str, float]] = {
    "commercial": {
        "w_throughput": 2.6,  # Favor throughput
    },
    "hospital_sensitive": {
        "w_ambulance_speed": 6.0,  # Double ambulance priority
        "p_ambulance_blocking": 6.0,
    },
    "bottleneck": {
        "p_congestion": 3.0,  # Deadlock sensitivity
    },
    "arterial": {
        "w_throughput": 2.5,
        "w_speed": 2.0,
    },
    "residential": {
        "p_congestion": 2.5,
    },
}


class WardRewardCalculator:
    """Computes bounded, smooth rewards for ward RL agents."""

    def __init__(
        self,
        zone_type: str,
        custom_weights: dict[str, float] | None = None,
    ) -> None:
        self.zone_type = zone_type
        self.weights = DEFAULT_WEIGHTS.copy()

        # Apply zone modulation
        if zone_type in ZONE_MODULATION:
            self.weights.update(ZONE_MODULATION[zone_type])

        # Apply any user overrides
        if custom_weights:
            self.weights.update(custom_weights)

        self._prev_queue_norm: float = 0.0
        self._prev_congestion: float = 0.0

    def compute(
        self,
        ward_state: dict[str, float],
        arrived: int,
        gnn_pressure: float = 0.0,
        invalid_action: bool = False,
    ) -> float:
        """Compute the normalized, clipped reward.
        
        Args:
            ward_state: Ward summary dict from ``SumoEnv.get_ward_summary()``.
            arrived: Number of vehicles that arrived at destination this step.
            gnn_pressure: GNN-predicted incoming pressure (0 = none, 1 = max).
            invalid_action: Whether the last action was invalid.

        Returns:
            Scalar reward value clipped to [-10, 10].
        """
        w = self.weights

        vehicle_count = ward_state.get("throughput", 0.0)  # total vehicles in ward
        queue = ward_state.get("queue", 0.0)
        avg_speed = ward_state.get("avg_speed", 0.0)
        congestion = ward_state.get("congestion", 0.0)
        ambulance_flag = ward_state.get("ambulance_flag", 0.0)
        incident_flag = ward_state.get("incident_flag", 0.0)
        inflow = ward_state.get("inflow", 0.0)

        # Normalization bases
        max_speed = 20.0  # ~72 km/h, standard urban cap
        safe_vehicle_count = max(vehicle_count, 1.0)

        # 1. Efficiency
        throughput_norm = float(arrived) / safe_vehicle_count
        speed_norm = min(avg_speed / max_speed, 1.0)
        r_efficiency = (w["w_throughput"] * throughput_norm) + (w["w_speed"] * speed_norm)

        # 2. Congestion Penalty
        queue_norm = queue / safe_vehicle_count
        p_congestion = (w["p_queue"] * queue_norm) + (w["p_congestion"] * congestion)

        # 3. Improvement Shaping
        delta_q = self._prev_queue_norm - queue_norm
        delta_c = self._prev_congestion - congestion
        r_improvement = (w["w_delta_queue"] * delta_q) + (w["w_delta_congestion"] * delta_c)

        # 4. Context Modifier
        r_emergency = 0.0
        r_incident = 0.0
        
        if ambulance_flag > 0:
            ambulance_speed_norm = min(avg_speed / max_speed, 1.0)
            r_emergency += w["w_ambulance_speed"] * ambulance_speed_norm
            r_emergency -= w["p_ambulance_blocking"] * congestion

        if incident_flag > 0:
            r_incident -= w["p_incident_queue"] * queue_norm
            r_incident += w["w_incident_recovery"] * delta_c

        # 5. Hierarchical Coordination Penalty
        inflow_norm = inflow / safe_vehicle_count
        p_hierarchy = w["p_hierarchy"] * gnn_pressure * inflow_norm

        # Invalid action penalty
        p_invalid = w["p_invalid"] if invalid_action else 0.0

        # Combine
        total_reward = (
            r_efficiency
            + r_improvement
            + r_emergency
            + r_incident
            - p_congestion
            - p_hierarchy
            - p_invalid
        )

        # Clip for PPO stability (Crucial for preventing gradient collapse)
        clipped_reward = float(np.clip(total_reward, -10.0, 10.0))

        # State tracking for next step
        self._prev_queue_norm = queue_norm
        self._prev_congestion = congestion

        return clipped_reward

    def reset(self) -> None:
        """Reset state tracking between episodes."""
        self._prev_queue_norm = 0.0
        self._prev_congestion = 0.0
