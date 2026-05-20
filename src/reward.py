"""Zone-aware multi-component ward reward calculator.

Computes a 14-component reward signal with zone-type modulation.
Designed to be called by the RL adapter using state from ``SumoEnv``.

Penalties are weighted 2-3× heavier than rewards to strongly discourage
congestion, deadlocks, and emergency blocking. Two additional penalty
components target near-gridlock speed collapse and ambulance travel time.

Does NOT import ``traci`` — receives all state via function arguments.
"""

from __future__ import annotations

from typing import Any

from configs.traffic_profiles import get_zone_profile


# ------------------------------------------------------------------
# Default reward weights
# ------------------------------------------------------------------

DEFAULT_WEIGHTS: dict[str, float] = {
    # Positive components (moderate)
    "w_throughput": 2.5,
    "w_trip_completion": 6.0,
    "w_avg_speed": 2.5,
    "w_ambulance_progress": 5.0,
    "w_congestion_reduction": 2.0,
    # Negative components (STRONG — 2-3× base penalties)
    "p_wait_time": 0.05,
    "p_queue_length": 3.0,
    "p_spillback": 4.0,
    "p_deadlock": 20.0,
    "p_incident_duration": 3.0,
    "p_ambulance_blocking": 15.0,
    "p_unfairness": 2.5,
    # New penalty components
    "p_speed_collapse": 5.0,
    "p_ambulance_travel_time": 4.0,
    # GNN pressure penalty
    "lambda_pressure": 3.0,
}

# ------------------------------------------------------------------
# Zone-specific modulation overrides
# ------------------------------------------------------------------

ZONE_MODULATION: dict[str, dict[str, float]] = {
    "commercial": {
        "w_throughput": 3.5,
        "w_avg_speed": 3.0,
        "w_trip_completion": 4.0,
        "p_queue_length": 4.0,
    },
    "residential": {
        "p_unfairness": 4.0,
        "p_queue_length": 2.0,
        "w_throughput": 2.0,
        "p_speed_collapse": 3.0,
    },
    "mixed": {},
    "arterial": {
        "w_throughput": 3.5,
        "w_avg_speed": 3.0,
        "p_spillback": 5.0,
    },
    "hospital_sensitive": {
        "w_ambulance_progress": 12.0,
        "p_ambulance_blocking": 25.0,
        "p_ambulance_travel_time": 8.0,
    },
    "bottleneck": {
        "w_throughput": 4.0,
        "w_congestion_reduction": 4.0,
        "p_deadlock": 30.0,
    },
    "it_corridor": {
        "w_throughput": 4.0,
        "w_trip_completion": 5.0,
        "w_avg_speed": 3.5,
        "p_queue_length": 4.0,
    },
}


class WardRewardCalculator:
    """Computes zone-modulated multi-component rewards for ward RL agents.

    The reward function balances throughput, safety, fairness, and
    emergency responsiveness. Zone type automatically adjusts weights.
    Penalties are intentionally 2-3× stronger than rewards to firmly
    discourage negative traffic outcomes.

    Reward formula::

        R = + w1 * throughput
            + w2 * trip_completion
            + w3 * avg_speed
            + w4 * ambulance_progress
            + w5 * congestion_reduction
            - p1 * wait_time
            - p2 * queue_length
            - p3 * spillback
            - p4 * deadlock
            - p5 * incident_duration
            - p6 * ambulance_blocking
            - p7 * unfairness
            - p8 * speed_collapse
            - p9 * ambulance_travel_time
            - λ  * gnn_pressure * outflow
    """

    def __init__(
        self,
        zone_type: str,
        custom_weights: dict[str, float] | None = None,
    ) -> None:
        self.zone_type = zone_type
        self.weights = DEFAULT_WEIGHTS.copy()

        # Apply zone modulation
        zone_mod = ZONE_MODULATION.get(zone_type, {})
        self.weights.update(zone_mod)

        # Apply any user overrides
        if custom_weights:
            self.weights.update(custom_weights)

        self._prev_queue: float = 0.0
        self._prev_congestion: float = 0.0

    def compute(
        self,
        ward_state: dict[str, float],
        arrived: int,
        gnn_pressure: float = 0.0,
        invalid_action: bool = False,
    ) -> float:
        """Compute the reward from ward traffic metrics.

        Args:
            ward_state: Ward summary dict from ``SumoEnv.get_ward_summary()``.
            arrived: Number of vehicles that arrived at destination this step.
            gnn_pressure: GNN-predicted incoming pressure (0 = none, 1 = max).
            invalid_action: Whether the last action was invalid.

        Returns:
            Scalar reward value.
        """
        w = self.weights

        vehicle_count = ward_state.get("throughput", 0.0)
        queue = ward_state.get("queue", 0.0)
        avg_speed = ward_state.get("avg_speed", 0.0)
        congestion = ward_state.get("congestion", 0.0)
        ambulance = ward_state.get("ambulance_flag", 0.0)
        incident = ward_state.get("incident_flag", 0.0)
        waiting_time = ward_state.get("waiting_time", 0.0)

        # --- Positive components ---
        throughput_reward = w["w_throughput"] * avg_speed
        trip_reward = w["w_trip_completion"] * float(arrived)
        speed_reward = w["w_avg_speed"] * avg_speed

        # Ambulance progress: positive if ambulance present and speed is good
        ambulance_reward = w["w_ambulance_progress"] * ambulance * avg_speed

        # Congestion reduction from previous step
        congestion_delta = self._prev_congestion - congestion
        congestion_reward = w["w_congestion_reduction"] * max(congestion_delta, 0.0)

        # --- Negative components ---
        wait_penalty = w["p_wait_time"] * congestion * vehicle_count
        queue_penalty = w["p_queue_length"] * queue

        # Spillback: queue growing faster than expected
        queue_delta = queue - self._prev_queue
        spillback_penalty = w["p_spillback"] * max(queue_delta, 0.0)

        # Deadlock: very high queue ratio with near-zero speed
        deadlock_penalty = 0.0
        if congestion > 0.8 and avg_speed < 0.5:
            deadlock_penalty = w["p_deadlock"]

        incident_penalty = w["p_incident_duration"] * incident
        ambulance_block_penalty = w["p_ambulance_blocking"] * ambulance * congestion

        # Unfairness: penalize extreme congestion imbalance
        unfairness_penalty = w["p_unfairness"] * max(congestion - 0.5, 0.0)

        # Speed collapse: near-gridlock conditions (avg_speed < 1.0 m/s)
        speed_collapse_penalty = 0.0
        if avg_speed < 1.0 and vehicle_count > 0:
            speed_collapse_penalty = w["p_speed_collapse"] * (1.0 - avg_speed)

        # Ambulance travel time penalty: ambulance stuck in traffic
        ambulance_tt_penalty = w["p_ambulance_travel_time"] * ambulance * waiting_time * 0.01

        # GNN pressure penalty
        pressure_penalty = w["lambda_pressure"] * gnn_pressure * vehicle_count * 0.01

        # Invalid action penalty
        invalid_penalty = 15.0 if invalid_action else 0.0

        # --- Combine ---
        reward = (
            throughput_reward
            + trip_reward
            + speed_reward
            + ambulance_reward
            + congestion_reward
            - wait_penalty
            - queue_penalty
            - spillback_penalty
            - deadlock_penalty
            - incident_penalty
            - ambulance_block_penalty
            - unfairness_penalty
            - speed_collapse_penalty
            - ambulance_tt_penalty
            - pressure_penalty
            - invalid_penalty
        )

        # Update state for next step
        self._prev_queue = queue
        self._prev_congestion = congestion

        return float(reward)

    def reset(self) -> None:
        """Reset stateful tracking between episodes."""
        self._prev_queue = 0.0
        self._prev_congestion = 0.0
