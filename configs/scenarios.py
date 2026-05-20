"""Traffic scenario definitions for simulation and training.

Each scenario specifies:
    vehicle_mix     : proportion of each vehicle type (must sum to ~1.0)
    spawn_multiplier: overall traffic intensity relative to baseline
    breakdown_prob  : per-tick probability of a random vehicle breakdown
    blocked_roads   : number of randomly blocked edges
    ambulance_count : forced ambulance injections
    convoy_size     : government convoy vehicle count (0 = none)
    description     : human-readable scenario purpose

All scenarios include baseline breakdown probability and blocked-road
factors to ensure realistic disturbance patterns across every evaluation.
"""

from __future__ import annotations

from typing import Any


SCENARIOS: dict[str, dict[str, Any]] = {
    "normal": {
        "vehicle_mix": {
            "normal_car": 0.70,
            "aggressive": 0.05,
            "slow_driver": 0.10,
            "bmtc_bus": 0.08,
            "truck": 0.05,
            "ambulance": 0.01,
            "govt_convoy": 0.01,
        },
        "spawn_multiplier": 1.0,
        "boundary_spawn_fraction": 0.7,
        "breakdown_prob": 0.005,
        "blocked_roads": 1,
        "ambulance_count": 1,
        "convoy_size": 0,
        "description": "Baseline traffic with light disturbances (minor breakdowns + 1 blocked road)",
    },
    "peak_congestion": {
        "vehicle_mix": {
            "normal_car": 0.62,
            "aggressive": 0.12,
            "slow_driver": 0.08,
            "bmtc_bus": 0.10,
            "truck": 0.06,
            "ambulance": 0.01,
            "govt_convoy": 0.01,
        },
        "spawn_multiplier": 2.5,
        "boundary_spawn_fraction": 0.9,
        "breakdown_prob": 0.01,
        "blocked_roads": 1,
        "ambulance_count": 1,
        "convoy_size": 0,
        "description": "Heavy peak-hour volume with aggressive drivers and moderate disturbances",
    },
    "breakdown_cascade": {
        "vehicle_mix": {
            "normal_car": 0.72,
            "aggressive": 0.05,
            "slow_driver": 0.10,
            "bmtc_bus": 0.07,
            "truck": 0.04,
            "ambulance": 0.01,
            "govt_convoy": 0.01,
        },
        "spawn_multiplier": 1.5,
        "boundary_spawn_fraction": 0.7,
        "breakdown_prob": 0.05,
        "blocked_roads": 2,
        "ambulance_count": 1,
        "convoy_size": 0,
        "description": "Frequent cascading vehicle failures causing severe lane blockages",
    },
    "ambulance_emergency": {
        "vehicle_mix": {
            "normal_car": 0.65,
            "aggressive": 0.05,
            "slow_driver": 0.10,
            "bmtc_bus": 0.08,
            "truck": 0.05,
            "ambulance": 0.06,
            "govt_convoy": 0.01,
        },
        "spawn_multiplier": 1.3,
        "boundary_spawn_fraction": 0.75,
        "breakdown_prob": 0.01,
        "blocked_roads": 1,
        "ambulance_count": 4,
        "convoy_size": 0,
        "description": "Multiple simultaneous ambulances requiring priority corridor clearance",
    },
    "vip_convoy": {
        "vehicle_mix": {
            "normal_car": 0.62,
            "aggressive": 0.05,
            "slow_driver": 0.08,
            "bmtc_bus": 0.08,
            "truck": 0.04,
            "ambulance": 0.01,
            "govt_convoy": 0.12,
        },
        "spawn_multiplier": 1.4,
        "boundary_spawn_fraction": 0.7,
        "breakdown_prob": 0.01,
        "blocked_roads": 1,
        "ambulance_count": 1,
        "convoy_size": 5,
        "description": "Government convoy with corridor dominance under moderate traffic load",
    },
    "chaos_mode": {
        "vehicle_mix": {
            "normal_car": 0.42,
            "aggressive": 0.28,
            "slow_driver": 0.05,
            "bmtc_bus": 0.08,
            "truck": 0.05,
            "ambulance": 0.06,
            "govt_convoy": 0.06,
        },
        "spawn_multiplier": 2.5,
        "boundary_spawn_fraction": 0.9,
        "breakdown_prob": 0.08,
        "blocked_roads": 3,
        "ambulance_count": 3,
        "convoy_size": 3,
        "description": "Maximum simultaneous stress: rash drivers, breakdowns, emergencies, convoys",
    },
}


def get_scenario(scenario_id: str) -> dict[str, Any]:
    """Return scenario config by name, falling back to normal."""
    return SCENARIOS.get(scenario_id, SCENARIOS["normal"])


def list_scenarios() -> list[str]:
    """Return all available scenario IDs."""
    return list(SCENARIOS.keys())
