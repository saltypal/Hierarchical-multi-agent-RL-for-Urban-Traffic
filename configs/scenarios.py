"""Traffic scenario definitions for simulation and training.

Each scenario specifies:
    vehicle_mix     : proportion of each vehicle type (must sum to ~1.0)
    spawn_multiplier: overall traffic intensity relative to baseline
    breakdown_prob  : per-tick probability of a random vehicle breakdown
    blocked_roads   : number of randomly blocked edges
    ambulance_count : forced ambulance injections
    convoy_size     : government convoy vehicle count (0 = none)
    description     : human-readable scenario purpose
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
        "breakdown_prob": 0.0,
        "blocked_roads": 0,
        "ambulance_count": 0,
        "convoy_size": 0,
        "description": "Baseline normal traffic conditions",
    },
    "peak_congestion": {
        "vehicle_mix": {
            "normal_car": 0.65,
            "aggressive": 0.10,
            "slow_driver": 0.08,
            "bmtc_bus": 0.10,
            "truck": 0.05,
            "ambulance": 0.01,
            "govt_convoy": 0.01,
        },
        "spawn_multiplier": 2.0,
        "breakdown_prob": 0.005,
        "blocked_roads": 0,
        "ambulance_count": 0,
        "convoy_size": 0,
        "description": "Heavy morning/evening peak traffic pressure",
    },
    "traffic_surge": {
        "vehicle_mix": {
            "normal_car": 0.75,
            "aggressive": 0.08,
            "slow_driver": 0.05,
            "bmtc_bus": 0.06,
            "truck": 0.04,
            "ambulance": 0.01,
            "govt_convoy": 0.01,
        },
        "spawn_multiplier": 2.5,
        "breakdown_prob": 0.0,
        "blocked_roads": 0,
        "ambulance_count": 0,
        "convoy_size": 0,
        "description": "Sudden unusual traffic influx from external source",
    },
    "breakdown": {
        "vehicle_mix": {
            "normal_car": 0.72,
            "aggressive": 0.05,
            "slow_driver": 0.10,
            "bmtc_bus": 0.07,
            "truck": 0.04,
            "ambulance": 0.01,
            "govt_convoy": 0.01,
        },
        "spawn_multiplier": 1.2,
        "breakdown_prob": 0.02,
        "blocked_roads": 0,
        "ambulance_count": 0,
        "convoy_size": 0,
        "description": "Random vehicle breakdowns causing temporary obstructions",
    },
    "blocked_road": {
        "vehicle_mix": {
            "normal_car": 0.72,
            "aggressive": 0.05,
            "slow_driver": 0.10,
            "bmtc_bus": 0.07,
            "truck": 0.04,
            "ambulance": 0.01,
            "govt_convoy": 0.01,
        },
        "spawn_multiplier": 1.0,
        "breakdown_prob": 0.0,
        "blocked_roads": 2,
        "ambulance_count": 0,
        "convoy_size": 0,
        "description": "Multiple road segments fully blocked, forcing rerouting",
    },
    "ambulance_emergency": {
        "vehicle_mix": {
            "normal_car": 0.68,
            "aggressive": 0.05,
            "slow_driver": 0.10,
            "bmtc_bus": 0.08,
            "truck": 0.05,
            "ambulance": 0.03,
            "govt_convoy": 0.01,
        },
        "spawn_multiplier": 1.3,
        "breakdown_prob": 0.0,
        "blocked_roads": 0,
        "ambulance_count": 3,
        "convoy_size": 0,
        "description": "Multiple ambulances requiring priority corridor clearance",
    },
    "vip_convoy": {
        "vehicle_mix": {
            "normal_car": 0.65,
            "aggressive": 0.05,
            "slow_driver": 0.08,
            "bmtc_bus": 0.08,
            "truck": 0.04,
            "ambulance": 0.01,
            "govt_convoy": 0.09,
        },
        "spawn_multiplier": 1.2,
        "breakdown_prob": 0.0,
        "blocked_roads": 0,
        "ambulance_count": 0,
        "convoy_size": 5,
        "description": "Government convoy with temporary corridor dominance",
    },
    "asymmetric_overload": {
        "vehicle_mix": {
            "normal_car": 0.70,
            "aggressive": 0.08,
            "slow_driver": 0.08,
            "bmtc_bus": 0.07,
            "truck": 0.05,
            "ambulance": 0.01,
            "govt_convoy": 0.01,
        },
        "spawn_multiplier": 1.8,
        "breakdown_prob": 0.005,
        "blocked_roads": 1,
        "ambulance_count": 1,
        "convoy_size": 0,
        "description": "One zone heavily overloaded while others remain lighter",
    },
    "chaos_mode": {
        "vehicle_mix": {
            "normal_car": 0.45,
            "aggressive": 0.30,
            "slow_driver": 0.05,
            "bmtc_bus": 0.08,
            "truck": 0.05,
            "ambulance": 0.03,
            "govt_convoy": 0.04,
        },
        "spawn_multiplier": 2.0,
        "breakdown_prob": 0.10,
        "blocked_roads": 2,
        "ambulance_count": 2,
        "convoy_size": 3,
        "description": "Maximum disturbance: rash drivers, breakdowns, incidents, emergencies",
    },
    "low_baseline": {
        "vehicle_mix": {
            "normal_car": 0.80,
            "aggressive": 0.02,
            "slow_driver": 0.08,
            "bmtc_bus": 0.05,
            "truck": 0.03,
            "ambulance": 0.01,
            "govt_convoy": 0.01,
        },
        "spawn_multiplier": 0.4,
        "breakdown_prob": 0.0,
        "blocked_roads": 0,
        "ambulance_count": 0,
        "convoy_size": 0,
        "description": "Low-traffic sanity check for baseline agent behavior",
    },
}


def get_scenario(scenario_id: str) -> dict[str, Any]:
    """Return scenario config by name, falling back to normal."""
    return SCENARIOS.get(scenario_id, SCENARIOS["normal"])


def list_scenarios() -> list[str]:
    """Return all available scenario IDs."""
    return list(SCENARIOS.keys())
