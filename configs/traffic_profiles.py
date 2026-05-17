"""Zone-semantic traffic priors for demand generation and reward modulation.

Each zone type defines:
    spawn_intensity: base vehicle spawn rate multiplier relative to baseline
    peak_multiplier: multiplier applied during peak-hour scenarios
    reward_bias: dominant reward component for zone-aware reward modulation
"""

from __future__ import annotations

from typing import Any


ZONE_PROFILES: dict[str, dict[str, Any]] = {
    "commercial": {
        "spawn_intensity": 1.8,
        "peak_multiplier": 2.5,
        "reward_bias": "throughput",
        "description": "High inbound traffic, economically critical flow efficiency",
    },
    "residential": {
        "spawn_intensity": 1.0,
        "peak_multiplier": 1.5,
        "reward_bias": "fairness",
        "description": "Moderate spawning, prioritize fair load distribution",
    },
    "mixed": {
        "spawn_intensity": 1.3,
        "peak_multiplier": 2.0,
        "reward_bias": "balanced",
        "description": "Balanced commercial and residential characteristics",
    },
    "arterial": {
        "spawn_intensity": 1.5,
        "peak_multiplier": 2.2,
        "reward_bias": "throughput",
        "description": "Major through-corridors, maximize flow capacity",
    },
    "hospital_sensitive": {
        "spawn_intensity": 0.8,
        "peak_multiplier": 1.3,
        "reward_bias": "emergency",
        "description": "Emergency vehicle priority dominates all other objectives",
    },
    "bottleneck": {
        "spawn_intensity": 2.0,
        "peak_multiplier": 3.0,
        "reward_bias": "throughput",
        "description": "Congestion-prone junctions, aggressive flow optimization",
    },
    "it_corridor": {
        "spawn_intensity": 2.2,
        "peak_multiplier": 3.5,
        "reward_bias": "throughput",
        "description": "Tech-park heavy corridors with extreme peak asymmetry",
    },
}


def get_zone_profile(zone_type: str) -> dict[str, Any]:
    """Return the traffic profile for a given zone type, with a safe default."""
    return ZONE_PROFILES.get(zone_type, ZONE_PROFILES["mixed"])
