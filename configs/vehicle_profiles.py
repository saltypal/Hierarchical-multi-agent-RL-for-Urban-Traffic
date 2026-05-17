"""Heterogeneous SUMO vehicle type definitions.

Each profile maps directly to a SUMO ``vType`` element written into
``.rou.xml`` files during route generation. Profiles are referenced by
the traffic generator and scenario configurations.

Units:
    maxSpeed : m/s (e.g. 16.67 m/s ≈ 60 km/h)
    accel    : m/s²
    decel    : m/s²
    sigma    : driver imperfection [0..1]
    minGap   : metres
    length   : metres
"""

from __future__ import annotations

from typing import Any


VEHICLE_PROFILES: dict[str, dict[str, Any]] = {
    "normal_car": {
        "vClass": "passenger",
        "maxSpeed": 16.67,
        "accel": 2.6,
        "decel": 4.5,
        "sigma": 0.5,
        "minGap": 2.5,
        "length": 5.0,
        "color": "0.8,0.8,0.8",
    },
    "aggressive": {
        "vClass": "passenger",
        "maxSpeed": 22.22,
        "accel": 4.0,
        "decel": 6.0,
        "sigma": 0.9,
        "minGap": 1.0,
        "length": 5.0,
        "color": "1.0,0.2,0.2",
    },
    "slow_driver": {
        "vClass": "passenger",
        "maxSpeed": 11.11,
        "accel": 1.5,
        "decel": 3.0,
        "sigma": 0.2,
        "minGap": 4.0,
        "length": 5.0,
        "color": "0.5,0.5,1.0",
    },
    "bmtc_bus": {
        "vClass": "bus",
        "maxSpeed": 11.11,
        "accel": 1.0,
        "decel": 3.0,
        "sigma": 0.3,
        "minGap": 3.0,
        "length": 12.0,
        "color": "0.0,0.6,0.0",
    },
    "truck": {
        "vClass": "truck",
        "maxSpeed": 13.89,
        "accel": 0.8,
        "decel": 2.5,
        "sigma": 0.3,
        "minGap": 3.5,
        "length": 10.0,
        "color": "0.6,0.4,0.2",
    },
    "ambulance": {
        "vClass": "emergency",
        "maxSpeed": 22.22,
        "accel": 3.5,
        "decel": 5.0,
        "sigma": 0.1,
        "minGap": 1.5,
        "length": 6.5,
        "color": "1.0,1.0,1.0",
    },
    "govt_convoy": {
        "vClass": "authority",
        "maxSpeed": 16.67,
        "accel": 2.0,
        "decel": 4.0,
        "sigma": 0.1,
        "minGap": 5.0,
        "length": 5.5,
        "color": "0.0,0.0,0.0",
    },
}


def get_vehicle_profile(profile_name: str) -> dict[str, Any]:
    """Return a vehicle profile by name, falling back to normal_car."""
    return VEHICLE_PROFILES.get(profile_name, VEHICLE_PROFILES["normal_car"])


def build_sumo_vtype_xml(profile_name: str, profile: dict[str, Any]) -> str:
    """Generate a SUMO ``<vType>`` XML element string."""
    attrs = " ".join(
        f'{key}="{value}"'
        for key, value in profile.items()
    )
    return f'<vType id="{profile_name}" {attrs}/>'


def build_all_vtypes_xml() -> str:
    """Generate XML for all vehicle type definitions."""
    lines = ['<!-- Auto-generated vehicle type definitions -->']
    for name, profile in VEHICLE_PROFILES.items():
        lines.append(build_sumo_vtype_xml(name, profile))
    return "\n".join(lines)
