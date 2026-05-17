"""Discrete semantic ward action catalog.

These actions are policy-level traffic actions. They do not replace SUMO's
vehicle dynamics with a different simulator.
"""

from __future__ import annotations

from enum import IntEnum


class WardAction(IntEnum):
    NO_OP = 0
    REROUTE_HOTSPOT_GROUP = 1
    DEPRIORITIZE_MOST_CONGESTED_EDGE = 2
    PRIORITIZE_ALTERNATE_EDGE = 3
    CLEAR_AMBULANCE_PATH = 4
    INCIDENT_REROUTE = 5
    HOLD_COMMERCIAL_INFLOW = 6
    RELEASE_HELD_FLOW = 7
    REROUTE_AGGRESSIVE_DRIVERS = 8
    REROUTE_HEAVY_VEHICLES = 9


WARD_ACTIONS = {
    action.value: action.name
    for action in WardAction
}
