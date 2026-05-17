"""Cross-layer directive schema for hierarchical control."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class DirectiveType(str, Enum):
    NO_OP = "NO_OP"
    REROUTE_AWAY_FROM_EDGE = "REROUTE_AWAY_FROM_EDGE"
    REDUCE_INGRESS_TO_WARD = "REDUCE_INGRESS_TO_WARD"
    PRIORITIZE_AMBULANCE_ROUTE = "PRIORITIZE_AMBULANCE_ROUTE"
    INCIDENT_REROUTE = "INCIDENT_REROUTE"
    HOLD_COMMERCIAL_INFLOW = "HOLD_COMMERCIAL_INFLOW"
    RELEASE_HELD_FLOW = "RELEASE_HELD_FLOW"


@dataclass(frozen=True)
class Directive:
    source_layer: str
    directive_type: DirectiveType
    target_scope: str
    target_id: str
    priority: int = 0
    ttl_seconds: int = 30
    cooldown_seconds: int = 0
    reason: str = ""
    confidence: float = 1.0
    metadata: dict[str, str] = field(default_factory=dict)
