"""Core metrics container for simulation and training reporting."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SimulationMetrics:
    avg_travel_time: float = 0.0
    avg_waiting_time: float = 0.0
    throughput: float = 0.0
    completed_trips: int = 0
    avg_speed: float = 0.0
    queue_length: float = 0.0
    reroute_count: int = 0
    ambulance_response_time: float = 0.0
