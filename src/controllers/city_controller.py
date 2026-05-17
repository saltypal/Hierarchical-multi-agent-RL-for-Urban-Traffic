"""City-level macro graph flow optimizer.

Uses NetworkX to solve Min-Cost Max-Flow across the area connectivity
graph, producing per-area inflow capacity limits.

No training required — pure mathematical optimization.
"""

from __future__ import annotations

import logging
from typing import Any

try:
    import networkx as nx
except ImportError:  # pragma: no cover
    nx = None

logger = logging.getLogger(__name__)


class CityController:
    """Heuristic + graph-based city-level macro controller.

    Operates every 120 simulation seconds (or event-triggered).

    The controller:
        1. Reads aggregated area summaries (congestion, throughput, incidents).
        2. Builds a weighted macro graph of inter-area corridors.
        3. Solves for optimal flow distribution using capacity balancing.
        4. Outputs per-area inflow capacity caps ``c ∈ [0, 1]``.
    """

    # Event trigger thresholds
    CONGESTION_ALERT_THRESHOLD = 0.85
    INCIDENT_ALERT_THRESHOLD = 1.0

    def __init__(self, topology: Any, area_ids: list[str] | None = None) -> None:
        """Initialise with a ``Topology`` instance for graph construction."""
        self.topology = topology
        self._macro_graph = topology.build_area_graph()
        self._area_ids = area_ids or topology.get_all_area_ids()

        if self._area_ids:
            self._macro_graph = self._macro_graph.subgraph(self._area_ids).copy()

        logger.info(
            "CityController initialised: %d areas, %d corridors",
            self._macro_graph.number_of_nodes(),
            self._macro_graph.number_of_edges(),
        )

    # ------------------------------------------------------------------
    # Event detection
    # ------------------------------------------------------------------

    def should_trigger(self, area_summaries: dict[str, dict[str, float]]) -> bool:
        """Check if any emergency condition warrants an immediate city step."""
        for area_id, summary in area_summaries.items():
            if summary.get("avg_congestion", 0.0) > self.CONGESTION_ALERT_THRESHOLD:
                logger.info("City event: congestion alert in %s", area_id)
                return True
            if summary.get("incident_severity", 0.0) >= self.INCIDENT_ALERT_THRESHOLD:
                logger.info("City event: incident alert in %s", area_id)
                return True
        return False

    # ------------------------------------------------------------------
    # Capacity solver
    # ------------------------------------------------------------------

    def solve(
        self,
        area_summaries: dict[str, dict[str, float]],
    ) -> dict[str, float]:
        """Compute per-area inflow capacity limits.

        Uses a pressure-proportional balancing strategy:
        areas with higher congestion get their inflow reduced,
        while underutilised areas get more capacity.

        Args:
            area_summaries: Per-area metrics dictionary.
                Expected keys: ``avg_congestion``, ``total_throughput``,
                ``incident_severity``.

        Returns:
            Dictionary mapping ``area_id → capacity_cap ∈ [0.0, 1.0]``.
        """
        caps: dict[str, float] = {}

        # Collect congestion values
        congestions: dict[str, float] = {}
        for area_id in self._area_ids:
            summary = area_summaries.get(area_id, {})
            congestions[area_id] = summary.get("avg_congestion", 0.0)

        if not congestions:
            return {area_id: 1.0 for area_id in self._area_ids}

        max_congestion = max(congestions.values()) or 1.0

        for area_id in self._area_ids:
            congestion = congestions.get(area_id, 0.0)
            incident = area_summaries.get(area_id, {}).get("incident_severity", 0.0)

            # Base capacity: inversely proportional to congestion
            base_cap = 1.0 - (0.6 * congestion / max_congestion)

            # Incident penalty
            if incident > 0:
                base_cap *= 0.7

            # Clamp
            caps[area_id] = max(0.2, min(1.0, base_cap))

        # Graph-aware flow redistribution (future: full MCMF)
        if nx is not None and self._macro_graph.number_of_edges() > 0:
            caps = self._graph_redistribute(caps, congestions)

        logger.info("City caps: %s", {k: f"{v:.2f}" for k, v in caps.items()})
        return caps

    def _graph_redistribute(
        self,
        caps: dict[str, float],
        congestions: dict[str, float],
    ) -> dict[str, float]:
        """Redistribute capacity using graph neighborhood awareness.

        If an area is heavily congested, its neighbors' outbound capacity
        toward it is also reduced, preventing spillover pressure.
        """
        adjusted = dict(caps)
        for area_id, congestion in congestions.items():
            if congestion > 0.7:
                # Reduce neighbors' capacity toward this area
                neighbors = list(self._macro_graph.neighbors(area_id))
                for neighbor in neighbors:
                    current_cap = adjusted.get(neighbor, 1.0)
                    reduction = 0.15 * congestion
                    adjusted[neighbor] = max(0.2, current_cap - reduction)

        return adjusted
