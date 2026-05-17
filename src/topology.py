"""Static geographic intelligence for the ward-based hierarchy.

Owns ward definitions, area membership, adjacency graphs, boundary
connectivity, map stitching, and edge ownership.

Does NOT own: live traffic state, RL logic, simulation stepping.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

try:
    import networkx as nx
except ImportError:  # pragma: no cover
    nx = None

logger = logging.getLogger(__name__)


class Topology:
    """Pure geographic truth for the hierarchical traffic system.

    Loads ward registry and region hierarchy, builds adjacency structures,
    and provides graph operations for the GNN and city controller.
    """

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.ward_registry = self._load_json("configs/hierarchy/ward_registry.json")
        self.blr_regions = self._load_json("configs/hierarchy/blr_regions.json")

        self._wards: dict[str, dict[str, Any]] = self.ward_registry.get("wards", {})
        self._ward_to_area: dict[str, str] = {}
        self._area_to_wards: dict[str, list[str]] = {}
        self._build_area_mapping()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _load_json(self, relative_path: str) -> dict[str, Any]:
        path = self.project_root / relative_path
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def _build_area_mapping(self) -> None:
        """Build ward<->area lookup tables from the ward registry."""
        for area_id, area_meta in self.ward_registry.get("areas", {}).items():
            ward_ids = area_meta.get("wards", [])
            self._area_to_wards[area_id] = ward_ids
            for wid in ward_ids:
                self._ward_to_area[wid] = area_id

    # ------------------------------------------------------------------
    # Ward queries
    # ------------------------------------------------------------------

    def get_all_ward_ids(self) -> list[str]:
        """Return all registered ward IDs."""
        return sorted(self._wards.keys())

    def get_ward_metadata(self, ward_id: str) -> dict[str, Any]:
        """Return metadata for a single ward."""
        return self._wards.get(ward_id, {})

    def get_ward_zone_type(self, ward_id: str) -> str:
        return self._wards.get(ward_id, {}).get("zone_type", "mixed")

    def get_ward_neighbors(self, ward_id: str) -> list[str]:
        """Return neighbor ward IDs from the registry."""
        meta = self._wards.get(ward_id, {})
        neighbor_nums = meta.get("neighbors", [])
        return [f"ward_{n:03d}" for n in neighbor_nums]

    def get_ward_boundaries(self, ward_id: str) -> dict[str, Any]:
        """Load boundaries.json for a processed ward."""
        path = self.project_root / "maps" / os.getenv("HMRL_MAP_DIR", "processed") / ward_id / "boundaries.json"
        if not path.exists():
            return {"valid_ingress_edges": [], "valid_egress_edges": [], "spawn_candidates": []}
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    # ------------------------------------------------------------------
    # Area queries
    # ------------------------------------------------------------------

    def get_all_area_ids(self) -> list[str]:
        return sorted(self._area_to_wards.keys())

    def get_area_wards(self, area_id: str) -> list[str]:
        return self._area_to_wards.get(area_id, [])

    def get_ward_area(self, ward_id: str) -> str:
        return self._ward_to_area.get(ward_id, "unknown")

    # ------------------------------------------------------------------
    # Adjacency graph
    # ------------------------------------------------------------------

    def build_ward_graph(self, area_id: str) -> Any:
        """Build a NetworkX graph of ward adjacency within an area.

        Returns:
            ``networkx.Graph`` with ward IDs as nodes.
        """
        if nx is None:
            raise ImportError("networkx is required for graph operations")

        ward_ids = self.get_area_wards(area_id)
        ward_set = set(ward_ids)
        graph = nx.Graph()

        for wid in ward_ids:
            graph.add_node(wid, **self.get_ward_metadata(wid))
            for neighbor in self.get_ward_neighbors(wid):
                if neighbor in ward_set:
                    graph.add_edge(wid, neighbor)

        return graph

    def get_adjacency_matrix(self, area_id: str) -> np.ndarray:
        """Compute the normalized adjacency matrix for the GNN.

        Returns:
            ``A_hat = D^{-1/2} (A + I) D^{-1/2}`` as a numpy array.
        """
        ward_ids = self.get_area_wards(area_id)
        n = len(ward_ids)
        ward_index = {wid: i for i, wid in enumerate(ward_ids)}

        # Build A + I (adjacency with self-loops)
        adj = np.eye(n, dtype=np.float32)
        for wid in ward_ids:
            i = ward_index[wid]
            for neighbor in self.get_ward_neighbors(wid):
                if neighbor in ward_index:
                    j = ward_index[neighbor]
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0

        # Normalize: D^{-1/2} A_hat D^{-1/2}
        degree = adj.sum(axis=1)
        d_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degree, 1e-8)))
        a_hat = d_inv_sqrt @ adj @ d_inv_sqrt

        return a_hat

    # ------------------------------------------------------------------
    # Map stitching
    # ------------------------------------------------------------------

    def stitch_ward_maps(
        self,
        ward_ids: list[str],
        output_name: str | None = None,
    ) -> Path:
        """Merge multiple ward networks into a single SUMO net file.

        Uses netconvert to concatenate ward networks, merging boundary
        nodes where wards are adjacent.

        Args:
            ward_ids: List of ward IDs to stitch together.

        Returns:
            Path to the stitched ``area.net.xml``.
        """
        import subprocess

        net_files = []
        for wid in ward_ids:
            net = self.project_root / "maps" / os.getenv("HMRL_MAP_DIR", "processed") / wid / "ward.net.xml"
            if not net.exists():
                raise FileNotFoundError(f"Network not found for {wid}: {net}")
            net_files.append(str(net))

        # Derive a stable output label from the first ward unless the caller
        # supplies a custom name for a multi-area deployment.
        area_id = output_name or self._ward_to_area.get(ward_ids[0], "area")
        output_dir = self.project_root / "maps" / "stitched" / area_id
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{area_id}.net.xml"

        cmd = [
            "netconvert",
            "--sumo-net-file", ",".join(net_files),
            "--output-file", str(output_path),
            "--junctions.join",
            "--no-turnarounds",
        ]

        logger.info("Stitching %d ward networks → %s", len(ward_ids), output_path)
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error("netconvert stitching failed:\n%s", result.stderr)
            raise RuntimeError(f"Stitching failed: {result.stderr[:400]}")

        logger.info("Stitched network: %s", output_path)
        return output_path

    # ------------------------------------------------------------------
    # City-level macro graph
    # ------------------------------------------------------------------

    def build_area_graph(self) -> Any:
        """Build a macro graph of inter-area connectivity for the city layer.

        Two areas are connected if any of their wards are neighbors.

        Returns:
            ``networkx.Graph`` with area IDs as nodes.
        """
        if nx is None:
            raise ImportError("networkx is required for graph operations")

        graph = nx.Graph()
        for area_id in self.get_all_area_ids():
            graph.add_node(area_id)

        for area_id, ward_ids in self._area_to_wards.items():
            for wid in ward_ids:
                for neighbor in self.get_ward_neighbors(wid):
                    neighbor_area = self._ward_to_area.get(neighbor)
                    if neighbor_area and neighbor_area != area_id:
                        graph.add_edge(area_id, neighbor_area)

        return graph

    # ------------------------------------------------------------------
    # Edge ownership
    # ------------------------------------------------------------------

    def get_edge_owner(self, edge_id: str, ward_edges: dict[str, list[str]]) -> str:
        """Determine which ward owns a given SUMO edge ID.

        Args:
            edge_id: SUMO edge identifier.
            ward_edges: Mapping of ward_id → list of edge IDs.

        Returns:
            Ward ID that owns the edge, or ``"unowned"``.
        """
        for wid, edges in ward_edges.items():
            if edge_id in edges:
                return wid
        return "unowned"
