"""Ward-level map preprocessing pipeline.

Converts raw OSM files into SUMO networks, extracts structural metadata,
and detects boundary edges for traffic generation.

Reuses ``summarize_net_xml`` and ``classify_junctions`` from the existing
``map_pipeline`` module for network analysis.
"""

from __future__ import annotations

import json
import logging
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from .map_pipeline import classify_junctions, summarize_net_xml
from .osm_fetcher import load_ward_registry

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Path helpers
# ------------------------------------------------------------------

import os


def ward_processed_dir(
    project_root: Path,
    ward_id: str,
    output_dir_name: str | None = None,
) -> Path:
    """Return the processed asset directory for a ward."""
    map_dir = output_dir_name or os.getenv("HMRL_MAP_DIR", "processed")
    return project_root / "maps" / map_dir / ward_id


def ward_osm_path(project_root: Path, ward_id: str) -> Path:
    """Return the raw OSM file path for a ward."""
    return project_root / "maps" / "raw_osm" / f"{ward_id}.osm"


def ward_net_path(
    project_root: Path,
    ward_id: str,
    output_dir_name: str | None = None,
) -> Path:
    """Return the compiled net.xml path for a ward."""
    return ward_processed_dir(
        project_root, ward_id, output_dir_name=output_dir_name,
    ) / "ward.net.xml"


# ------------------------------------------------------------------
# OSM → SUMO network conversion
# ------------------------------------------------------------------

def convert_osm_to_net(
    ward_id: str,
    project_root: Path,
    output_dir_name: str | None = None,
    extra_netconvert_args: list[str] | None = None,
) -> Path:
    """Run ``netconvert`` to compile a ward OSM file into a SUMO network.

    Returns:
        Path to the generated ``ward.net.xml``.
    """
    osm_path = ward_osm_path(project_root, ward_id)
    if not osm_path.exists():
        raise FileNotFoundError(f"OSM file not found: {osm_path}")

    output_dir = ward_processed_dir(
        project_root, ward_id, output_dir_name=output_dir_name,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    net_path = output_dir / "ward.net.xml"

    cmd = [
        "netconvert",
        "--osm-files", str(osm_path),
        "--output-file", str(net_path),
        "--junctions.join",
        "--junctions.join-same", "1.0",
        "--edges.join",
        "--ramps.guess",
        "--tls.guess-signals",
        "--tls.discard-simple",
        "--geometry.remove",
        "--roundabouts.guess",
        "--no-turnarounds",
    ]
    if extra_netconvert_args:
        cmd.extend(extra_netconvert_args)


    logger.info("Running netconvert for %s", ward_id)
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error("netconvert failed for %s:\n%s", ward_id, result.stderr)
        raise RuntimeError(f"netconvert failed for {ward_id}: {result.stderr[:500]}")

    logger.info("Generated network: %s", net_path)
    return net_path


# ------------------------------------------------------------------
# Metadata extraction
# ------------------------------------------------------------------

def extract_ward_metadata(
    ward_id: str,
    project_root: Path,
    output_dir_name: str | None = None,
) -> dict[str, Any]:
    """Parse the compiled net.xml and extract structural metadata.

    Combines network statistics with ward registry information.

    Returns:
        Metadata dictionary, also written to ``metadata.json``.
    """
    net_path = ward_net_path(
        project_root, ward_id, output_dir_name=output_dir_name,
    )
    registry = load_ward_registry(project_root)
    ward_meta = registry["wards"].get(ward_id, {})

    net_summary = summarize_net_xml(net_path)

    metadata: dict[str, Any] = {
        "ward_id": ward_id,
        "label": ward_meta.get("label", ward_id),
        "zone_type": ward_meta.get("zone_type", "mixed"),
        "parent_area": ward_meta.get("parent_area", "unknown"),
        "parent_region": ward_meta.get("parent_region", "unknown"),
        "congestion_prior": ward_meta.get("congestion_prior", "medium"),
        "hospital_sensitive": ward_meta.get("hospital_sensitive", False),
        "priority_level": ward_meta.get("priority_level", 1),
        "network": net_summary,
    }

    output_path = ward_processed_dir(
        project_root, ward_id, output_dir_name=output_dir_name,
    ) / "metadata.json"
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
        fh.write("\n")

    logger.info("Wrote metadata for %s → %s", ward_id, output_path)
    return metadata


# ------------------------------------------------------------------
# Boundary edge detection
# ------------------------------------------------------------------

def detect_ward_boundaries(
    ward_id: str,
    project_root: Path,
    strict_mode: bool = True,
    output_dir_name: str | None = None,
) -> dict[str, Any]:
    """Analyse the SUMO network to identify boundary edges.

    Boundary edges are dead-end edges or edges at the network perimeter,
    suitable for ingress/egress spawning by the traffic generator.

    Args:
        ward_id: Ward ID.
        project_root: Project root.
        strict_mode: If True, filters out pedestrian footways, steps, and cycleways.

    Returns:
        Boundary dictionary, also written to ``boundaries.json``.
    """
    net_path = ward_net_path(
        project_root, ward_id, output_dir_name=output_dir_name,
    )
    if not net_path.exists():
        raise FileNotFoundError(f"Network not found: {net_path}")

    root = ET.parse(net_path).getroot()

    # Build edge connectivity
    edges: dict[str, dict[str, Any]] = {}
    for edge_elem in root.findall("edge"):
        eid = edge_elem.get("id", "")
        if eid.startswith(":"):
            continue

        if strict_mode:
            # Filter out edges that do not allow passenger vehicles (e.g. footways, steps, cycleways)
            has_passenger_lane = False
            for lane_elem in edge_elem.findall("lane"):
                disallowed = (lane_elem.get("disallow", "") or "").split()
                allowed = (lane_elem.get("allow", "") or "").split()

                is_allowed = True
                if allowed and "passenger" not in allowed:
                    is_allowed = False
                if disallowed and ("passenger" in disallowed or "all" in disallowed):
                    is_allowed = False

                if is_allowed:
                    has_passenger_lane = True
                    break

            if not has_passenger_lane:
                continue

        edges[eid] = {
            "from": edge_elem.get("from", ""),
            "to": edge_elem.get("to", ""),
            "lanes": len(edge_elem.findall("lane")),
        }

    # Build junction degree maps
    junction_in_degree: dict[str, int] = {}
    junction_out_degree: dict[str, int] = {}
    for eid, edata in edges.items():
        from_j = edata["from"]
        to_j = edata["to"]
        junction_out_degree[from_j] = junction_out_degree.get(from_j, 0) + 1
        junction_in_degree[to_j] = junction_in_degree.get(to_j, 0) + 1

    # Classify boundary and internal edges
    ingress_edges: list[dict[str, Any]] = []
    egress_edges: list[dict[str, Any]] = []
    spawn_candidates: list[dict[str, Any]] = []
    internal_edges: list[dict[str, Any]] = []

    boundary_edge_ids: set[str] = set()

    for eid, edata in edges.items():
        from_j = edata["from"]
        to_j = edata["to"]
        from_in = junction_in_degree.get(from_j, 0)
        to_out = junction_out_degree.get(to_j, 0)
        edge_entry = {"edge_id": eid, "lanes": edata["lanes"]}

        # Source-like: junction has no incoming edges → ingress point
        if from_in == 0:
            ingress_edges.append(edge_entry)
            spawn_candidates.append(edge_entry)
            boundary_edge_ids.add(eid)

        # Sink-like: junction has no outgoing edges → egress point
        if to_out == 0:
            egress_edges.append(edge_entry)
            boundary_edge_ids.add(eid)

    # Internal edges: drivable edges that are neither ingress nor egress
    for eid, edata in edges.items():
        if eid not in boundary_edge_ids:
            internal_edges.append({"edge_id": eid, "lanes": edata["lanes"]})

    boundaries: dict[str, Any] = {
        "ward_id": ward_id,
        "total_edges": len(edges),
        "valid_ingress_edges": sorted(ingress_edges, key=lambda e: e["edge_id"]),
        "valid_egress_edges": sorted(egress_edges, key=lambda e: e["edge_id"]),
        "spawn_candidates": sorted(spawn_candidates, key=lambda e: e["edge_id"]),
        "internal_edges": sorted(internal_edges, key=lambda e: e["edge_id"]),
        "ingress_count": len(ingress_edges),
        "egress_count": len(egress_edges),
        "internal_edge_count": len(internal_edges),
    }

    output_path = ward_processed_dir(
        project_root, ward_id, output_dir_name=output_dir_name,
    ) / "boundaries.json"
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(boundaries, fh, indent=2)
        fh.write("\n")

    logger.info(
        "Boundaries for %s: %d ingress, %d egress, %d internal edges (strict_mode=%s)",
        ward_id, len(ingress_edges), len(egress_edges), len(internal_edges), strict_mode,
    )
    return boundaries


# ------------------------------------------------------------------
# Orchestration
# ------------------------------------------------------------------

def process_ward(
    ward_id: str,
    project_root: Path,
    strict_mode: bool = True,
    output_dir_name: str | None = None,
    extra_netconvert_args: list[str] | None = None,
) -> dict[str, Any]:
    """Run the full preprocessing pipeline for a single ward.

    Steps:
        1. ``netconvert`` : OSM → SUMO network
        2. Metadata extraction from compiled network
        3. Boundary edge detection

    Returns:
        Combined result dictionary.
    """
    logger.info("Processing ward: %s", ward_id)

    net_path = convert_osm_to_net(
        ward_id,
        project_root,
        output_dir_name=output_dir_name,
        extra_netconvert_args=extra_netconvert_args,
    )
    metadata = extract_ward_metadata(
        ward_id, project_root, output_dir_name=output_dir_name,
    )
    boundaries = detect_ward_boundaries(
        ward_id,
        project_root,
        strict_mode=strict_mode,
        output_dir_name=output_dir_name,
    )

    return {
        "ward_id": ward_id,
        "net_path": str(net_path),
        "metadata": metadata,
        "boundaries": boundaries,
        "status": "ok",
    }


def process_all_wards(project_root: Path, strict_mode: bool = True) -> list[dict[str, Any]]:
    """Batch-process every ward that has an OSM file in ``maps/raw_osm/``.

    Returns:
        List of per-ward result dictionaries.
    """
    raw_dir = project_root / "maps" / "raw_osm"
    osm_files = sorted(raw_dir.glob("ward_*.osm"))

    if not osm_files:
        logger.warning("No ward OSM files found in %s", raw_dir)
        return []

    results: list[dict[str, Any]] = []
    for osm_file in osm_files:
        ward_id = osm_file.stem  # e.g. "ward_001"
        try:
            result = process_ward(ward_id, project_root, strict_mode=strict_mode)
            results.append(result)
        except Exception as exc:
            logger.error("Failed to process %s: %s", ward_id, exc)
            results.append({"ward_id": ward_id, "status": f"error: {exc}"})

    return results
