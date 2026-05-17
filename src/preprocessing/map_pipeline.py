"""Reusable map preprocessing helpers for maps/ area directories.

This module plans and tracks preprocessing; it does not run RL or TraCI control.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET


REQUIRED_SUBDIRS = (
    "raw_osm",
    "sumo_network",
    "routes",
    "trips",
    "grid_metadata",
    "preprocessed",
)

PLACEHOLDER_CONTENT = {
    "raw_osm": "Drop raw .osm export here.\n",
    "sumo_network": "Generated .net.xml/.edg.xml/.nod.xml/.con.xml/.poly.xml will be written here.\n",
    "routes": "Generated .rou.xml files will be written here.\n",
    "trips": "Generated .trips.xml files will be written here.\n",
    "grid_metadata": "Area-level metadata, junction classes, and clusters are written here.\n",
    "preprocessed": "Normalized outputs and checks are written here.\n",
}


@dataclass(frozen=True)
class AreaPaths:
    area_id: str
    area_root: Path
    raw_osm_dir: Path
    sumo_network_dir: Path
    routes_dir: Path
    trips_dir: Path
    grid_metadata_dir: Path
    preprocessed_dir: Path


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_maps_catalog(project_root: Path) -> dict[str, Any]:
    path = project_root / "configs" / "preprocessing" / "maps_area_catalog.json"
    return load_json(path)


def build_area_paths(maps_root: Path, area_id: str) -> AreaPaths:
    area_root = maps_root / area_id
    return AreaPaths(
        area_id=area_id,
        area_root=area_root,
        raw_osm_dir=area_root / "raw_osm",
        sumo_network_dir=area_root / "sumo_network",
        routes_dir=area_root / "routes",
        trips_dir=area_root / "trips",
        grid_metadata_dir=area_root / "grid_metadata",
        preprocessed_dir=area_root / "preprocessed",
    )


def ensure_maps_scaffold(project_root: Path, catalog: dict[str, Any] | None = None) -> None:
    if catalog is None:
        catalog = load_maps_catalog(project_root)
    maps_root = project_root / catalog.get("maps_root", "maps")
    maps_root.mkdir(parents=True, exist_ok=True)

    for area_id in catalog["areas"]:
        area_paths = build_area_paths(maps_root, area_id)
        for folder_name in REQUIRED_SUBDIRS:
            folder_path = area_paths.area_root / folder_name
            folder_path.mkdir(parents=True, exist_ok=True)
            placeholder = folder_path / f"PLACE_{folder_name.upper()}_HERE.txt"
            if not placeholder.exists():
                placeholder.write_text(PLACEHOLDER_CONTENT[folder_name], encoding="utf-8")


def detect_raw_osm(area_paths: AreaPaths) -> list[Path]:
    return sorted(area_paths.raw_osm_dir.glob("*.osm"))


def build_sumo_command_plan(
    area_paths: AreaPaths,
    scenario_id: str,
    begin: int = 0,
    end: int = 10800,
    period: float = 0.4,
    seed: int = 42,
    trip_attributes: str | None = None,
) -> dict[str, str]:
    if trip_attributes is None:
        trip_attributes = 'type="normal_car" departLane="best" departSpeed="max"'

    osm_files = detect_raw_osm(area_paths)
    osm_input = osm_files[0] if osm_files else (area_paths.raw_osm_dir / f"{area_paths.area_id}.osm")

    net_xml = area_paths.sumo_network_dir / f"{area_paths.area_id}.net.xml"
    nod_xml = area_paths.sumo_network_dir / f"{area_paths.area_id}.nod.xml"
    edg_xml = area_paths.sumo_network_dir / f"{area_paths.area_id}.edg.xml"
    con_xml = area_paths.sumo_network_dir / f"{area_paths.area_id}.con.xml"
    poly_xml = area_paths.sumo_network_dir / f"{area_paths.area_id}.poly.xml"
    trips_xml = area_paths.trips_dir / f"{scenario_id}.trips.xml"
    routes_xml = area_paths.routes_dir / f"{scenario_id}.rou.xml"

    random_trips_cmd = (
        f'python "$SUMO_HOME/tools/randomTrips.py" '
        f'-n "{net_xml}" -o "{trips_xml}" '
        f'--seed {seed} --begin {begin} --end {end} --period {period} '
        f'--validate --trip-attributes \'{trip_attributes}\''
    )

    return {
        "netconvert": (
            f'netconvert --osm-files "{osm_input}" --output-file "{net_xml}" '
            f'--plain-output-prefix "{area_paths.sumo_network_dir / area_paths.area_id}" '
            f"--junctions.join --ramps.guess --tls.guess-signals --tls.discard-simple"
        ),
        "polyconvert": (
            f'polyconvert --net-file "{net_xml}" --osm-files "{osm_input}" '
            f'--output-file "{poly_xml}"'
        ),
        "randomTrips": random_trips_cmd,
        "duarouter": (
            f'duarouter --net-file "{net_xml}" --route-files "{trips_xml}" '
            f'--output-file "{routes_xml}" --ignore-errors --repair'
        ),
        "artifacts": json.dumps(
            {
                "osm_input": str(osm_input),
                "net_xml": str(net_xml),
                "nod_xml": str(nod_xml),
                "edg_xml": str(edg_xml),
                "con_xml": str(con_xml),
                "poly_xml": str(poly_xml),
                "trips_xml": str(trips_xml),
                "routes_xml": str(routes_xml),
            },
            indent=2,
        ),
    }


def summarize_net_xml(net_xml_path: Path) -> dict[str, Any]:
    if not net_xml_path.exists():
        return {
            "exists": False,
            "edge_count": 0,
            "junction_count": 0,
            "traffic_signal_count": 0,
            "average_lane_count": 0.0,
        }

    root = ET.parse(net_xml_path).getroot()
    edges = [edge for edge in root.findall("edge") if edge.get("function") != "internal"]
    junctions = [junction for junction in root.findall("junction") if not junction.get("id", "").startswith(":")]
    lane_counts = [len(edge.findall("lane")) for edge in edges]
    traffic_signal_count = sum(1 for junction in junctions if junction.get("type") == "traffic_light")
    average_lane_count = (sum(lane_counts) / len(lane_counts)) if lane_counts else 0.0

    return {
        "exists": True,
        "edge_count": len(edges),
        "junction_count": len(junctions),
        "traffic_signal_count": traffic_signal_count,
        "average_lane_count": round(average_lane_count, 3),
        "junction_classification": classify_junctions(root),
    }


def classify_junctions(root: ET.Element) -> dict[str, int]:
    result = {
        "three_way": 0,
        "four_way": 0,
        "roundabout_like": 0,
        "arterial_merge": 0,
        "corridor_junction": 0,
        "other": 0,
    }
    for junction in root.findall("junction"):
        junction_id = junction.get("id", "")
        if junction_id.startswith(":"):
            continue
        junction_type = junction.get("type", "")
        incoming_lanes = junction.get("incLanes", "").split()
        incoming_count = len(incoming_lanes)

        if junction_type == "traffic_light" and incoming_count >= 8:
            result["corridor_junction"] += 1
        elif junction_type in {"priority", "traffic_light"} and incoming_count in {5, 6}:
            result["three_way"] += 1
        elif junction_type in {"priority", "traffic_light"} and incoming_count >= 7:
            result["four_way"] += 1
        elif "roundabout" in junction_type.lower():
            result["roundabout_like"] += 1
        elif incoming_count <= 3:
            result["arterial_merge"] += 1
        else:
            result["other"] += 1
    return result


def build_area_report(
    area_id: str,
    area_meta: dict[str, Any],
    area_paths: AreaPaths,
    scenario_id: str,
) -> dict[str, Any]:
    commands = build_sumo_command_plan(area_paths, scenario_id=scenario_id)
    osm_files = detect_raw_osm(area_paths)
    net_xml = area_paths.sumo_network_dir / f"{area_id}.net.xml"

    return {
        "area_id": area_id,
        "label": area_meta.get("label", area_id),
        "zone_type": area_meta.get("zone_type", "unknown"),
        "parent_domain": area_meta.get("parent_domain", "unknown"),
        "raw_osm_count": len(osm_files),
        "raw_osm_files": [str(path) for path in osm_files],
        "net_summary": summarize_net_xml(net_xml),
        "sumo_command_plan": commands,
    }


def write_area_report(area_paths: AreaPaths, report: dict[str, Any]) -> Path:
    output_path = area_paths.grid_metadata_dir / "preprocessing_report.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    return output_path


def build_pipeline_report(
    project_root: Path,
    scenario_id: str,
) -> dict[str, Any]:
    catalog = load_maps_catalog(project_root)
    maps_root = project_root / catalog.get("maps_root", "maps")
    ensure_maps_scaffold(project_root, catalog)

    reports = []
    for area_id, area_meta in catalog["areas"].items():
        area_paths = build_area_paths(maps_root, area_id)
        report = build_area_report(area_id, area_meta, area_paths, scenario_id)
        write_area_report(area_paths, report)
        reports.append(report)

    return {
        "schema_version": catalog.get("schema_version", "0.1.0"),
        "scenario_id": scenario_id,
        "maps_root": str(maps_root),
        "area_count": len(reports),
        "areas": reports,
    }


def write_pipeline_report(project_root: Path, scenario_id: str) -> Path:
    report = build_pipeline_report(project_root, scenario_id)
    output_path = project_root / "maps" / f"{scenario_id}_pipeline_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    return output_path
