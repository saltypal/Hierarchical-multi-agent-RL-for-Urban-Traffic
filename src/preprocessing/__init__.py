"""Preprocessing package for map processing and route generation.

This package must stay independent from RL, RSU control, PPO/DQN training,
and TraCI vehicle-control logic.
"""

from .map_pipeline import (
    AreaPaths,
    build_area_paths,
    build_pipeline_report,
    ensure_maps_scaffold,
    load_maps_catalog,
    write_pipeline_report,
)
from .osm_fetcher import (
    fetch_all_wards,
    fetch_ward_osm,
    load_ward_registry,
    validate_osm_file,
)
from .ward_processor import (
    convert_osm_to_net,
    detect_ward_boundaries,
    extract_ward_metadata,
    process_all_wards,
    process_ward,
)

__all__ = [
    # Legacy area-based pipeline
    "AreaPaths",
    "build_area_paths",
    "build_pipeline_report",
    "ensure_maps_scaffold",
    "load_maps_catalog",
    "write_pipeline_report",
    # Ward-based pipeline
    "convert_osm_to_net",
    "detect_ward_boundaries",
    "extract_ward_metadata",
    "fetch_all_wards",
    "fetch_ward_osm",
    "load_ward_registry",
    "process_all_wards",
    "process_ward",
    "validate_osm_file",
]

