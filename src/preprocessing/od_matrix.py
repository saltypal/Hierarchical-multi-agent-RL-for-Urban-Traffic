"""Config-driven OD matrix generation for preprocessing.

The first implementation works at area level because real TAZ extraction is not
implemented yet. Once TAZs exist, this module can receive TAZ-level origin and
destination weights without changing the notebook or route-generation stages.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import random
from pathlib import Path
from typing import Any


DEFAULT_SCENARIO_ID = "wednesday_morning_office_commute_v1"


@dataclass(frozen=True)
class ODMatrixResult:
    """Generated OD matrix plus artifact paths."""

    scenario_id: str
    entries: list[dict[str, Any]]
    json_path: Path
    csv_path: Path


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_area_od_matrix(
    region_config: dict[str, Any],
    od_config: dict[str, Any],
    scenario_id: str = DEFAULT_SCENARIO_ID,
    maps_catalog: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build a reproducible area-level OD matrix from configured priors.

    This intentionally avoids pretending that we already have Google Maps or
    measured OD data. The matrix is synthetic, configurable, and traceable.
    """

    scenario = od_config["scenarios"][scenario_id]
    area_map = _resolve_area_map(region_config, scenario, maps_catalog)
    area_weights = scenario["area_weights"]

    random_source = random.Random(scenario.get("seed", 0))
    raw_pairs: list[dict[str, Any]] = []

    for origin_id, origin in area_map.items():
        for destination_id, destination in area_map.items():
            if origin_id == destination_id:
                continue

            origin_weight = _area_weight(area_weights, origin_id, "origin_weight")
            destination_weight = _area_weight(
                area_weights, destination_id, "destination_weight"
            )
            calibration = _pair_calibration(area_weights, origin_id, destination_id)

            score = origin_weight * destination_weight
            score *= calibration["congestion_weight"]
            score *= calibration["route_pressure"]
            score *= calibration["corridor_density"]
            score *= calibration["peak_hour_intensity"]
            score *= _domain_multiplier(scenario, origin, destination)
            score *= _corridor_multiplier(scenario, origin, destination)
            score *= _jitter_multiplier(scenario, random_source)

            raw_pairs.append(
                {
                    "scenario_id": scenario_id,
                    "origin_area_id": origin_id,
                    "destination_area_id": destination_id,
                    "origin_taz_id": f"{origin_id}:area_origin",
                    "destination_taz_id": f"{destination_id}:area_destination",
                    "score": score,
                    "time_window": scenario["time_window"],
                    "day_type": scenario["day_type"],
                    "day_name": scenario["day_name"],
                    "dominant_flow": scenario["dominant_flow"],
                    "calibration_features": calibration,
                }
            )

    total_score = sum(pair["score"] for pair in raw_pairs)
    if total_score <= 0:
        raise ValueError("OD matrix score total must be positive")

    total_vehicle_count = int(scenario["total_vehicle_count"])
    entries = []
    for pair in raw_pairs:
        probability = pair["score"] / total_score
        entry = dict(pair)
        entry["probability"] = probability
        entry["expected_vehicle_count"] = int(probability * total_vehicle_count)
        entries.append(entry)

    _distribute_rounding_residue(entries, total_vehicle_count)
    entries.sort(
        key=lambda item: (
            item["origin_area_id"],
            item["destination_area_id"],
        )
    )
    return entries


def generate_and_write_area_od_matrix(
    project_root: Path,
    scenario_id: str = DEFAULT_SCENARIO_ID,
    output_root: Path | None = None,
) -> ODMatrixResult:
    """Generate an OD matrix and write JSON + CSV artifacts."""

    config_root = project_root / "configs" / "preprocessing"
    region_config = load_json(config_root / "region_decomposition.json")
    od_config = load_json(config_root / "od_matrix_config.json")
    maps_catalog_path = config_root / "maps_area_catalog.json"
    maps_catalog = load_json(maps_catalog_path) if maps_catalog_path.exists() else None

    entries = build_area_od_matrix(
        region_config,
        od_config,
        scenario_id,
        maps_catalog=maps_catalog,
    )
    scenario = od_config["scenarios"][scenario_id]

    if output_root is None:
        output_root = project_root / od_config.get("default_output_root", "maps/od_matrices")

    scenario_output_root = output_root / scenario_id
    scenario_output_root.mkdir(parents=True, exist_ok=True)

    json_path = scenario_output_root / "area_od_matrix.json"
    csv_path = scenario_output_root / "area_od_matrix.csv"

    payload = {
        "schema_version": od_config["schema_version"],
        "scenario": {
            "scenario_id": scenario_id,
            "label": scenario["label"],
            "day_type": scenario["day_type"],
            "day_name": scenario["day_name"],
            "time_window": scenario["time_window"],
            "dominant_flow": scenario["dominant_flow"],
            "total_vehicle_count": scenario["total_vehicle_count"],
            "seed": scenario.get("seed"),
        },
        "entries": entries,
    }

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")

    _write_csv(csv_path, entries)
    return ODMatrixResult(scenario_id, entries, json_path, csv_path)


def _area_weight(
    area_weights: dict[str, dict[str, Any]], area_id: str, field_name: str
) -> float:
    return float(area_weights.get(area_id, {}).get(field_name, 1.0))


def _pair_calibration(
    area_weights: dict[str, dict[str, Any]], origin_id: str, destination_id: str
) -> dict[str, float]:
    origin = area_weights.get(origin_id, {})
    destination = area_weights.get(destination_id, {})
    return {
        "congestion_weight": _mean_weight(origin, destination, "congestion_weight"),
        "route_pressure": _mean_weight(origin, destination, "route_pressure"),
        "corridor_density": _mean_weight(origin, destination, "corridor_density"),
        "peak_hour_intensity": _mean_weight(origin, destination, "peak_hour_intensity"),
    }


def _mean_weight(
    origin: dict[str, Any], destination: dict[str, Any], field_name: str
) -> float:
    return (float(origin.get(field_name, 1.0)) + float(destination.get(field_name, 1.0))) / 2.0


def _domain_multiplier(
    scenario: dict[str, Any], origin: dict[str, Any], destination: dict[str, Any]
) -> float:
    origin_domain = origin.get("parent_domain", "unknown")
    destination_domain = destination.get("parent_domain", "unknown")
    if origin_domain == destination_domain:
        return float(scenario.get("same_parent_domain_multiplier", 1.0))

    flow_key = f"{origin_domain}->{destination_domain}"
    flow_bias = scenario.get("parent_domain_flow_bias", {})
    return float(flow_bias.get(flow_key, scenario.get("cross_parent_domain_multiplier", 1.0)))


def _corridor_multiplier(
    scenario: dict[str, Any], origin: dict[str, Any], destination: dict[str, Any]
) -> float:
    corridor_preference = scenario.get("corridor_preference", {})
    origin_corridors = origin.get("corridors", origin.get("major_corridors", []))
    destination_corridors = destination.get("corridors", destination.get("major_corridors", []))
    shared_corridors = set(origin_corridors) & set(destination_corridors)
    if not shared_corridors:
        return 1.0
    return max(float(corridor_preference.get(corridor, 1.0)) for corridor in shared_corridors)


def _resolve_area_map(
    region_config: dict[str, Any],
    scenario: dict[str, Any],
    maps_catalog: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    area_source = scenario.get("area_source", "region_decomposition")
    if area_source == "maps_area_catalog":
        if not maps_catalog:
            raise ValueError(
                "Scenario requests maps_area_catalog, but configs/preprocessing/maps_area_catalog.json is missing."
            )
        return maps_catalog["areas"]
    return region_config["local_regions"]


def _jitter_multiplier(scenario: dict[str, Any], random_source: random.Random) -> float:
    jitter = scenario.get("stochastic_jitter", {})
    if not jitter.get("enabled", False):
        return 1.0
    return random_source.uniform(
        float(jitter.get("min_multiplier", 1.0)),
        float(jitter.get("max_multiplier", 1.0)),
    )


def _distribute_rounding_residue(entries: list[dict[str, Any]], total_vehicle_count: int) -> None:
    assigned = sum(entry["expected_vehicle_count"] for entry in entries)
    residue = total_vehicle_count - assigned
    if residue <= 0:
        return

    ranked_entries = sorted(entries, key=lambda entry: entry["probability"], reverse=True)
    for index in range(residue):
        ranked_entries[index % len(ranked_entries)]["expected_vehicle_count"] += 1


def _write_csv(csv_path: Path, entries: list[dict[str, Any]]) -> None:
    fieldnames = [
        "scenario_id",
        "origin_area_id",
        "destination_area_id",
        "origin_taz_id",
        "destination_taz_id",
        "probability",
        "expected_vehicle_count",
        "time_window",
        "day_type",
        "day_name",
        "dominant_flow",
        "congestion_weight",
        "route_pressure",
        "corridor_density",
        "peak_hour_intensity",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            row = {
                key: entry[key]
                for key in fieldnames
                if key in entry
            }
            row.update(entry["calibration_features"])
            writer.writerow(row)


if __name__ == "__main__":
    result = generate_and_write_area_od_matrix(Path.cwd())
    print(f"Wrote {len(result.entries)} OD entries")
    print(result.json_path)
    print(result.csv_path)
