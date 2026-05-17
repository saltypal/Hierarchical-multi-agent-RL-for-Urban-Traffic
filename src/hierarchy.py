from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class WardGroup:
    region_id: str
    group_id: str
    group_type: str
    wards: list[int]


def load_hierarchy(project_root: Path) -> dict[str, Any]:
    path = project_root / "configs" / "hierarchy" / "blr_regions.json"
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def list_regions(hierarchy: dict[str, Any]) -> list[str]:
    return sorted(hierarchy["blr_regions"].keys())


def list_groups(hierarchy: dict[str, Any], region_id: str) -> list[str]:
    region = hierarchy["blr_regions"][region_id]
    return sorted(region.keys())


def get_group(hierarchy: dict[str, Any], region_id: str, group_id: str) -> WardGroup:
    region = hierarchy["blr_regions"][region_id]
    group = region[group_id]
    wards = [int(ward) for ward in group["wards"]]
    return WardGroup(region_id=region_id, group_id=group_id, group_type=group["type"], wards=wards)


def iter_all_groups(hierarchy: dict[str, Any]) -> Iterable[WardGroup]:
    for region_id, groups in hierarchy["blr_regions"].items():
        for group_id in groups:
            yield get_group(hierarchy, region_id, group_id)


def ward_folder_name(ward_number: int) -> str:
    return f"ward_{ward_number:03d}"

