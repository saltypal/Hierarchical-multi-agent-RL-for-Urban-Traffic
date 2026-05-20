"""Shared evaluation runner over reusable runtime code."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.runtime import run_simulation

from .metrics import normalize_run_metrics


FIXED_WARD_IDS = [
    "ward_070", "ward_071", "ward_072",
    "ward_017", "ward_018", "ward_019", "ward_020",
    "ward_007", "ward_008", "ward_009", "ward_010",
    "ward_011", "ward_012", "ward_013", "ward_014", "ward_015",
]


@dataclass(frozen=True)
class EvaluationCase:
    scenario_id: str
    ward_id: str
    max_ticks: int = 1800
    gui: bool = False
    use_rl: bool = True
    use_area: bool = True
    use_city: bool = True
    algorithm: str = "dqn"


def load_evaluation_wards(project_root: Path) -> list[str]:
    """Load and validate the fixed 16-ward evaluation set."""
    registry_path = project_root / "configs" / "hierarchy" / "ward_registry.json"
    with registry_path.open("r", encoding="utf-8") as fh:
        registry = json.load(fh)
    known_wards = set(registry.get("wards", {}))
    missing = [ward_id for ward_id in FIXED_WARD_IDS if ward_id not in known_wards]
    if missing:
        raise ValueError(f"Missing evaluation wards in registry: {missing}")
    return list(FIXED_WARD_IDS)


def run_case(
    project_root: Path,
    case: EvaluationCase,
    *,
    collect_tick_records: bool = False,
) -> dict[str, Any]:
    """Run one evaluation case and normalize its metric row."""
    result = run_simulation(
        scope="ward",
        identifier=case.ward_id,
        project_root=project_root,
        gui=case.gui,
        scenario_id=case.scenario_id,
        max_ticks=case.max_ticks,
        algorithm=case.algorithm,
        dashboard=False,
        use_rl=case.use_rl,
        use_area=case.use_area,
        use_city=case.use_city,
        collect_tick_records=collect_tick_records,
        persist_results=False,
    )
    normalized = normalize_run_metrics(result)
    normalized.update({
        "scenario_id": case.scenario_id,
        "ward_id": case.ward_id,
        "scope": "ward",
        "max_ticks": case.max_ticks,
        "gui": case.gui,
        "use_rl": case.use_rl,
        "use_area": case.use_area,
        "use_city": case.use_city,
        "algorithm": case.algorithm,
    })
    result["normalized_metrics"] = normalized
    return result
