"""Master evaluation entrypoint — backward-compatible wrapper.

For the new ablation-based evaluation framework, use ``evaluate.py`` directly.
This module is kept for programmatic usage and backward compatibility.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from configs.scenarios import list_scenarios

from src.evaluation.common.logger import ensure_output_dirs, write_csv, write_json, write_text_report
from src.evaluation.common.metrics import aggregate_full_sweep
from src.evaluation.common.runner import EvaluationCase, load_evaluation_wards, run_case
from src.evaluation.ward_eval import build_ward_summary


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full evaluation sweep for the HMRL traffic stack")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--max-ticks", type=int, default=900)
    parser.add_argument("--algorithm", choices=["ppo", "dqn"], default="ppo")
    parser.add_argument("--write-ticks", action="store_true")
    parser.add_argument("--disable-rl", action="store_true")
    parser.add_argument("--disable-area", action="store_true")
    parser.add_argument("--disable-city", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    project_root = Path(args.project_root).resolve()
    scenarios = list_scenarios()
    wards = load_evaluation_wards(project_root)
    output_dirs = ensure_output_dirs(project_root / "src" / "evaluation" / "outputs")

    LOGGER.info("Starting evaluation sweep: %d scenarios x %d wards", len(scenarios), len(wards))

    results: list[dict[str, Any]] = []
    normalized_rows: list[dict[str, Any]] = []
    for scenario_id in scenarios:
        for ward_id in wards:
            case = EvaluationCase(
                scenario_id=scenario_id,
                ward_id=ward_id,
                max_ticks=args.max_ticks,
                gui=False,
                use_rl=not args.disable_rl,
                use_area=not args.disable_area,
                use_city=not args.disable_city,
                algorithm=args.algorithm,
            )
            LOGGER.info("Running %s / %s", scenario_id, ward_id)
            result = run_case(
                project_root,
                case,
                collect_tick_records=True,
            )
            results.append(result)
            normalized_rows.append(result["normalized_metrics"])

    aggregate = aggregate_full_sweep(normalized_rows)
    ward_summary = build_ward_summary(normalized_rows)

    write_csv(output_dirs["csv"] / "full_run_summary.csv", normalized_rows)
    write_csv(output_dirs["csv"] / "by_ward_summary.csv", ward_summary)
    write_csv(output_dirs["csv"] / "by_scenario_summary.csv", aggregate["by_scenario"])

    if args.write_ticks:
        tick_rows: list[dict[str, Any]] = []
        for result in results:
            tick_rows.extend(result.get("tick_records", []))
        write_csv(output_dirs["csv"] / "tick_records.csv", tick_rows)

    manifest = {
        "scenarios": scenarios,
        "wards": wards,
        "max_ticks": args.max_ticks,
        "algorithm": args.algorithm,
        "use_rl": not args.disable_rl,
        "use_area": not args.disable_area,
        "use_city": not args.disable_city,
        "total_runs": len(normalized_rows),
    }
    write_json(output_dirs["reports"] / "run_manifest.json", manifest)
    write_json(output_dirs["reports"] / "aggregate_summary.json", aggregate)

    LOGGER.info("Evaluation complete. Outputs written to %s", output_dirs["base"])


if __name__ == "__main__":
    main()
