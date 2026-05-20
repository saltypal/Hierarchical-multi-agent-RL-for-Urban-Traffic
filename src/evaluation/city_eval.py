"""Master evaluation entrypoint for the full 10-scenario x 16-ward sweep."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from configs.scenarios import list_scenarios

from src.evaluation.area_eval import build_area_prediction_summary
from src.evaluation.common.logger import ensure_output_dirs, write_csv, write_json, write_text_report
from src.evaluation.common.metrics import aggregate_full_sweep
from src.evaluation.common.plotter import plot_scenario_series
from src.evaluation.common.runner import EvaluationCase, load_evaluation_wards, run_case
from src.evaluation.dataset_builder import export_temporal_dataset
from src.evaluation.ward_eval import build_ward_summary


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full evaluation sweep for the HMRL traffic stack")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--max-ticks", type=int, default=1800)
    parser.add_argument("--algorithm", choices=["ppo", "a2c", "dqn"], default="dqn")
    parser.add_argument("--write-ticks", action="store_true")
    parser.add_argument("--export-dataset", action="store_true")
    parser.add_argument("--disable-rl", action="store_true")
    parser.add_argument("--disable-area", action="store_true")
    parser.add_argument("--disable-city", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _build_report_lines(
    aggregate: dict[str, Any],
    scenarios: list[str],
    wards: list[str],
    args: argparse.Namespace,
) -> list[str]:
    overall = aggregate.get("overall", {})
    return [
        "HMRL Full Evaluation Report",
        "",
        f"Scenarios: {len(scenarios)}",
        f"Wards: {len(wards)}",
        f"Ticks per run: {args.max_ticks}",
        f"GUI: False",
        f"Algorithm: {args.algorithm}",
        f"Controllers: RL={not args.disable_rl}, Area={not args.disable_area}, City={not args.disable_city}",
        f"Total runs: {aggregate.get('total_runs', 0)}",
        "",
        "Overall Means",
        f"avg_speed: {overall.get('avg_speed', 0.0):.4f}",
        f"congestion_score: {overall.get('congestion_score', 0.0):.4f}",
        f"queue_length: {overall.get('queue_length', 0.0):.4f}",
        f"throughput: {overall.get('throughput', 0.0):.4f}",
        f"trip_completion: {overall.get('trip_completion', 0.0):.4f}",
        f"travel_time: {overall.get('travel_time', 0.0):.4f}",
        f"waiting_time: {overall.get('waiting_time', 0.0):.4f}",
        f"ambulance_delay: {overall.get('ambulance_delay', 0.0):.4f}",
        f"incident_delay: {overall.get('incident_delay', 0.0):.4f}",
        f"reroute_count: {overall.get('reroute_count', 0.0):.4f}",
    ]


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
    area_summary = build_area_prediction_summary(results)

    write_csv(output_dirs["csv"] / "full_run_summary.csv", normalized_rows)
    write_csv(output_dirs["csv"] / "by_ward_summary.csv", ward_summary)
    write_csv(output_dirs["csv"] / "by_scenario_summary.csv", aggregate["by_scenario"])
    write_csv(output_dirs["csv"] / "area_prediction_summary.csv", area_summary)

    if args.write_ticks:
        tick_rows: list[dict[str, Any]] = []
        ward_tick_rows: list[dict[str, Any]] = []
        for result in results:
            tick_rows.extend(result.get("tick_records", []))
            ward_tick_rows.extend(result.get("ward_tick_records", []))
        write_csv(output_dirs["csv"] / "tick_records.csv", tick_rows)
        write_csv(output_dirs["csv"] / "ward_tick_records.csv", ward_tick_rows)

    plot_paths = plot_scenario_series(results, output_dirs["plots"])

    dataset_paths = None
    if args.export_dataset:
        dataset_paths = export_temporal_dataset(results, output_dirs["reports"])

    manifest = {
        "scenarios": scenarios,
        "wards": wards,
        "max_ticks": args.max_ticks,
        "gui": False,
        "algorithm": args.algorithm,
        "use_rl": not args.disable_rl,
        "use_area": not args.disable_area,
        "use_city": not args.disable_city,
        "total_runs": len(normalized_rows),
        "plots": [str(path) for path in plot_paths],
        "dataset_paths": dataset_paths,
    }
    write_json(output_dirs["reports"] / "run_manifest.json", manifest)
    write_json(output_dirs["reports"] / "aggregate_summary.json", aggregate)
    write_text_report(
        output_dirs["reports"] / "evaluation_report.txt",
        _build_report_lines(aggregate, scenarios, wards, args),
    )

    LOGGER.info("Evaluation complete. Outputs written to %s", output_dirs["base"])


if __name__ == "__main__":
    main()
