"""Operator entrypoint for the HMRL traffic orchestration system.

Supports simulation/inference mode only. Training happens exclusively
in notebooks via ``src/rl/train.py``.

Usage:
    python hmrl.py --scope ward --id ward_001 --gui
    python hmrl.py --scope area --id HSR_Layout --scenario peak_congestion
    python hmrl.py --scope city --gui --max-ticks 1800
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.runtime import run_simulation

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HMRL Hierarchical Traffic Orchestration System",
    )
    parser.add_argument(
        "--scope",
        default="ward",
        choices=["ward", "area", "city"],
        help="Simulation scope: ward, area, or city",
    )
    parser.add_argument(
        "--id",
        default="ward_001",
        help="Target scope identifier (ward ID or area name)",
    )
    parser.add_argument(
        "--scenario",
        default="normal",
        help="Traffic scenario ID (see configs/scenarios.py)",
    )
    parser.add_argument(
        "--algorithm",
        default="dqn",
        choices=["ppo", "a2c", "dqn"],
        help="RL algorithm used for ward agents",
    )
    parser.add_argument(
        "--areas",
        nargs="*",
        default=None,
        help="Optional area IDs to deploy together; defaults to the first two areas for city scope",
    )
    parser.add_argument(
        "--preset",
        choices=["two-area-dqn"],
        default=None,
        help="Shortcut deployment preset for the stitched two-area DQN path",
    )
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=3600,
        help="Maximum simulation duration in seconds",
    )
    parser.add_argument(
        "--disable-rl",
        action="store_true",
        help="Disable ward RL control",
    )
    parser.add_argument(
        "--disable-area",
        action="store_true",
        help="Disable area prediction/control",
    )
    parser.add_argument(
        "--disable-city",
        action="store_true",
        help="Disable city control",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Run SUMO in GUI mode",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    scope = args.scope
    algorithm = args.algorithm
    area_ids = args.areas
    identifier = args.id
    if args.preset == "two-area-dqn":
        scope = "city"
        algorithm = "dqn"
        area_ids = args.areas or None
        identifier = "two-area-dqn"

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    result = run_simulation(
        scope=scope,
        identifier=identifier,
        project_root=PROJECT_ROOT,
        gui=args.gui,
        scenario_id=args.scenario,
        max_ticks=args.max_ticks,
        algorithm=algorithm,
        area_ids=area_ids,
        use_rl=not args.disable_rl,
        use_area=not args.disable_area,
        use_city=not args.disable_city,
    )

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
