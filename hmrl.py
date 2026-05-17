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
        default="ppo",
        choices=["ppo", "a2c", "dqn"],
        help="RL algorithm used for ward agents",
    )
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=3600,
        help="Maximum simulation duration in seconds",
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

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    result = run_simulation(
        scope=args.scope,
        identifier=args.id,
        project_root=PROJECT_ROOT,
        gui=args.gui,
        scenario_id=args.scenario,
        max_ticks=args.max_ticks,
        algorithm=args.algorithm,
    )

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
