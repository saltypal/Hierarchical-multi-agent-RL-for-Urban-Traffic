"""Backward-compatible legacy DQN entrypoint."""

from __future__ import annotations

import json

from src.training import run_legacy_hierarchical_dqn


def main() -> None:
    summary = run_legacy_hierarchical_dqn(
        sumo_config="Traffic/sumo/config.sumocfg",
        episodes=50,
        max_steps=300,
        gui=True,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
