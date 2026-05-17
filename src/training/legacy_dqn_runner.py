"""Legacy hierarchical DQN training loop extracted from main.py."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import sys

import traci


def run_legacy_hierarchical_dqn(
    sumo_config: str,
    episodes: int = 50,
    max_steps: int = 300,
    gui: bool = True,
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from Traffic.agents.master_agent import MasterAgent
    from Traffic.agents.rsu_agent import RSUAgent
    from Traffic.env.traffic_env import TrafficEnv

    env = TrafficEnv(sumo_config)
    env.start(gui=gui)

    all_junctions = traci.junction.getIDList()
    rsu_junctions = [junction for junction in all_junctions if not junction.startswith(":")]
    preferred = ["B0", "A1", "B1", "C1", "B2"]
    selected = [junction for junction in preferred if junction in rsu_junctions]
    if not selected:
        selected = rsu_junctions[:5]

    rsu_agents = {
        junction: RSUAgent(state_dim=3, action_dim=3)
        for junction in selected
    }
    master = MasterAgent(state_dim=3, action_dim=3)

    episode_rewards: list[float] = []

    for _ in range(episodes):
        master_state = env.reset()
        done = False
        total_reward = 0.0
        step = 0

        while not done and step < max_steps:
            master_action = master.act(env.get_master_state())

            rsu_states = {}
            rsu_actions = {}
            for junction, agent in rsu_agents.items():
                state = env.get_rsu_state(junction)
                action = agent.act(state)
                rsu_states[junction] = state
                rsu_actions[junction] = action
                env.apply_rsu_action(junction, action, master_action)

            traci.simulationStep()

            master_reward = env.compute_global_reward()
            next_master_state = env.get_master_state()
            master.remember((master_state, master_action, master_reward, next_master_state, done))
            if step % 5 == 0:
                master.train()

            for junction, agent in rsu_agents.items():
                if junction not in rsu_states:
                    continue
                next_state = env.get_rsu_state(junction)
                reward = env.compute_local_reward(junction)
                agent.remember((rsu_states[junction], rsu_actions[junction], reward, next_state, done))
                if step % 5 == 0:
                    agent.train()

            master_state = next_master_state
            total_reward += float(master_reward)
            done = traci.simulation.getMinExpectedNumber() == 0
            step += 1

        episode_rewards.append(total_reward)

    env.close()

    avg_reward = (sum(episode_rewards) / len(episode_rewards)) if episode_rewards else 0.0
    return {
        "episodes": episodes,
        "max_steps": max_steps,
        "junctions": selected,
        "average_reward": avg_reward,
        "best_reward": max(episode_rewards) if episode_rewards else 0.0,
    }
