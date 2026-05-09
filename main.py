from env.traffic_env import TrafficEnv
from agents.rsu_agent import RSUAgent
from agents.master_agent import MasterAgent
import traci

# 🔹 Initialize environment
env = TrafficEnv("sumo/config.sumocfg")

# 🔹 Start SUMO first (needed before accessing junctions)
env.start()

# 🔹 Get valid junctions (exclude internal ones like ":A0_0")
all_junctions = traci.junction.getIDList()
rsu_junctions = [j for j in all_junctions if not j.startswith(":")]

# 🔹 Pick important junctions (center + midpoints)
# Adjust if needed after printing
print("Available Junctions:", rsu_junctions)

# Example selection (only if they exist)
preferred = ["B0", "A1", "B1", "C1", "B2"]
rsu_junctions = [j for j in preferred if j in rsu_junctions]

print("Using RSU Junctions:", rsu_junctions)

# 🔹 Initialize RSU agents
rsu_agents = {
    j: RSUAgent(state_dim=3, action_dim=3)
    for j in rsu_junctions
}

# 🔹 Initialize Master agent
master = MasterAgent(state_dim=3, action_dim=3)

# 🔹 Training parameters
episodes = 50
max_steps = 300

# 🔹 Reward tracking
episode_rewards = []

# ================= TRAINING LOOP =================
for ep in range(episodes):
    print(f"\n===== EPISODE {ep} =====")

    state_master = env.reset()
    done = False

    total_reward = 0
    step = 0

    while not done and step < max_steps:

        # 🔴 MASTER ACTION
        master_state = env.get_master_state()
        master_action = master.act(master_state)

        # 🔵 RSU ACTIONS
        rsu_states = {}
        rsu_actions = {}

        for j, agent in rsu_agents.items():
            try:
                state = env.get_rsu_state(j)
                action = agent.act(state)

                rsu_states[j] = state
                rsu_actions[j] = action

                env.apply_rsu_action(j, action, master_action)

            except Exception as e:
                print(f"Error at junction {j}: {e}")
                continue

        # ▶ Advance simulation
        traci.simulationStep()

        # 🎯 MASTER REWARD
        master_reward = env.compute_global_reward()
        next_master_state = env.get_master_state()

        # 🔴 Train MASTER
        master.remember((master_state, master_action, master_reward, next_master_state, done))
        if step % 5 == 0:
            master.train()

        # 🔵 Train RSUs
        for j, agent in rsu_agents.items():
            if j not in rsu_states:
                continue  # ✅ prevents KeyError

            try:
                next_state = env.get_rsu_state(j)
                reward = env.compute_local_reward(j)

                agent.remember((rsu_states[j], rsu_actions[j], reward, next_state, done))

                if step % 5 == 0:
                    agent.train()

            except Exception as e:
                print(f"Training error at {j}: {e}")
                continue

        # 🔄 Update
        master_state = next_master_state
        total_reward += master_reward

        done = traci.simulation.getMinExpectedNumber() == 0
        step += 1

    # 🔥 Store reward
    episode_rewards.append(total_reward)

    # ✅ Clean output
    print(f"Episode {ep} | Reward: {total_reward:.2f} | Steps: {step}")

# 🔚 Close SUMO
env.close()

# ================= FINAL SUMMARY =================
print("\n===== TRAINING COMPLETE =====")

avg_reward = sum(episode_rewards) / len(episode_rewards)
best_reward = max(episode_rewards)

print(f"Average Reward: {avg_reward:.2f}")
print(f"Best Reward: {best_reward:.2f}")