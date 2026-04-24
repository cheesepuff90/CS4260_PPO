import os
import numpy as np
import torch
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from mpe2 import simple_spread_v3
from networks import SharedActor

DEVICE = torch.device("cpu")

# -------------------------
# Config
# -------------------------
METHOD = "baseline"   # "baseline" or "mappo"
SEED = 0
MODEL_DIR = "models_count"
OUT_DIR = "trajectory_plots"
MAX_CYCLES = 100
USE_SAMPLED_ACTIONS = False   # False = argmax, True = sample
SAVE_GIF = True

os.makedirs(OUT_DIR, exist_ok=True)

ACTOR_PATH = os.path.join(MODEL_DIR, f"{METHOD}_actor_{SEED}.pt")
OUT_PATH = os.path.join(OUT_DIR, f"{METHOD}_trajectory_seed_{SEED}.gif")

# -------------------------
# Environment
# -------------------------
env = simple_spread_v3.parallel_env(
    N=3,
    local_ratio=0.5,
    max_cycles=MAX_CYCLES,
    render_mode=None
)

obs, infos = env.reset(seed=SEED)
agents = env.agents

obs_dim = len(obs[agents[0]])
action_dim = env.action_space(agents[0]).n

# -------------------------
# Load actor
# -------------------------
actor = SharedActor(obs_dim, action_dim).to(DEVICE)
actor.load_state_dict(torch.load(ACTOR_PATH, map_location=DEVICE))
actor.eval()

# -------------------------
# Storage for trajectories
# -------------------------
agent_positions = {agent: [] for agent in agents}
landmark_positions = []

def get_world():
    # Access underlying MPE world
    return env.unwrapped.world

def record_positions():
    world = get_world()
    for i, agent_name in enumerate(agents):
        pos = world.agents[i].state.p_pos.copy()
        agent_positions[agent_name].append(pos)

    # landmarks fixed, but record anyway for safety
    lm_pos = [lm.state.p_pos.copy() for lm in world.landmarks]
    landmark_positions.append(lm_pos)

# initial positions
record_positions()

done = False

while not done:
    actions_dict = {}

    with torch.no_grad():
        for agent in agents:
            agent_obs = obs[agent]
            obs_tensor = torch.tensor(agent_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)

            logits = actor(obs_tensor)

            if USE_SAMPLED_ACTIONS:
                dist = Categorical(logits=logits)
                action = dist.sample()
            else:
                # action = torch.argmax(logits, dim=-1)
                dist = Categorical(logits=logits)
                action = dist.sample()

            actions_dict[agent] = int(action.item())

    next_obs, rewards, terminations, truncations, infos = env.step(actions_dict)
    obs = next_obs

    record_positions()

    done = all(terminations[a] or truncations[a] for a in agents)

env.close()

# convert to arrays
for agent in agents:
    agent_positions[agent] = np.array(agent_positions[agent])   # [T, 2]
landmark_positions = np.array(landmark_positions)               # [T, n_landmarks, 2]

T = len(agent_positions[agents[0]])

# -------------------------
# Plot setup
# -------------------------
fig, ax = plt.subplots(figsize=(6, 6))
colors = ["tab:blue", "tab:orange", "tab:green"]

# static landmark positions: use first frame
initial_landmarks = landmark_positions[0]
ax.scatter(
    initial_landmarks[:, 0],
    initial_landmarks[:, 1],
    c="black",
    s=120,
    marker="x",
    label="Landmarks"
)

# prepare line and point objects
lines = {}
points = {}

for agent, color in zip(agents, colors):
    (line,) = ax.plot([], [], color=color, linewidth=2, label=agent)
    (point,) = ax.plot([], [], "o", color=color, markersize=8)
    lines[agent] = line
    points[agent] = point

ax.set_title(f"{METHOD.upper()} Trajectories (seed {SEED})")
ax.set_xlabel("x position")
ax.set_ylabel("y position")
ax.legend(loc="upper right")
ax.grid(True)

# auto bounds from all trajectories
all_xy = np.concatenate([agent_positions[a] for a in agents], axis=0)
xmin, ymin = all_xy.min(axis=0) - 0.2
xmax, ymax = all_xy.max(axis=0) + 0.2
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_aspect("equal", adjustable="box")


def init():
    for agent in agents:
        lines[agent].set_data([], [])
        points[agent].set_data([], [])
    return list(lines.values()) + list(points.values())


def update(frame):
    for agent in agents:
        traj = agent_positions[agent][: frame + 1]
        lines[agent].set_data(traj[:, 0], traj[:, 1])
        points[agent].set_data([traj[-1, 0]], [traj[-1, 1]])
    ax.set_title(f"{METHOD.upper()} Trajectories (seed {SEED}) | step {frame}")
    return list(lines.values()) + list(points.values())


anim = FuncAnimation(
    fig,
    update,
    frames=range(0, T, 2),
    init_func=init,
    interval=30,
    blit=True
)

if SAVE_GIF:
    anim.save(OUT_PATH, writer=PillowWriter(fps=10))
    print(f"Saved trajectory animation to {OUT_PATH}")
