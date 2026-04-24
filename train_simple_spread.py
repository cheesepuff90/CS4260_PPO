import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam
from mpe2 import simple_spread_v3

from utils import set_seed, ensure_dir, concat_agent_obs, compute_returns_and_advantages
from networks import DecentralizedCritic, SharedActor, CentralizedCritic
from buffer import MultiAgentBuffer


# =========================
# Config
# =========================
SEED = 2
N_AGENTS = 3
MAX_CYCLES = 25
TOTAL_EPISODES = 500
ROLLOUT_EPISODES = 10
PPO_EPOCHS = 4
CLIP_EPS = 0.2
GAMMA = 0.99
LAMBDA = 0.95
LR = 3e-4
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5

DEVICE = torch.device("cpu")

# RESULTS_DIR = "results/simple_spread"
# MODELS_DIR = "models"

RESULTS_DIR = "results_count/simple_spread"
MODELS_DIR = "models_count"

ensure_dir(RESULTS_DIR)
ensure_dir(MODELS_DIR)

set_seed(SEED)


# =========================
# Env setup
# =========================
env = simple_spread_v3.parallel_env(
    N=N_AGENTS,
    local_ratio=0.5,
    max_cycles=MAX_CYCLES,
    render_mode=None
)

obs, infos = env.reset(seed=SEED)
agents = env.agents

obs_dim = len(obs[agents[0]])
action_dim = env.action_space(agents[0]).n
joint_obs_dim = obs_dim * N_AGENTS

print("Agents:", agents)
print("obs_dim:", obs_dim)
print("action_dim:", action_dim)
print("joint_obs_dim:", joint_obs_dim)


# =========================
# Networks + optimizers
# =========================
actor = SharedActor(obs_dim, action_dim).to(DEVICE)

# # mappo
# critic = CentralizedCritic(joint_obs_dim).to(DEVICE)

# baseline
critic = DecentralizedCritic(obs_dim).to(DEVICE)

actor_optimizer = Adam(actor.parameters(), lr=LR)
critic_optimizer = Adam(critic.parameters(), lr=LR)


def collect_rollout():
    buffer = MultiAgentBuffer()
    episode_returns = []
    episode_lengths = []
    
    # for collision count
    episode_collisions = []

    for _ in range(ROLLOUT_EPISODES):
        obs, infos = env.reset()
        done = False
        ep_return = 0.0
        ep_len = 0
        
        # for collision count
        ep_collisions = 0

        while not done:
            joint_obs = concat_agent_obs(obs, agents)
            
            # baseline
            first_agent_obs = obs[agents[0]]
            obs_tensor = torch.tensor(first_agent_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            
            value = critic(obs_tensor).item()
            
            # # mappo
            # joint_obs_tensor = torch.tensor(joint_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)

            # value = critic(joint_obs_tensor).item()

            actions_dict = {}
            agent_obs_list = []
            action_list = []
            log_prob_list = []

            for agent in agents:
                agent_obs = obs[agent]
                agent_obs_list.append(agent_obs)

                obs_tensor = torch.tensor(agent_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                logits = actor(obs_tensor)
                dist = Categorical(logits=logits)

                action = dist.sample()
                log_prob = dist.log_prob(action)

                actions_dict[agent] = int(action.item())
                action_list.append(int(action.item()))
                log_prob_list.append(float(log_prob.item()))

            next_obs, rewards, terminations, truncations, infos = env.step(actions_dict)

            team_reward = float(np.mean([rewards[a] for a in agents]))
            
            # for collision count
            step_collisions = sum(1 for a in agents if rewards[a] < -0.5)
            
            done = all(terminations[a] or truncations[a] for a in agents)

            buffer.store(
                agent_obs=agent_obs_list,
                joint_obs=joint_obs,
                actions=action_list,
                log_probs=log_prob_list,
                reward=team_reward,
                done=done,
                value=value
            )

            obs = next_obs
            ep_return += team_reward
            ep_len += 1
            
            # for collision count
            ep_collisions += step_collisions

        episode_returns.append(ep_return)
        episode_lengths.append(ep_len)
        
        # for collision count
        episode_collisions.append(ep_collisions)

    return buffer, episode_returns, episode_lengths, episode_collisions


def update(buffer):
    data = buffer.get()

    returns, advantages = compute_returns_and_advantages(
        rewards=data["rewards"],
        dones=data["dones"],
        values=data["values"],
        gamma=GAMMA,
        lam=LAMBDA
    )

    returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
    advantages = torch.tensor(advantages, dtype=torch.float32, device=DEVICE)

    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    agent_obs = data["agent_obs"].to(DEVICE)      # [T, n_agents, obs_dim]
    joint_obs = data["joint_obs"].to(DEVICE)      # [T, joint_obs_dim]
    actions = data["actions"].to(DEVICE)          # [T, n_agents]
    old_log_probs = data["log_probs"].to(DEVICE)  # [T, n_agents]

    T = agent_obs.shape[0]

    for _ in range(PPO_EPOCHS):
        # flatten across time and agents for actor
        flat_obs = agent_obs.reshape(T * N_AGENTS, obs_dim)
        flat_actions = actions.reshape(T * N_AGENTS)
        flat_old_log_probs = old_log_probs.reshape(T * N_AGENTS)

        logits = actor(flat_obs)
        dist = Categorical(logits=logits)
        new_log_probs = dist.log_prob(flat_actions)
        entropy = dist.entropy().mean()

        # repeat each timestep advantage once per agent
        flat_advantages = advantages.repeat_interleave(N_AGENTS)

        ratios = torch.exp(new_log_probs - flat_old_log_probs)
        surr1 = ratios * flat_advantages
        surr2 = torch.clamp(ratios, 1 - CLIP_EPS, 1 + CLIP_EPS) * flat_advantages
        actor_loss = -torch.min(surr1, surr2).mean() - ENTROPY_COEF * entropy

        # # mappo
        # values_pred = critic(joint_obs).squeeze(-1)
        
        #baseline
        first_agent_obs = agent_obs[:, 0, :]   # [T, obs_dim]
        values_pred = critic(first_agent_obs).squeeze(-1)
        
        critic_loss = F.mse_loss(values_pred, returns)

        loss = actor_loss + VALUE_COEF * critic_loss

        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()
        loss.backward()
        actor_optimizer.step()
        critic_optimizer.step()

    return actor_loss.item(), critic_loss.item()


# =========================
# Training loop
# =========================
all_returns = []
all_collisions = []  # for collision count

for episode_block in range(0, TOTAL_EPISODES, ROLLOUT_EPISODES):
    buffer, ep_returns, ep_lengths, ep_collisions = collect_rollout()
    actor_loss, critic_loss = update(buffer)

    mean_return = float(np.mean(ep_returns))
    mean_length = float(np.mean(ep_lengths))
    
    # for collision count
    mean_collisions = float(np.mean(ep_collisions))
    
    all_returns.append(mean_return)
    
    all_collisions.append(mean_collisions)  # for collision count

    # print(
    #     f"Episodes {episode_block + 1}-{episode_block + ROLLOUT_EPISODES} | "
    #     f"mean_return={mean_return:.3f} | "
    #     f"mean_length={mean_length:.2f} | "
    #     f"actor_loss={actor_loss:.4f} | "
    #     f"critic_loss={critic_loss:.4f}"
    # )
    
    # for collision count
    print(
        f"Episodes {episode_block + 1}-{episode_block + ROLLOUT_EPISODES} | "
        f"mean_return={mean_return:.3f} | "
        f"mean_collisions={mean_collisions:.2f} | "
        f"actor_loss={actor_loss:.4f} | "
        f"critic_loss={critic_loss:.4f}"
    )

torch.save(actor.state_dict(), os.path.join(MODELS_DIR, "baseline_actor_2.pt"))
torch.save(critic.state_dict(), os.path.join(MODELS_DIR, "baseline_critic_2.pt"))
np.save(os.path.join(RESULTS_DIR, "baseline_returns_2.npy"), np.array(all_returns))
np.save(os.path.join(RESULTS_DIR, "baseline_collisions_2.npy"), np.array(all_collisions))


env.close()
print("Training complete.")