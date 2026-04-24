import os
import random
import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def concat_agent_obs(obs_dict, agent_order):
    return np.concatenate([obs_dict[a] for a in agent_order], axis=0)


def compute_returns_and_advantages(rewards, dones, values, gamma=0.99, lam=0.95):
    """
    GAE-Lambda advantage computation.
    Inputs are lists of scalars, one per timestep.
    Returns:
        returns: np.array shape [T]
        advantages: np.array shape [T]
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    returns = np.zeros(T, dtype=np.float32)

    last_gae = 0.0
    next_value = 0.0

    for t in reversed(range(T)):
        mask = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_value * mask - values[t]
        last_gae = delta + gamma * lam * mask * last_gae
        advantages[t] = last_gae
        returns[t] = advantages[t] + values[t]
        next_value = values[t]

    return returns, advantages