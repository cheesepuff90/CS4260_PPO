import numpy as np
import torch


class MultiAgentBuffer:
    def __init__(self):
        self.agent_obs = []      # shape per step: [n_agents, obs_dim]
        self.joint_obs = []      # shape per step: [joint_obs_dim]
        self.actions = []        # shape per step: [n_agents]
        self.log_probs = []      # shape per step: [n_agents]
        self.rewards = []        # scalar team reward per step
        self.dones = []          # scalar done per step
        self.values = []         # scalar critic value per step

    def store(self, agent_obs, joint_obs, actions, log_probs, reward, done, value):
        self.agent_obs.append(np.array(agent_obs, dtype=np.float32))
        self.joint_obs.append(np.array(joint_obs, dtype=np.float32))
        self.actions.append(np.array(actions, dtype=np.int64))
        self.log_probs.append(np.array(log_probs, dtype=np.float32))
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.values.append(float(value))

    def get(self):
        data = {
            "agent_obs": torch.tensor(np.array(self.agent_obs), dtype=torch.float32),   # [T, n_agents, obs_dim]
            "joint_obs": torch.tensor(np.array(self.joint_obs), dtype=torch.float32),   # [T, joint_obs_dim]
            "actions": torch.tensor(np.array(self.actions), dtype=torch.long),          # [T, n_agents]
            "log_probs": torch.tensor(np.array(self.log_probs), dtype=torch.float32),   # [T, n_agents]
            "rewards": np.array(self.rewards, dtype=np.float32),                        # [T]
            "dones": np.array(self.dones, dtype=np.float32),                            # [T]
            "values": np.array(self.values, dtype=np.float32),                          # [T]
        }
        return data

    def clear(self):
        self.agent_obs.clear()
        self.joint_obs.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()