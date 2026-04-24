import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

MODEL_PATH = "models/ppo_hopper_seed_0"
# MODEL_PATH = "models/ppo_hopper_seed_1"
# MODEL_PATH = "models/ppo_hopper_seed_2"
N_EPISODES = 10

env = gym.make("Hopper-v5")
model = PPO.load(MODEL_PATH)

returns = []

for _ in range(N_EPISODES):
    obs, info = env.reset()
    done = False
    truncated = False
    episode_return = 0.0

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        episode_return += reward

    returns.append(episode_return)

env.close()

print("Average return:", np.mean(returns))
print("Std return:", np.std(returns))