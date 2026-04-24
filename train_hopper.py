import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

# SEED = 0
# SEED = 1
SEED = 2
TOTAL_TIMESTEPS = 200_000
# LOG_DIR = "results/hopper_seed_0"
# MODEL_PATH = os.path.join("models", "ppo_hopper_seed_0")
# LOG_DIR = "results/hopper_seed_1"
# MODEL_PATH = os.path.join("models", "ppo_hopper_seed_1")
LOG_DIR = "results/hopper_seed_2"
MODEL_PATH = os.path.join("models", "ppo_hopper_seed_2")

# make folders if they do not exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

# create Hopper environment
env = gym.make("Hopper-v5")

# Monitor records episode rewards/lengths into files
env = Monitor(env, LOG_DIR)

# set random seed
env.reset(seed=SEED)

# create PPO model using a default MLP policy
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    seed=SEED,
    tensorboard_log=LOG_DIR
)

# train PPO
model.learn(total_timesteps=TOTAL_TIMESTEPS)

# save trained model
model.save(MODEL_PATH)

env.close()
print(f"Saved model to {MODEL_PATH}")