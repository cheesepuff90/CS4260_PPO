import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

paths = [
    "results/hopper_seed_0/monitor.csv",
    "results/hopper_seed_1/monitor.csv",
    "results/hopper_seed_2/monitor.csv",
]

def load_monitor(path):
    # skip first line (comment header)
    df = pd.read_csv(path, skiprows=1)
    return df["r"].values  # episode rewards

def smooth(x, window=10):
    return np.convolve(x, np.ones(window)/window, mode="valid")

curves = []

for p in paths:
    rewards = load_monitor(p)
    rewards = smooth(rewards, window=10)
    curves.append(rewards)

# make equal length
min_len = min(len(c) for c in curves)
curves = np.array([c[:min_len] for c in curves])

mean = curves.mean(axis=0)
std = curves.std(axis=0)

x = np.arange(len(mean))

plt.figure(figsize=(8,5))
plt.plot(x, mean, label="PPO (Hopper)")
plt.fill_between(x, mean-std, mean+std, alpha=0.2)

plt.xlabel("Episodes")
plt.ylabel("Episode Return")
plt.title("Hopper-v5 PPO Performance (3 Seeds)")
plt.legend()
plt.grid()

plt.savefig("hopper_results.png", dpi=300)
plt.show()