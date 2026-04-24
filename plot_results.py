import numpy as np
import matplotlib.pyplot as plt
import os

RESULTS_DIR = "results/simple_spread"

baseline_path = os.path.join(RESULTS_DIR, "baseline_returns.npy")
mappo_path = os.path.join(RESULTS_DIR, "mappo_returns.npy")

baseline = np.load(baseline_path)
mappo = np.load(mappo_path)

# smoothing (moving average)
def smooth(x, window=10):
    return np.convolve(x, np.ones(window)/window, mode='valid')

baseline_smooth = smooth(baseline)
mappo_smooth = smooth(mappo)

plt.figure(figsize=(8,5))

plt.plot(baseline_smooth, label="Baseline (Decentralized Critic)", linewidth=2)
plt.plot(mappo_smooth, label="MAPPO (Centralized Critic)", linewidth=2)

plt.xlabel("Training Iterations (per rollout block)")
plt.ylabel("Mean Episode Return")
plt.title("Simple Spread: Baseline vs MAPPO")
plt.legend()
plt.grid(True)

plt.tight_layout()

save_path = os.path.join(RESULTS_DIR, "baseline_vs_mappo.png")
plt.savefig(save_path, dpi=300)

plt.show()