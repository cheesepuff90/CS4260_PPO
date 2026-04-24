import os
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = "results/simple_spread"

baseline_paths = [
    os.path.join(RESULTS_DIR, "baseline_returns_0.npy"),
    os.path.join(RESULTS_DIR, "baseline_returns_1.npy"),
    os.path.join(RESULTS_DIR, "baseline_returns_2.npy"),
]

mappo_paths = [
    os.path.join(RESULTS_DIR, "mappo_returns_0.npy"),
    os.path.join(RESULTS_DIR, "mappo_returns_1.npy"),
    os.path.join(RESULTS_DIR, "mappo_returns_2.npy"),
]


def smooth(x, window=5):
    return np.convolve(x, np.ones(window) / window, mode="valid")


def load_and_stack(paths, window=5):
    curves = []
    for p in paths:
        arr = np.load(p)
        arr = smooth(arr, window=window)
        curves.append(arr)
    return np.stack(curves, axis=0)


baseline = load_and_stack(baseline_paths, window=5)
mappo = load_and_stack(mappo_paths, window=5)

baseline_mean = baseline.mean(axis=0)
baseline_std = baseline.std(axis=0)

mappo_mean = mappo.mean(axis=0)
mappo_std = mappo.std(axis=0)

x = np.arange(len(baseline_mean))

plt.figure(figsize=(8, 5))

plt.plot(x, baseline_mean, label="Baseline (Decentralized Critic)", linewidth=2)
plt.fill_between(
    x,
    baseline_mean - baseline_std,
    baseline_mean + baseline_std,
    alpha=0.2
)

plt.plot(x, mappo_mean, label="MAPPO (Centralized Critic)", linewidth=2)
plt.fill_between(
    x,
    mappo_mean - mappo_std,
    mappo_mean + mappo_std,
    alpha=0.2
)

plt.xlabel("Training Iterations (per rollout block)")
plt.ylabel("Mean Episode Return")
plt.title("Simple Spread: Baseline vs MAPPO (3 Seeds)")
plt.legend()
plt.grid(True)
plt.tight_layout()

save_path = os.path.join(RESULTS_DIR, "baseline_vs_mappo_3seeds.png")
plt.savefig(save_path, dpi=300)
print(f"Saved plot to {save_path}")

plt.show()