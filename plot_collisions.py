import numpy as np
import matplotlib.pyplot as plt
import os

RESULTS_DIR = "results_count/simple_spread"

paths = [
    os.path.join(RESULTS_DIR, "mappo_collisions_0.npy"),
    os.path.join(RESULTS_DIR, "mappo_collisions_1.npy"),
    os.path.join(RESULTS_DIR, "mappo_collisions_2.npy"),
]

def smooth(x, window=5):
    return np.convolve(x, np.ones(window)/window, mode='valid')

data = [smooth(np.load(p)) for p in paths]
data = np.stack(data)

mean = data.mean(axis=0)
std = data.std(axis=0)

x = np.arange(len(mean))

plt.plot(x, mean, label="MAPPO Collisions")
plt.fill_between(x, mean-std, mean+std, alpha=0.2)

plt.xlabel("Training Iterations")
plt.ylabel("Collision Count")
plt.title("MAPPO Collision Count (3 Seeds)")
plt.legend()
plt.grid(True)

save_path = os.path.join(RESULTS_DIR, "mappo_collisions.png")
plt.savefig(save_path, dpi=300)

plt.show()