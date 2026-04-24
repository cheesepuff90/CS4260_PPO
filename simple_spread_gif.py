import os
import torch
from torch.distributions import Categorical
import imageio.v2 as imageio
from PIL import Image, ImageDraw

from mpe2 import simple_spread_v3
from networks import SharedActor

DEVICE = torch.device("cpu")

# -------------------------
# Config
# -------------------------
METHOD = "baseline"          # "baseline" or "mappo"
SEED = 1
MODEL_DIR = "models_count"
OUTPUT_DIR = "gifs"
MAX_CYCLES = 100
FRAME_DURATION = 0.6      # seconds per frame for GIF
FPS = 2                   # for mp4
USE_SAMPLED_ACTIONS = False   # False = argmax, True = sample
PRINT_LANDMARK_POSITIONS = False

os.makedirs(OUTPUT_DIR, exist_ok=True)

ACTOR_PATH = os.path.join(MODEL_DIR, f"{METHOD}_actor_{SEED}.pt")
GIF_PATH = os.path.join(OUTPUT_DIR, f"{METHOD}_seed_{SEED}.gif")
MP4_PATH = os.path.join(OUTPUT_DIR, f"{METHOD}_seed_{SEED}.mp4")


def add_label(frame, method, step):
    """Add simple text label to a frame."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.rectangle((5, 5, 180, 35), fill=(255, 255, 255))
    draw.text((10, 10), f"{method.upper()} | step {step}", fill=(0, 0, 0))
    return img


# -------------------------
# Environment
# -------------------------
env = simple_spread_v3.parallel_env(
    N=3,
    local_ratio=0.5,
    max_cycles=MAX_CYCLES,
    render_mode="rgb_array",
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

frames = []
done = False
step = 0

while not done:
    frame = env.render()
    if frame is not None:
        labeled = add_label(frame, METHOD, step)
        frames.append(labeled)

    if PRINT_LANDMARK_POSITIONS:
        try:
            landmark_positions = [
                tuple(lm.state.p_pos.tolist())
                for lm in env.unwrapped.aec_env.env.world.landmarks
            ]
            print(f"step {step} landmarks:", landmark_positions)
        except Exception as e:
            print("Could not access landmark positions:", e)

    actions_dict = {}

    with torch.no_grad():
        for agent in agents:
            agent_obs = obs[agent]
            obs_tensor = torch.tensor(
                agent_obs, dtype=torch.float32, device=DEVICE
            ).unsqueeze(0)

            logits = actor(obs_tensor)

            if USE_SAMPLED_ACTIONS:
                dist = Categorical(logits=logits)
                action = dist.sample()
            else:
                action = torch.argmax(logits, dim=-1)

            actions_dict[agent] = int(action.item())

    next_obs, rewards, terminations, truncations, infos = env.step(actions_dict)

    obs = next_obs
    done = all(terminations[a] or truncations[a] for a in agents)
    step += 1

# final frame
frame = env.render()
if frame is not None:
    labeled = add_label(frame, METHOD, step)
    frames.append(labeled)

env.close()

if len(frames) == 0:
    raise RuntimeError("No frames captured. Check render_mode='rgb_array'.")

# Convert PIL images to arrays for saving
output_frames = [torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes())).numpy()
                 .reshape((img.size[1], img.size[0], 3)) for img in frames]

# Simpler, safer conversion:
output_frames = [__import__("numpy").array(img) for img in frames]

# Save GIF
imageio.mimsave(GIF_PATH, output_frames, duration=FRAME_DURATION)
print(f"Saved GIF to {GIF_PATH}")

# Save MP4
imageio.mimsave(MP4_PATH, output_frames, fps=FPS)
print(f"Saved MP4 to {MP4_PATH}")