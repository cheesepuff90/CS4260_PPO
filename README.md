# PPO & MAPPO (Hopper + Simple Spread)

## Overview
This project implements **PPO** on:
- **Hopper-v5** (single-agent reproduction)
- **Simple Spread (MPE2)** (multi-agent adaptation)

We compare:
- **Baseline PPO** (decentralized critic)
- **MAPPO** (centralized critic using joint observations)

---

## Key Idea
- Baseline: each agent learns with local value \( V(o_i) \)  
- MAPPO: uses joint value \( V(o_1, o_2, o_3) \) → improved coordination

---

## Files

### Training
- `train_hopper.py` — PPO on Hopper  
- `train_simple_spread.py` — baseline + MAPPO training  

### Models
- `networks.py` — actor + critics  
- `buffer.py` — rollout storage  
- `utils.py` — GAE + helpers  

### Evaluation / Plots
- `eval_hopper.py`, `eval_simple_spread.py`  
- `plot_hopper.py` — Hopper curves  
- `plot_results_multi_seeds.py` — return comparison  
- `plot_collisions_mappo_baseline.py` — collision comparison  

### Visualization
- `simple_spread_gif.py` — rollout GIF  
- `simple_spread_trajectory.py` — trajectory animation  

---

## Results

Metrics:
- **Return** (performance)
- **Collision count** (coordination)

Across **3 seeds (0,1,2)**.

**Findings:**
- MAPPO ≈ similar returns  
- MAPPO ↓ fewer collisions → better coordination  

---

## How to Run

```bash
python train_hopper.py
python train_simple_spread.py
python plot_results_multi_seeds.py
python plot_collisions_mappo_baseline.py