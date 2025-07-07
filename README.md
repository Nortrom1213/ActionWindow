# Action Window Planning for Stealth Missions Demo 

This repository contains **prototype code** for experimenting with VIP path planning in grid-based stealth games.  
It is **not** a finished framework—just a collection of scripts we used for early feasibility studies.  
A cleaner, general-purpose library (with formal APIs, unit tests, and full documentation) will follow in later work.

---

## Directory / Module Layout

| Path / file                       | Purpose |
|----------------------------------|---------|
| `environment.py`                 | Builds toy grid worlds with static obstacles and patrolling guards; also provides time-expanded graphs. |
| `planner.py`                     | Baseline **GA planner** operating directly on the time-expanded graph. |
| `safe_block_planner.py`          | Extracts spatio-temporal **safe blocks** and plans over them via Dijkstra. |
| `dispersion_block_planner.py`    | Adds dispersion-based regularisation on top of the block graph. |
| `visualize.py`                   | Quick 2-D/3-D plotting helpers (matplotlib + Plotly). |
| `main.py`                        | CLI wrapper for generating paths and visualisations. |
| `evaluate.py`                    | Batch runner that executes multiple trials and logs metrics / heat maps. |
| `demo_scripts/*.py` (optional)   | One-off notebooks or plotting scripts created during exploration. |
| `evaluation_results*/`           | Auto-generated output folders (routes, heat maps, CSV logs, GIFs). |

---

## Requirements

* Python ≥ 3.8  
* `numpy`, `matplotlib`, `imageio`, `plotly`

## Quick Start

### 1. Choose a map

```
from environment import build_complex_env, build_large_env
env = build_complex_env()   # 20×20 example
```
### 2. Generate a route

```
# Genetic-algorithm path
python main.py generate-ga --out vip_ga.json

# Safe-block path
python main.py generate-blocks --out vip_blocks.json

# Dispersion-regularised path
python main.py generate-dispersion --out vip_dispersion.json
```

### 3. Visualise
```
python main.py visualize --route vip_ga.json --gif ga.gif --html ga.html
```

### 4. Batch evaluation
```
# Runs 500 GA trials, logs metrics, and saves heat maps
python evaluate.py
```

## Notes & Limitations

*Code assumes Manhattan movement (4-neighbour grid + wait) and uses a simple sight radius for guards.

*Only a single VIP is modelled; multi-agent coordination is future work.

*Parameter settings (window thresholds, GA hyper-parameters, etc.) are hard-coded for convenience.

*Error handling, unit tests, and large-scale benchmarks are omitted in this demo.
