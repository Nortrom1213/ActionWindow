# Action Window Planning for Stealth Missions

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FCOG64752.2025.11114360-blue.svg)](
https://doi.org/10.1109/cog64752.2025.11114360)

This repository contains the **prototype code** for our IEEE CoG 2025 paper "Action Window Planning for Stealth Missions". The implementation demonstrates automated VIP path planning in grid-based stealth games with guard patrols, generating spatiotemporal action windows that enable player assassination opportunities.




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
| `evaluiate_large.py`             | Large-scale evaluation script for 100×100 mazes. |
| `evaluation_results*/`           | Auto-generated output folders (routes, heat maps, CSV logs, GIFs). |

---

## Requirements

* Python ≥ 3.8  
* `numpy`, `matplotlib`, `imageio`, `plotly`

## Quick Start

### 1. Choose a map

```python
from environment import build_complex_env, build_large_env
env = build_complex_env()   # 20×20 example
env = build_large_env()     # 100×100 example
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

### 3. Visualize
```
python main.py visualize --route vip_ga.json --gif ga.gif --html ga.html
```

### 4. Batch evaluation
```bash
# Runs 500 GA trials, logs metrics, and saves heat maps
python evaluate.py

# Large-scale evaluation (100×100 mazes)
python evaluiate_large.py
```

## Key Features

- **Two Planning Approaches**: Genetic Algorithm (GA) and Safe-Block Search with dispersion regularization
- **Time-Expanded Graphs**: Spatiotemporal pathfinding with guard patrol constraints
- **Action Window Generation**: Automatic identification of safe assassination opportunities
- **Multi-Objective Optimization**: Balancing window count, dispersion, smoothness, and coverage
- **Comprehensive Evaluation**: Metrics including reachability, dispersion, uniqueness, and runtime



## Notes & Limitations

- Code assumes Manhattan movement (4-neighbour grid + wait) and uses a simple sight radius for guards
- Only a single VIP is modelled; multi-agent coordination is future work
- Parameter settings (window thresholds, GA hyper-parameters, etc.) are hard-coded for convenience
- Error handling, unit tests, and large-scale benchmarks are omitted in this demo

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{xu2025action,
  title={Action Window Planning for Stealth Missions},
  author={Xu, Kaijie and Verbrugge, Clark},
  booktitle={2025 IEEE Conference on Games (CoG)},
  year={2025},
  organization={IEEE},
  doi={10.1109/COG64752.2025.11114360}
}
```
