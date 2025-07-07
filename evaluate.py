"""
Large-scale batch evaluation for the GA planner only.

Configuration notes
-------------------
NUM_RUNS   : number of independent GA trials (n = 500).
SAFE_PARAMS: thresholds for safe-block extraction; used solely for
             heat-map accumulation and metric computation.
DELTA_T    : maximum admissible temporal gap between safe blocks.
START, GOAL: spatial endpoints in the complex 20 × 20 test map.

For each run the script logs runtime and path-quality metrics to *LOG_CSV*
and accumulates a visitation heat map.
"""
import os
import time
import csv
import numpy as np

from environment import build_complex_env  # adjust import path if required

# Genetic-Algorithm planner
from planner import (
    a_star,
    optimize_vip_path,
    save_path as save_json_path,
    extract_windows,
    validate_reachability,
    dispersion_metrics,
    route_uniqueness_score,
    max_window_metrics,
    smoothness_score,
)

# Safe-Block and Dispersion modules are imported only for metric consistency
from safe_block_planner import (
    find_safe_blocks,
    build_2d_graph,
    build_block_graph,
    dijkstra_block_multi,
    instantiate_block_path,
)
from dispersion_block_planner import dispersion_search

# ─────────────────────────── Experimental constants ───────────────────────── #
NUM_RUNS = 500  # number of GA runs

SAFE_PARAMS = {"Tmin": 5, "Tmax_win": 30, "Smin": 3, "Smax": 20}
DELTA_T = 30  # temporal slack between safe blocks

GA_PARAMS = {"generations": 30, "pop_size": 50}
DISP_PARAMS = {"lambda_t": 1.0, "lambda_s": 1.0, "lambda_n": 0.5}

START = (1, 1)
GOAL = (18, 18)

# Output hierarchy
OUT_DIR = "evaluation_results_test"
LOG_CSV = os.path.join(OUT_DIR, "log.csv")
GA_DIR = os.path.join(OUT_DIR, "ga")
BLOCKS_DIR = os.path.join(OUT_DIR, "blocks")
DISP_DIR = os.path.join(OUT_DIR, "dispersion")
HEATMAP_DIR = os.path.join(OUT_DIR, "heatmaps")

for d in (GA_DIR, BLOCKS_DIR, DISP_DIR, OUT_DIR, HEATMAP_DIR):
    os.makedirs(d, exist_ok=True)

# Environment construction (20 × 20 maze, three guards)
env = build_complex_env()
W, H = env.width, env.height

# Heat-map accumulator per method
heatmaps = {
    "GA": np.zeros((W, H), dtype=int),
    "Blocks": np.zeros((W, H), dtype=int),
    "Dispersion": np.zeros((W, H), dtype=int),
}


# --------------------------------------------------------------------------- #
# Heat-map and metric utilities                                               #
# --------------------------------------------------------------------------- #
def update_heatmap(path, method):
    """Increment the visit counter for every spatial cell in *path*."""
    hm = heatmaps[method]
    for p in path:
        x, y = (p["x"], p["y"]) if isinstance(p, dict) else (p[0], p[1])
        if 0 <= x < W and 0 <= y < H:
            hm[x, y] += 1


def compute_metrics(env, path, filename, runtime, method, run_id):
    """
    Assemble a dictionary of quantitative metrics and update the
    method-specific heat map.
    """
    update_heatmap(path, method)

    wins = extract_windows(path, env)
    reach = validate_reachability(
        env, wins, env.build_safe_time_expanded_graph(), (START[0], START[1], 0)
    )
    t_disp, s_disp = dispersion_metrics(reach)
    r_uniq = route_uniqueness_score(path)
    max_len, max_sz = max_window_metrics(wins)
    smooth = smoothness_score(path)
    path_len = len(path) - 1

    return {
        "method": method,
        "run": run_id,
        "path_file": filename,
        "runtime_s": round(runtime, 3),
        "num_windows": len(wins),
        "reachable_windows": len(reach),
        "time_dispersion": round(t_disp, 3),
        "space_dispersion": round(s_disp, 3),
        "route_uniqueness": round(r_uniq, 3),
        "max_mid_window_len": max_len,
        "max_mid_window_size": max_sz,
        "route_smoothness": round(smooth, 3),
        "path_length": path_len,
    }


# --------------------------------------------------------------------------- #
# Method-specific evaluators                                                  #
# --------------------------------------------------------------------------- #
def evaluate_ga(env, run_id):
    """Run one GA trial and return its metric record."""
    t0 = time.time()
    static_graph = env.build_static_time_expanded_graph()
    safe_graph = env.build_safe_time_expanded_graph()
    start3 = (START[0], START[1], 0)

    init = a_star(static_graph, start3, GOAL, env.t_max)
    best = optimize_vip_path(
        init,
        env,
        static_graph,
        GOAL,
        start3,
        safe_graph,
        generations=GA_PARAMS["generations"],
        pop_size=GA_PARAMS["pop_size"],
    )
    duration = time.time() - t0

    fn = os.path.join(GA_DIR, f"route_{run_id:03d}.json")
    save_json_path(best, fn)
    return compute_metrics(env, best, fn, duration, "GA", run_id)


# --------------------------------------------------------------------------- #
# Main loop                                                                   #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # CSV initialisation
    fieldnames = [
        "method",
        "run",
        "path_file",
        "runtime_s",
        "num_windows",
        "reachable_windows",
        "time_dispersion",
        "space_dispersion",
        "route_uniqueness",
        "max_mid_window_len",
        "max_mid_window_size",
        "route_smoothness",
        "path_length",
    ]
    with open(LOG_CSV, "w", newline="") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()

        for run_id in range(1, NUM_RUNS + 1):
            stats = evaluate_ga(env, run_id)
            writer.writerow(stats)
            print(
                f"[GA #{run_id}] "
                f"windows={stats['num_windows']} "
                f"time={stats['runtime_s']} s"
            )

    # Persist heat maps
    for method, hm in heatmaps.items():
        np.save(os.path.join(HEATMAP_DIR, f"heatmap_{method}.npy"), hm)

    print(f"\n✅  Completed {NUM_RUNS} GA runs. Heat maps saved to '{HEATMAP_DIR}'.")
