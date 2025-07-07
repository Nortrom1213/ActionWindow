"""
Batch evaluation script for the three VIP-planning methods:

    • GA   : genetic-algorithm optimisation over a time-expanded graph
    • Blocks: shortest path over pre-computed safe spatio-temporal blocks
    • Dispersion: dispersion-regularised search over the block graph

For each run the script records runtime and multiple path-quality metrics,
writes them to *LOG_CSV*, and accumulates visitation heat maps.
"""
import os
import time
import csv
import numpy as np

# Replace the import below if your project structure differs
from environment import build_complex_env

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

# Safe-Block planner
from safe_block_planner import (
    find_safe_blocks,
    build_2d_graph,
    build_block_graph,
    dijkstra_block_multi,
    instantiate_block_path,
)

# Dispersion-Regularised planner
from dispersion_block_planner import dispersion_search

# ─────────────────────────── Experimental configuration ───────────────────── #
NUM_RUNS = 500  # number of independent trials

SAFE_PARAMS = {
    "Tmin": 5,
    "Tmax_win": 30,
    "Smin": 3,
    "Smax": 30,
}
DELTA_T = 30  # maximum temporal gap between successive blocks

GA_PARAMS = {"generations": 30, "pop_size": 50}
DISP_PARAMS = {"lambda_t": 1.0, "lambda_s": 1.0, "lambda_n": 0.5}

START = (1, 1)
GOAL = (98, 98)

# Output directories
OUT_DIR = "evaluation_results_large_test"
LOG_CSV = os.path.join(OUT_DIR, "log.csv")
GA_DIR = os.path.join(OUT_DIR, "ga")
BLOCKS_DIR = os.path.join(OUT_DIR, "blocks")
DISP_DIR = os.path.join(OUT_DIR, "dispersion")
HEATMAP_DIR = os.path.join(OUT_DIR, "heatmaps")

# Create directory structure
for d in (GA_DIR, BLOCKS_DIR, DISP_DIR, OUT_DIR, HEATMAP_DIR):
    os.makedirs(d, exist_ok=True)

# Construct the environment once
env = build_complex_env()
W, H = env.width, env.height

# Heat-map accumulators (per method)
heatmaps = {
    "GA": np.zeros((W, H), dtype=int),
    "Blocks": np.zeros((W, H), dtype=int),
    "Dispersion": np.zeros((W, H), dtype=int),
}


# --------------------------------------------------------------------------- #
# Utility functions                                                           #
# --------------------------------------------------------------------------- #
def update_heatmap(path, method):
    """Increment visit counts for every spatial cell along *path*."""
    hm = heatmaps[method]
    for p in path:
        if isinstance(p, dict):
            x, y = p["x"], p["y"]
        else:
            x, y, _ = p
        if 0 <= x < W and 0 <= y < H:
            hm[x, y] += 1


def compute_metrics(env, path, filename, runtime, method, run_id):
    """Return a dict of quantitative metrics for the given trajectory."""
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
# Method-specific evaluation routines                                         #
# --------------------------------------------------------------------------- #
def evaluate_ga(env, run_id):
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


def evaluate_blocks(env, run_id):
    t0 = time.time()
    blocks = find_safe_blocks(env, **SAFE_PARAMS)
    start_blk = {"region": {START}, "t_start": 0, "t_end": 0, "guard_id": None}
    goal_blk = {
        "region": {GOAL},
        "t_start": env.t_max,
        "t_end": env.t_max,
        "guard_id": None,
    }
    blocks_full = [start_blk] + blocks + [goal_blk]

    g2d = build_2d_graph(env)
    adj = build_block_graph(blocks_full, g2d, DELTA_T)
    seq = dijkstra_block_multi(
        adj,
        blocks_full,
        start_ids=[0],
        goal_ids=[len(blocks_full) - 1],
        min_cover=0,
    )
    path = instantiate_block_path(seq, blocks_full, env, g2d)
    duration = time.time() - t0

    fn = os.path.join(BLOCKS_DIR, f"route_{run_id:03d}.json")
    save_json_path(path, fn)
    return compute_metrics(env, path, fn, duration, "Blocks", run_id)


def evaluate_dispersion(env, run_id):
    t0 = time.time()
    blocks = find_safe_blocks(env, **SAFE_PARAMS)
    start_blk = {"region": {START}, "t_start": 0, "t_end": 0}
    goal_blk = {"region": {GOAL}, "t_start": env.t_max, "t_end": env.t_max}
    blocks_full = [start_blk] + blocks + [goal_blk]

    g2d = build_2d_graph(env)
    seq = dispersion_search(
        blocks_full,
        g2d,
        DELTA_T,
        lambda_t=DISP_PARAMS["lambda_t"],
        lambda_s=DISP_PARAMS["lambda_s"],
        lambda_n=DISP_PARAMS["lambda_n"],
    )
    path = instantiate_block_path(seq, blocks_full, env, g2d)
    duration = time.time() - t0

    fn = os.path.join(DISP_DIR, f"route_{run_id:03d}.json")
    save_json_path(path, fn)
    return compute_metrics(env, path, fn, duration, "Dispersion", run_id)


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # CSV header
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
            for fn in (evaluate_ga, evaluate_dispersion):
                stats = fn(env, run_id)
                writer.writerow(stats)
                print(
                    f"[{stats['method']} #{run_id}] "
                    f"windows={stats['num_windows']} "
                    f"time={stats['runtime_s']} s"
                )

    # Persist visit heat maps
    for method, hm in heatmaps.items():
        np.save(os.path.join(HEATMAP_DIR, f"heatmap_{method}.npy"), hm)

    print(f"\n✅  Completed {NUM_RUNS} run(s). Heat maps saved to '{HEATMAP_DIR}'.")
