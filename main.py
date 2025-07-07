import argparse
from environment import build_complex_env, build_large_env
import time

# ──────────────────────────── Genetic-Algorithm Planner ────────────────────── #
from planner import (
    a_star, optimize_vip_path, save_path,
    extract_windows, validate_reachability,
    dispersion_metrics, route_uniqueness_score,
    max_window_metrics, smoothness_score
)

# ───────────────────────────────── Safe-Block Planner ───────────────────────── #
from safe_block_planner import (
    find_safe_blocks, build_2d_graph,
    build_block_graph, dijkstra_block_multi,
    instantiate_block_path
)

# ───────────── Dispersion-Regularised Safe-Block Planner ───────────────────── #
from dispersion_block_planner import dispersion_search

# ─────────────────────────────── Visualisation ─────────────────────────────── #
from visualize import (
    make_simulation_gif, plot_time_expanded_graph,
    load_path
)

# Select an environment
# env = build_large_env()
env = build_complex_env()

# --------------------------------------------------------------------------- #
# Route-generation modes                                                      #
# --------------------------------------------------------------------------- #
def mode_generate_ga(args):
    """
    Genetic-algorithm optimisation on top of an initial A* path.
    Saves the best VIP route to *args.out*.
    """
    static_graph = env.build_static_time_expanded_graph()
    safe_graph   = env.build_safe_time_expanded_graph()
    start = (args.sx, args.sy, 0)
    goal  = (args.gx, args.gy)

    init = a_star(static_graph, start, goal, env.t_max)
    if init is None:
        print("❌  A* failed to find an initial path"); return

    best = optimize_vip_path(
        init, env, static_graph, goal, start,
        safe_graph,
        generations=args.generations,
        pop_size=args.pop_size
    )
    save_path(best, args.out)
    print(f"✅  GA-based VIP route → {args.out}")

def mode_generate_blocks(args):
    """
    Safe-Block method: extract spatio-temporal blocks, plan over the block
    graph, and instantiate a concrete VIP trajectory.
    """
    blocks = find_safe_blocks(
        env, Tmin=args.Tmin, Tmax_win=args.Tmax,
        Smin=args.Smin, Smax=args.Smax
    )
    start_blk = {'region': {(args.sx, args.sy)},
                 't_start': 0, 't_end': 0, 'guard_id': None}
    goal_blk  = {'region': {(args.gx, args.gy)},
                 't_start': env.t_max, 't_end': env.t_max, 'guard_id': None}
    blocks = [start_blk] + blocks + [goal_blk]

    graph2d = build_2d_graph(env)
    adj     = build_block_graph(blocks, graph2d, args.delta_t)

    seq = dijkstra_block_multi(
        adj, blocks,
        start_ids=[0], goal_ids=[len(blocks) - 1],
        min_cover=0
    )
    if not seq:
        print("❌  No feasible block sequence"); return

    path = instantiate_block_path(seq, blocks, env, graph2d)
    save_path(path, args.out)
    print(f"✅  Safe-Block VIP route → {args.out}")

def mode_generate_dispersion(args):
    """
    Dispersion-regularised search over the block graph.
    """
    blocks = find_safe_blocks(
        env, Tmin=args.Tmin, Tmax_win=args.Tmax,
        Smin=args.Smin, Smax=args.Smax
    )
    start_blk = {'region': {(args.sx, args.sy)},
                 't_start': 0, 't_end': 0}
    goal_blk  = {'region': {(args.gx, args.gy)},
                 't_start': env.t_max, 't_end': env.t_max}
    blocks = [start_blk] + blocks + [goal_blk]

    graph2d = build_2d_graph(env)
    seq = dispersion_search(
        blocks, graph2d, args.delta_t,
        lambda_t=args.lambda_t,
        lambda_s=args.lambda_s,
        lambda_n=args.lambda_n
    )
    if not seq:
        print("❌  No dispersion-regularised path"); return

    path = instantiate_block_path(seq, blocks, env, graph2d)
    save_path(path, args.out)
    print(f"✅  Dispersion-regularised VIP route → {args.out}")

def mode_visualize(args):
    """
    Compute diagnostic metrics and produce both a GIF and (optionally)
    a Plotly HTML file for the time-expanded graph.
    """
    path        = load_path(args.route)
    safe_graph  = env.build_safe_time_expanded_graph()

    wins        = extract_windows(path, env)
    reach       = validate_reachability(env, wins, safe_graph, (args.sx, args.sy, 0))
    t_disp, s_disp  = dispersion_metrics(reach)
    r_uniq          = route_uniqueness_score(path)
    max_len, max_sz = max_window_metrics(wins)
    smooth          = smoothness_score(path)

    print("\n=== Path Metrics ===")
    print(f"Total windows:       {len(wins)}")
    print(f"Reachable windows:   {len(reach)}")
    print(f"Time dispersion:     {t_disp:.2f}")
    print(f"Space dispersion:    {s_disp:.2f}")
    print(f"Route uniqueness:    {r_uniq:.2f}")
    print(f"Max mid-window len:  {max_len}")
    print(f"Max mid-window size: {max_sz}")
    print(f"Route smoothness:    {smooth:.2f}")
    print("\n--- Window Details ---")
    for i, w in enumerate(wins, 1):
        ok = "✓" if w in reach else "✗"
        print(f"Window {i:02d}: t[{w['t_start']}→{w['t_end']}] "
              f"len={w['length']} size={w['size']} reachable={ok}")
    print("=====================\n")

    make_simulation_gif(env, vip_path=path,
                        show_vip=True, out_gif=args.gif)
    # plot_time_expanded_graph(env, out_html=args.html)

def mode_visualize_env(args):
    """
    Visualise the environment (obstacles + guard patrols) without a VIP path.
    """
    make_simulation_gif(env, vip_path=None,
                        show_vip=False, out_gif=args.gif)
    plot_time_expanded_graph(env, out_html=args.html)

# --------------------------------------------------------------------------- #
# Command-line interface                                                     #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    t0 = time.time()
    parser = argparse.ArgumentParser()
    subs   = parser.add_subparsers(dest="cmd")

    # Genetic-Algorithm generator
    g = subs.add_parser("generate-ga")
    g.add_argument("--sx", type=int, default=1);   g.add_argument("--sy", type=int, default=1)
    g.add_argument("--gx", type=int, default=98);  g.add_argument("--gy", type=int, default=98)
    g.add_argument("--generations", type=int, default=30)
    g.add_argument("--pop-size",    type=int, default=50)
    g.add_argument("--out",         default="vip_ga.json")

    # Safe-Block generator
    b = subs.add_parser("generate-blocks")
    b.add_argument("--Tmin",    type=int, default=5)
    b.add_argument("--Tmax",    type=int, default=50)
    b.add_argument("--Smin",    type=int, default=3)
    b.add_argument("--Smax",    type=int, default=50)
    b.add_argument("--delta-t", type=int, default=50)
    b.add_argument("--sx", type=int, default=1);   b.add_argument("--sy", type=int, default=1)
    b.add_argument("--gx", type=int, default=98);  b.add_argument("--gy", type=int, default=98)
    b.add_argument("--out",  default="vip_blocks.json")

    # Dispersion-Regularised generator
    d = subs.add_parser("generate-dispersion")
    d.add_argument("--Tmin",     type=int, default=30)
    d.add_argument("--Tmax",     type=int, default=100)
    d.add_argument("--Smin",     type=int, default=30)
    d.add_argument("--Smax",     type=int, default=100)
    d.add_argument("--delta-t",  type=int, default=100)
    d.add_argument("--lambda-t", type=float, default=1.0)
    d.add_argument("--lambda-s", type=float, default=1.0)
    d.add_argument("--lambda-n", type=float, default=0.5)
    d.add_argument("--sx",       type=int, default=1)
    d.add_argument("--sy",       type=int, default=1)
    d.add_argument("--gx",       type=int, default=98)
    d.add_argument("--gy",       type=int, default=98)
    d.add_argument("--out",      default="vip_dispersion.json")

    # Full visualisation
    v = subs.add_parser("visualize")
    v.add_argument("--route", default="vip_dispersion.json")
    v.add_argument("--sx",    type=int, default=1)
    v.add_argument("--sy",    type=int, default=1)
    v.add_argument("--gif",   default="demo.gif")
    v.add_argument("--html",  default="graph.html")

    # Environment-only visualisation
    e = subs.add_parser("visualize-env")
    e.add_argument("--gif",  default="env_only.gif")
    e.add_argument("--html", default="env_only.html")

    args = parser.parse_args()
    if   args.cmd == "generate-ga":         mode_generate_ga(args)
    elif args.cmd == "generate-blocks":     mode_generate_blocks(args)
    elif args.cmd == "generate-dispersion": mode_generate_dispersion(args)
    elif args.cmd == "visualize":           mode_visualize(args)
    elif args.cmd == "visualize-env":       mode_visualize_env(args)
    else:
        parser.print_help()

# Example invocations:
#   python main.py visualize --route vip_blocks.json     --gif blocks.gif     --html blocks.html
#   python main.py visualize --route vip_ga.json         --gif ga.gif         --html ga.html
#   python main.py visualize --route vip_dispersion.json --gif dispersion.gif --html dispersion.html
