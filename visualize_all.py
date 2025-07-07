"""
Route-visualisation utilities for the stealth-planning benchmark.

This script supports
    • a 2-D overview that overlays guard patrols and VIP trajectories, and
    • a 3-D space–time visualisation that highlights action windows.

The command-line entry point at the bottom provides example usage.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from environment import build_complex_env
from planner import extract_windows

# Colour palette for action-window overlays
WINDOW_COLORS = ["cyan", "yellow", "lime", "pink", "gray", "orange"]


# --------------------------------------------------------------------------- #
# I/O                                                                         #
# --------------------------------------------------------------------------- #
def load_route(route_file):
    """Deserialize a space–time path from *route_file*."""
    with open(route_file, "r") as f:
        data = json.load(f)
    pts = []
    for p in data:
        if isinstance(p, dict):
            x, y, t = int(p["x"]), int(p["y"]), int(p["t"])
        else:
            x, y, t = map(int, p)
        pts.append((x, y, t))
    return pts


# --------------------------------------------------------------------------- #
# 2-D overview (guards + VIP trajectories)                                    #
# --------------------------------------------------------------------------- #
def plot_2d_overview(env, route_files, labels, out_image):
    """
    Render a 2-D map that shows
        • static obstacles,
        • guard patrol loops with arrows, and
        • one or more VIP trajectories.
    Action windows are shaded *only* for the GA-optimised route.

    Parameters
    ----------
    env          : GridEnvironment
    route_files  : list[str]
        Filenames of VIP routes to overlay.
    labels       : list[str]
        Legend entries matching *route_files*.
    out_image    : str
        Output PNG filename.
    """
    W, H = env.width, env.height
    num_guards = len(env.guard_routes)
    total_agents = num_guards + len(route_files)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(0, W + 1))
    ax.set_yticks(np.arange(0, H + 1))
    ax.grid(color="lightgray", linewidth=0.5)

    # Static obstacles (black squares)
    ox, oy = np.where(env.obstacles)
    ax.scatter(ox + 0.5, oy + 0.5, marker="s", s=100, color="black")

    # Guard patrols (arrowed polylines, one colour per guard)
    guard_colors = ["blue", "green", "saddlebrown", "purple", "cyan"]
    for i, route in enumerate(env.guard_routes):
        angle = 2 * np.pi * i / total_agents          # radial offset to de-overlap arrows
        dx, dy = 0.2 * np.cos(angle), 0.2 * np.sin(angle)
        xs = [p[0] + 0.5 + dx for p in route]
        ys = [p[1] + 0.5 + dy for p in route]
        dxs, dys = np.diff(xs), np.diff(ys)
        ax.quiver(
            xs[:-1],
            ys[:-1],
            dxs,
            dys,
            scale_units="xy",
            angles="xy",
            scale=1,
            width=0.005,
            color=guard_colors[i],
            alpha=0.8,
            label=f"Guard {i + 1}",
        )

    # VIP trajectories (red or magenta depending on label)
    for idx, (route_file, label) in enumerate(zip(route_files, labels)):
        vip_idx = num_guards + idx
        angle = 2 * np.pi * vip_idx / total_agents
        dx, dy = 0.2 * np.cos(angle), 0.2 * np.sin(angle)

        route = load_route(route_file)
        xs = [x + 0.5 + dx for x, y, t in route]
        ys = [y + 0.5 + dy for x, y, t in route]
        dxs, dys = np.diff(xs), np.diff(ys)

        vip_color = "magenta" if "Dispersion" in label else "red"
        ax.quiver(
            xs[:-1],
            ys[:-1],
            dxs,
            dys,
            scale_units="xy",
            angles="xy",
            scale=1,
            width=0.007,
            color=vip_color,
            label=label,
        )

        # Shade action windows for the GA trajectory
        if "GA" in label:
            wins = extract_windows(route, env)
            for j, w in enumerate(wins):
                xs_w = [c[0] for c in w["region"]]
                ys_w = [c[1] for c in w["region"]]
                x0, x1 = min(xs_w), max(xs_w) + 1
                y0, y1 = min(ys_w), max(ys_w) + 1
                rect = Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    facecolor=WINDOW_COLORS[j % len(WINDOW_COLORS)],
                    edgecolor="k",
                    alpha=0.3,
                )
                ax.add_patch(rect)

    # Invert axes so that (0, 0) appears at the upper-left corner
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.invert_xaxis()
    ax.invert_yaxis()

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(loc="upper right", fontsize=16)
    plt.tight_layout()
    plt.savefig(out_image, dpi=300)
    plt.close()
    print(f"Saved 2-D overview to {out_image}")


# --------------------------------------------------------------------------- #
# 3-D space–time visualisation                                               #
# --------------------------------------------------------------------------- #
def get_cuboid_verts(x0, x1, y0, y1, z0, z1):
    """Return the six faces of an axis-aligned cuboid as lists of (x, y, z)."""
    return [
        [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
        [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
        [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
        [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
    ]


def plot_3d_route(env, route_file, label, out_image):
    """
    Plot a single VIP trajectory in 3-D space–time.
    Path segments falling inside action windows are coloured red; the rest are
    navy.  Each window is visualised as a translucent cuboid.
    """
    z_scale = 0.2  # compress the time axis for visual clarity
    W, H, T = env.width, env.height, env.t_max

    route = load_route(route_file)
    wins = extract_windows(route, env)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect((W, H, T * z_scale))
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Static obstacles on the floor plane
    ox, oy = np.where(env.obstacles)
    ax.scatter(
        ox + 0.5,
        oy + 0.5,
        0,
        marker="s",
        s=40,
        color="gray",
        alpha=0.5,
        depthshade=False,
    )

    # VIP trajectory (colour-coded by window membership)
    def in_win(t):
        return any(w["t_start"] <= t <= w["t_end"] for w in wins)

    pts = [(x + 0.5, y + 0.5, t * z_scale) for x, y, t in route]
    segments, curr, flag = [], [pts[0]], in_win(route[0][2])
    for i in range(1, len(pts)):
        inside = in_win(route[i][2])
        if inside != flag:
            segments.append((flag, curr))
            curr = [pts[i - 1], pts[i]]
            flag = inside
        else:
            curr.append(pts[i])
    segments.append((flag, curr))

    for inside, seg in segments:
        xs = [p[0] for p in seg]
        ys = [p[1] for p in seg]
        zs = [p[2] for p in seg]
        col = "red" if inside else "navy"
        ax.plot(xs, ys, zs, color=col, linewidth=3, alpha=0.9)

    # Ground-plane projection (dashed dark-red polyline)
    fx = [p[0] for p in pts]
    fy = [p[1] for p in pts]
    ax.plot(fx, fy, 0, "--", color="darkred", linewidth=1.5, alpha=0.8)

    # Action windows as translucent cuboids
    for j, w in enumerate(wins):
        xs_w = [c[0] for c in w["region"]]
        ys_w = [c[1] for c in w["region"]]
        x0, x1 = min(xs_w), max(xs_w) + 1
        y0, y1 = min(ys_w), max(ys_w) + 1
        z0, z1 = w["t_start"] * z_scale, w["t_end"] * z_scale
        col = WINDOW_COLORS[j % len(WINDOW_COLORS)]

        # 3-D cuboid
        verts = get_cuboid_verts(x0, x1, y0, y1, z0, z1)
        ax.add_collection3d(
            Poly3DCollection(verts, facecolors=col, edgecolors="k", alpha=0.3)
        )
        # Footprint on the ground plane
        floor = [(x0, y0, 0), (x1, y0, 0), (x1, y1, 0), (x0, y1, 0)]
        ax.add_collection3d(
            Poly3DCollection([floor], facecolors=col, edgecolors="k", alpha=0.2)
        )

    ax.set_xlim(-0.5, W + 0.5)
    ax.set_ylim(-0.5, H + 0.5)
    ax.set_zlim(0, T * z_scale)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Time")
    ax.view_init(elev=25, azim=-60)

    plt.tight_layout()
    plt.savefig(out_image, dpi=300)
    plt.close()
    print(f"Saved 3-D {label} visualisation to {out_image}")


# --------------------------------------------------------------------------- #
# Example usage                                                               #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    env = build_complex_env()
    disp = "evaluation_results/dispersion/route_001.json"
    ga = "small_data/vip_ga.json"

    # 2-D overview: guards + both VIP routes (windows shown only for GA)
    plot_2d_overview(
        env,
        [disp, ga],
        ["Dispersion-Reg", "GA"],
        out_image="routes_windows_2d.png",
    )

    # Space–time plots (guards omitted for clarity)
    plot_3d_route(env, disp, "Dispersion-Reg", "te_dispersion.png")
    plot_3d_route(env, ga, "GA", "te_ga.png")
