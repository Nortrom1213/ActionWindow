import numpy as np
import matplotlib.pyplot as plt
import imageio
import json
import plotly.graph_objects as go

# --------------------------------------------------------------------------- #
# I/O utilities                                                               #
# --------------------------------------------------------------------------- #
def load_path(filename):
    """Deserialize a space–time path saved by ``save_path``."""
    with open(filename, "r") as f:
        data = json.load(f)
    return [(p["x"], p["y"], p["t"]) for p in data]

# --------------------------------------------------------------------------- #
# 2-D animation of guard patrol and VIP motion                                #
# --------------------------------------------------------------------------- #
def make_simulation_gif(env, vip_path=None, show_vip=True, out_gif="demo.gif"):
    """
    Render an animated GIF that visualises static obstacles (grey),
    patrolling guards (red squares with translucent vision discs), and,
    optionally, the VIP trajectory (blue star).

    Parameters
    ----------
    env : GridEnvironment
        The simulation world.
    vip_path : list[(x, y, t)] or None
        Pre-computed VIP trajectory; ignored if ``show_vip`` is False.
    show_vip : bool
        If True, render the VIP symbol and iterate frames via ``vip_path``;
        otherwise iterate solely over simulation time steps.
    out_gif : str
        Output filename for the generated GIF.
    """
    frames = []

    # If the VIP is hidden, iterate over every time step in the planning horizon
    time_samples = vip_path if (show_vip and vip_path) else list(range(env.t_max))

    for frame_info in time_samples:
        if show_vip and vip_path:
            xvip, yvip, t = frame_info
        else:
            t = frame_info  # interpret frame_info purely as the time index

        # 1. Static background (obstacle grid)
        fig, ax = plt.subplots(figsize=(4, 4))
        grid = env.obstacles.T.astype(int)
        ax.imshow(grid, cmap="gray_r")

        # 2. Guards and their vision radii
        for i in range(len(env.guard_routes)):
            gx, gy = env.guard_pos(i, t)
            ax.scatter(gx, gy, c="red", s=50, marker="s")
            circle = plt.Circle((gx, gy), env.vision_radius, color="red", alpha=0.2)
            ax.add_patch(circle)

        # 3. Optional: VIP position
        if show_vip and vip_path:
            ax.scatter(xvip, yvip, c="blue", s=50, marker="*")

        ax.set_title(f"t = {t}")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.canvas.draw()

        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(img)
        plt.close(fig)

    imageio.mimsave(out_gif, frames, fps=5)
    print(f"GIF saved to {out_gif}")

# --------------------------------------------------------------------------- #
# 3-D visualisation of the time-expanded graph                                #
# --------------------------------------------------------------------------- #
def plot_time_expanded_graph(env, out_html="graph.html"):
    """
    Generate an interactive Plotly HTML visualisation of the full
    time-expanded graph (nodes: safe space–timepoints; edges: admissible
    single-step transitions).  The VIP trajectory is intentionally omitted.
    """
    graph = env.build_time_expanded_graph()
    xs, ys, zs = [], [], []
    for (x, y, t), _ in graph.items():
        xs.append(x)
        ys.append(y)
        zs.append(t)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="markers",
            marker=dict(size=2),
            name="nodes",
        )
    )
    for (x, y, t), neis in graph.items():
        for (nx, ny, nt) in neis:
            fig.add_trace(
                go.Scatter3d(
                    x=[x, nx],
                    y=[y, ny],
                    z=[t, nt],
                    mode="lines",
                    line=dict(width=1),
                    showlegend=False,
                )
            )

    fig.update_layout(
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Time"),
        width=700,
        height=700,
    )
    fig.write_html(out_html)
    print(f"3D graph saved to {out_html}")
