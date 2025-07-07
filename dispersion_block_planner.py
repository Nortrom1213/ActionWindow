import math
import heapq
from safe_block_planner import (
    find_safe_blocks,      # noqa: F401  (import retained for external use)
    build_2d_graph,        # noqa: F401
    build_block_graph,
    instantiate_block_path # noqa: F401
)

# ------------------------------------------------------------------------- #
# Auxiliary geometry                                                        #
# ------------------------------------------------------------------------- #
def _mid_time(block):
    """Return the temporal midpoint of a safe block."""
    return (block['t_start'] + block['t_end']) / 2.0

def _centroid(block):
    """Return the arithmetic centroid of the block’s spatial region."""
    xs = [x for x, _ in block['region']]
    ys = [y for _, y in block['region']]
    return (sum(xs) / len(xs), sum(ys) / len(ys))

# ------------------------------------------------------------------------- #
# Best-first search with dispersion-based rewards                           #
# ------------------------------------------------------------------------- #
def dispersion_search(blocks, graph2d, delta_t,
                      lambda_t=1.0, lambda_s=1.0, lambda_n=0.5):
    """
    Conduct a best-first search over the *block graph* with a composite
    priority function that encourages temporal and spatial dispersion as well
    as increased window count.  Specifically,
        priority = path_cost
                   − λ_t · σ_time
                   − λ_s · σ_space
                   − λ_n · N_mid
    where
        σ_time  : standard deviation of block mid-times along the partial path
        σ_space : standard deviation of block centroids
        N_mid   : number of intermediate blocks (i.e., |path| − 2)

    Parameters
    ----------
    blocks   : list[dict]
        Output of ``find_safe_blocks``; the first element is assumed to be the
        start block and the last the goal block.
    graph2d  : dict
        2-D adjacency list produced by ``build_2d_graph``.
    delta_t  : int
        Maximum allowable temporal gap between successive blocks.
    lambda_t, lambda_s, lambda_n : float
        Weights for temporal dispersion, spatial dispersion, and intermediate
        window count, respectively.

    Returns
    -------
    list[int] or None
        Sequence of block indices from start to goal, or ``None`` if no plan
        is found.
    """
    adj = build_block_graph(blocks, graph2d, delta_t)

    # Pre-compute block features for rapid dispersion updates
    times   = [_mid_time(b)  for b in blocks]
    centers = [_centroid(b) for b in blocks]

    def compute_dispersion(seq):
        """Return (σ_time, σ_space) for the block index sequence *seq*."""
        n      = len(seq)
        # Temporal dispersion
        tvals  = [times[i] for i in seq]
        mu_t   = sum(tvals) / n
        sigma_t = math.sqrt(sum((t - mu_t) ** 2 for t in tvals) / n)

        # Spatial dispersion
        pts    = [centers[i] for i in seq]
        mu_x   = sum(x for x, y in pts) / n
        mu_y   = sum(y for x, y in pts) / n
        sigma_s = math.sqrt(
            sum((x - mu_x) ** 2 + (y - mu_y) ** 2 for x, y in pts) / n
        )
        return sigma_t, sigma_s

    # Priority queue initialisation (start from block 0)
    pq         = []
    init_seq   = [0]
    init_cost  = 0.0
    st, ss     = 0.0, 0.0
    wn         = max(0, len(init_seq) - 2)
    init_prio  = init_cost - lambda_t * st - lambda_s * ss - lambda_n * wn
    heapq.heappush(pq, (init_prio, init_cost, init_seq))

    seen      = {}                       # (block_id, path_len) → best_priority
    goal_idx  = len(blocks) - 1

    while pq:
        prio, cost, seq = heapq.heappop(pq)
        u               = seq[-1]
        key             = (u, len(seq))
        if key in seen and seen[key] <= prio:
            continue
        seen[key] = prio

        if u == goal_idx:
            return seq

        for v, w in adj[u]:
            new_seq  = seq + [v]
            new_cost = cost + w
            st, ss   = compute_dispersion(new_seq)
            wn       = max(0, len(new_seq) - 2)     # intermediate blocks
            new_prio = new_cost - lambda_t * st - lambda_s * ss - lambda_n * wn
            heapq.heappush(pq, (new_prio, new_cost, new_seq))
    return None
