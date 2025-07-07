import math, heapq
from collections import deque

def find_safe_blocks(env, Tmin, Tmax_win, Smin, Smax):
    """
    Identify axis-aligned *safe rectangles* in space–time.
    For each temporal window [t_start, t_end] whose duration satisfies
    Tmin ≤ L ≤ Tmax_win, we first derive the set of cells that remain safe
    throughout the interval.  Connected components that are already rectangular
    are retained directly; otherwise the algorithm extracts the largest
    axis-aligned rectangle fully contained in the component.
    The first and last windows (t_start = 0 or t_end = env.t_max) are exempt
    from duration and area thresholds.
    """
    blocks = []
    W, H = env.width, env.height

    for t0 in range(0, env.t_max - Tmin + 1):
        for L in range(Tmin, Tmax_win + 1):
            t1 = t0 + L
            if t1 > env.t_max:
                break

            # Boolean mask: cells that are simultaneously safe for all t ∈ [t0, t1]
            safe = [[True] * H for _ in range(W)]
            for t in range(t0, t1 + 1):
                for x in range(W):
                    for y in range(H):
                        if not env.is_safe(x, y, t):
                            safe[x][y] = False

            visited = [[False] * H for _ in range(W)]
            for x in range(W):
                for y in range(H):
                    if not safe[x][y] or visited[x][y]:
                        continue

                    # Breadth-first search to obtain a connected component *comp*
                    comp = []
                    dq = deque([(x, y)])
                    visited[x][y] = True
                    while dq:
                        ux, uy = dq.popleft()
                        comp.append((ux, uy))
                        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                            vx, vy = ux + dx, uy + dy
                            if (0 <= vx < W and 0 <= vy < H and
                                    safe[vx][vy] and not visited[vx][vy]):
                                visited[vx][vy] = True
                                dq.append((vx, vy))

                    size    = len(comp)
                    length  = t1 - t0 + 1

                    # Thresholds are waived for the first or last temporal block
                    if not (t0 == 0 or t1 == env.t_max):
                        if length < Tmin or length > Tmax_win:
                            continue
                        if size   < Smin or size   > Smax:
                            continue

                    # Axis-aligned bounding box of the component
                    xs, ys   = zip(*comp)
                    x0, x1   = min(xs), max(xs)
                    y0, y1   = min(ys), max(ys)
                    bbox_area = (x1 - x0 + 1) * (y1 - y0 + 1)

                    if bbox_area == size:
                        # The component itself is rectangular
                        region = set(comp)
                    else:
                        # Extract the largest rectangle fully contained in *comp*
                        rect = maximal_rectangle(comp, x0, y0, x1, y1)
                        if rect is None:
                            continue
                        rx0, ry0, rx1, ry1 = rect
                        region = {(i, j)
                                  for i in range(rx0, rx1 + 1)
                                  for j in range(ry0, ry1 + 1)}
                        size = len(region)
                        # Re-apply area thresholds to interior blocks
                        if not (t0 == 0 or t1 == env.t_max):
                            if size < Smin or size > Smax:
                                continue

                    blocks.append({
                        'region':  region,
                        't_start': t0,
                        't_end':   t1
                    })
    return blocks


def maximal_rectangle(comp, x0, y0, x1, y1):
    """
    Given a connected component *comp* (set of (x, y) pairs) and its bounding
    box [x0, x1] × [y0, y1], compute—via the “largest rectangle in a
    histogram’’ technique—the maximal axis-aligned rectangle entirely
    contained in *comp*.
    Returns (rx0, ry0, rx1, ry1) or None if no rectangle exists.
    """
    W           = x1 - x0 + 1
    heights     = [0] * W
    best_area   = 0
    best_rect   = None
    comp_set    = set(comp)

    for row in range(y0, y1 + 1):
        # Update histogram heights for the current row
        for col in range(x0, x1 + 1):
            if (col, row) in comp_set:
                heights[col - x0] += 1
            else:
                heights[col - x0]  = 0

        # Largest rectangle in the current histogram
        area, (l, r, h) = largest_histogram_rect(heights)
        if area > best_area:
            best_area = area
            best_rect = (x0 + l, row - h + 1, x0 + r, row)
    return best_rect


def largest_histogram_rect(heights):
    """
    Monotone-stack algorithm for the “largest rectangle in a histogram’’
    problem.
    Returns (max_area, (left_idx, right_idx, height)).
    """
    stack, max_area, best = [], 0, (0, 0, 0)
    for i, h in enumerate(heights + [0]):      # sentinel zero height
        while stack and heights[stack[-1]] >= h:
            height = heights[stack.pop()]
            left   = stack[-1] + 1 if stack else 0
            right  = i - 1
            area   = height * (right - left + 1)
            if area > max_area:
                max_area, best = area, (left, right, height)
        stack.append(i)
    return max_area, best


def build_2d_graph(env):
    """
    Construct a 2-D adjacency list that ignores guard vision and considers
    only static obstacles.
    """
    graph = {}
    for x in range(env.width):
        for y in range(env.height):
            if env.is_free(x, y):
                nbrs = []
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (-1, 0)]:  # note: north step omitted by design
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < env.width and 0 <= ny < env.height
                            and env.is_free(nx, ny)):
                        nbrs.append((nx, ny))
                graph[(x, y)] = nbrs
    return graph


def a_star_2d(graph, start, goal):
    """
    Standard A* search in two spatial dimensions; returns a path or None.
    """
    open_set = [(abs(start[0] - goal[0]) + abs(start[1] - goal[1]), start)]
    came, g  = {}, {start: 0}
    while open_set:
        _, cur = heapq.heappop(open_set)
        if cur == goal:
            path = []
            while cur in came:
                path.append(cur)
                cur = came[cur]
            path.append(start)
            return list(reversed(path))
        for nei in graph[cur]:
            ng = g[cur] + 1
            if ng < g.get(nei, math.inf):
                g[nei] = ng
                f      = ng + abs(nei[0] - goal[0]) + abs(nei[1] - goal[1])
                heapq.heappush(open_set, (f, nei))
                came[nei] = cur
    return None


def build_block_graph(blocks, graph2d, delta_t_max):
    """
    Construct a directed graph whose vertices are safe blocks.
    For each feasible pair (i → j), we record the minimal spatial cost
    (shortest 2-D path length) required to travel from any cell in block *i*
    to any cell in block *j* within the permissible temporal gap
    0 ≤ Δt ≤ delta_t_max.  A transition is viable only if its spatial cost
    does not exceed the temporal slack Δt.
    """
    N   = len(blocks)
    adj = {i: [] for i in range(N)}
    for i, bi in enumerate(blocks):
        for j, bj in enumerate(blocks):
            if i == j:
                continue
            dt = bj['t_start'] - bi['t_end']
            if dt < 0 or dt > delta_t_max:
                continue
            best = math.inf
            for u in bi['region']:
                for v in bj['region']:
                    seg = a_star_2d(graph2d, u, v)
                    if seg:
                        d = len(seg) - 1
                        if d < best:
                            best = d
                        if best <= 1:
                            break
                if best <= 1:
                    break
            if best < math.inf and best <= dt:
                adj[i].append((j, best))
    return adj


def dijkstra_block_multi(adj, blocks, start_ids, goal_ids, min_cover):
    """
    Multi-state Dijkstra search.
    A state encodes (block_id, guard-coverage mask).  The algorithm seeks a
    path from any *start_ids* block to any *goal_ids* block that covers at
    least *min_cover* distinct guards.
    """
    dist, prev = {}, {}
    pq = []
    for s in start_ids:
        dist[(s, 0)] = 0
        heapq.heappush(pq, (0, s, 0))

    final = None
    while pq:
        cost, i, mask = heapq.heappop(pq)
        if cost > dist.get((i, mask), math.inf):
            continue
        if i in goal_ids and bin(mask).count('1') >= min_cover:
            final = (i, mask)
            break
        for j, w in adj[i]:
            state = (j, mask)
            nc    = cost + w
            if nc < dist.get(state, math.inf):
                dist[state] = nc
                prev[state] = (i, mask)
                heapq.heappush(pq, (nc, j, mask))

    if final is None:
        return None

    seq, cur = [], final
    while cur in prev:
        seq.append(cur[0])
        cur = prev[cur]
    seq.append(cur[0])
    return list(reversed(seq))


def instantiate_block_path(seq, blocks, env, graph2d):
    """
    Convert a sequence of block indices into a concrete space–time path.
    The agent waits in place when temporal slack remains after completing a
    spatial transfer.
    """
    path   = []
    b0     = blocks[seq[0]]
    u      = next(iter(b0['region']))
    t_prev = b0['t_end']
    path.append((u[0], u[1], t_prev))

    for idx in range(len(seq) - 1):
        bi, bj = blocks[seq[idx]], blocks[seq[idx + 1]]
        u      = next(iter(bi['region']))
        v      = next(iter(bj['region']))
        seg2d  = a_star_2d(graph2d, u, v)
        for k, (x, y) in enumerate(seg2d[1:], 1):
            path.append((x, y, t_prev + k))
        t_prev += len(seg2d) - 1
        while t_prev < bj['t_start']:
            path.append((v[0], v[1], t_prev + 1))
            t_prev += 1
    return path
