import heapq
import random
import json
import math
from collections import deque

# -----------------------------
# Minimum window constraints
# -----------------------------
MIN_WIN_TIME   = 5      # Minimum duration of a window (≥ 5 frames)
MIN_WIN_SIZE   = 3      # Minimum spatial extent of a window (≥ 3 distinct cells)

# -----------------------------------------------------------
# Maximum thresholds for intermediate windows (excluding the
# first and last windows on the path)
# -----------------------------------------------------------
MAX_WIN_TIME   = 30     # Maximum duration for an intermediate window (≤ 30 frames)
MAX_WIN_SIZE   = 20     # Maximum spatial extent for an intermediate window (≤ 20 distinct cells)

# -----------------
# Objective weights
# -----------------
W_WIN_COUNT    = 1.0    # Weight for the number of reachable action windows
W_TIME_DISP    = 0.1    # Weight for temporal dispersion of windows
W_SPACE_DISP   = 0.1    # Weight for spatial dispersion of windows
W_ROUTE_UNIQ   = 0.1    # Weight for path uniqueness (variance from other routes)
W_TIME_MAX     = 0.1    # Penalty weight for the longest intermediate window
W_SIZE_MAX     = 0.1    # Penalty weight for the largest intermediate window
W_SMOOTHNESS   = 0.2    # Weight for geometric smoothness of the route

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])

def a_star(graph, start, goal, t_max):
    open_set = [(heuristic(start, (goal[0], goal[1], 0)), start)]
    came_from = {}
    g_score = {start: 0}
    while open_set:
        _, current = heapq.heappop(open_set)
        if current[0:2] == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return list(reversed(path))
        for nei in graph.get(current, []):
            tentative = g_score[current] + 1
            if tentative < g_score.get(nei, float('inf')):
                g_score[nei] = tentative
                f = tentative + heuristic(nei, (goal[0], goal[1], nei[2]))
                heapq.heappush(open_set, (f, nei))
                came_from[nei] = current
    return None

def generate_intersecting_path(graph, start, goal, guard_routes, t_max):
    pts = [random.choice(route) for route in guard_routes]
    random.shuffle(pts)
    full = [start]
    current = start
    for px, py in pts:
        seg = a_star(graph, current, (px, py), t_max)
        if seg is None:
            return None
        full.extend(seg[1:])
        current = full[-1]
    seg = a_star(graph, current, goal, t_max)
    if seg is None:
        return None
    full.extend(seg[1:])
    return full

def extract_windows(path, env):
    wins = []
    region = []
    t0 = None
    for x, y, t in path:
        if env.is_safe(x, y, t):
            if t0 is None:
                t0 = t
                region = [(x, y)]
            else:
                region.append((x, y))
        else:
            if t0 is not None:
                t1 = t - 1
                length = t1 - t0 + 1
                size = len(set(region))
                if length >= MIN_WIN_TIME and size >= MIN_WIN_SIZE:
                    wins.append({
                        't_start': t0,
                        't_end':   t1,
                        'region':  list(set(region)),
                        'center':  region[len(region)//2],
                        'length':  length,
                        'size':    size
                    })
                t0 = None
    if t0 is not None:
        t1 = path[-1][2]
        length = t1 - t0 + 1
        size = len(set(region))
        if length >= MIN_WIN_TIME and size >= MIN_WIN_SIZE:
            wins.append({
                't_start': t0,
                't_end':   t1,
                'region':  list(set(region)),
                'center':  region[len(region)//2],
                'length':  length,
                'size':    size
            })
    return wins

def validate_reachability(env, windows, graph, player_start):
    reachable = []
    for w in windows:
        seen = {player_start}
        dq = deque([player_start])
        found = False
        while dq and not found:
            cur = dq.popleft()
            for nei in graph.get(cur, []):
                if nei in seen:
                    continue
                seen.add(nei)
                dq.append(nei)
                if w['t_start'] <= nei[2] <= w['t_end'] and (nei[0], nei[1]) in w['region']:
                    found = True
                    break
        if found:
            reachable.append(w)
    return reachable

def intersects_all_guards(path, guard_routes):
    for route in guard_routes:
        s = set(route)
        if not any((x, y) in s for x, y, t in path):
            return False
    return True

def route_uniqueness_score(path):
    total = len(path)
    uniq  = len({(x, y) for x, y, t in path})
    return uniq / total if total > 0 else 0

def dispersion_metrics(windows):
    times = [ (w['t_start'] + w['t_end'])/2 for w in windows ]
    if len(times) < 2:
        t_disp = 0
    else:
        mu   = sum(times)/len(times)
        t_disp = math.sqrt(sum((tt - mu)**2 for tt in times)/len(times))
    centers = [ w['center'] for w in windows ]
    if len(centers) < 2:
        s_disp = 0
    else:
        mx = sum(x for x,y in centers)/len(centers)
        my = sum(y for x,y in centers)/len(centers)
        s_disp = sum(math.hypot(x-mx, y-my) for x,y in centers)/len(centers)
    return t_disp, s_disp

def max_window_metrics(windows):
    if len(windows) <= 2:
        return 0, 0
    mids = windows[1:-1]
    max_len  = max(w['length'] for w in mids)
    max_size = max(w['size']   for w in mids)
    return max_len, max_size

def smoothness_score(path):
    if len(path) < 3:
        return 1.0
    changes = 0
    for i in range(1, len(path)-1):
        dx1 = path[i][0] - path[i-1][0]
        dy1 = path[i][1] - path[i-1][1]
        dx2 = path[i+1][0] - path[i][0]
        dy2 = path[i+1][1] - path[i][1]
        if (dx1, dy1) != (dx2, dy2):
            changes += 1
    return 1.0 - changes / (len(path)-2)

def mutate_path(path, env):
    new = path.copy()
    idx = random.randint(1, len(path)-2)
    x, y, t = path[idx]
    cands = []
    for dx, dy in [(0,1),(1,0),(0,-1),(-1,0),(0,0)]:
        nx, ny = x+dx, y+dy
        if env.is_free(nx, ny) and env.is_safe(nx, ny, t):
            cands.append((nx, ny, t))
    if cands:
        new[idx] = random.choice(cands)
    return new

def optimize_vip_path(init_path, env, static_graph, goal, player_start,
                      safe_graph, generations=30, pop_size=50):
    def fitness(path):
        if not intersects_all_guards(path, env.guard_routes):
            return -1
        wins = extract_windows(path, env)
        reach = validate_reachability(env, wins, safe_graph, player_start)
        cnt       = len(reach)
        t_disp, s_disp = dispersion_metrics(reach)
        r_uniq    = route_uniqueness_score(path)
        max_len, max_size = max_window_metrics(wins)
        # Enforce upper bounds on the duration and area of intermediate windows
        time_score = max(0, (MAX_WIN_TIME - max_len) / MAX_WIN_TIME)
        size_score = max(0, (MAX_WIN_SIZE - max_size) / MAX_WIN_SIZE)
        smooth    = smoothness_score(path)

        return (
            W_WIN_COUNT  * cnt +
            W_TIME_DISP  * t_disp +
            W_SPACE_DISP * s_disp +
            W_ROUTE_UNIQ * r_uniq +
            W_TIME_MAX   * time_score +
            W_SIZE_MAX   * size_score +
            W_SMOOTHNESS * smooth
        )

    population = []
    while len(population) < pop_size:
        p = generate_intersecting_path(static_graph,
                                       player_start,
                                       goal,
                                       env.guard_routes,
                                       env.t_max)
        if p:
            population.append(p)

    for _ in range(generations):
        scored = [(fitness(p), p) for p in population]
        scored.sort(key=lambda x: x[0], reverse=True)
        elites = [p for score,p in scored if score >= 0]
        if not elites:
            elites = [scored[0][1]]
        k = max(1, len(elites)//5)
        elite = elites[:k]

        newpop = elite.copy()
        while len(newpop) < pop_size:
            newpop.append(mutate_path(random.choice(elite), env))
        population = newpop

    best = max(population, key=lambda p: fitness(p))
    return best

def save_path(path, filename):
    data = [{'x':x, 'y':y, 't':t} for x,y,t in path]
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
