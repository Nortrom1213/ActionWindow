import numpy as np
from collections import defaultdict    # noqa: F401  (import retained for potential downstream use)

class GridEnvironment:
    """
    A discrete, time-expanded grid world in which mobile guards patrol fixed
    routes.  A cell (x, y, t) is considered *safe* if it is not a static
    obstacle and lies outside every guard’s vision radius at simulation step t.
    """

    def __init__(self, width, height, obstacles, guard_routes,
                 vision_radius, t_max):
        self.width         = width
        self.height        = height
        self.obstacles     = obstacles          # np.ndarray(bool) with shape (W, H)
        self.guard_routes  = guard_routes       # list[list[(x, y)]]
        self.vision_radius = vision_radius
        self.t_max         = t_max

    # --------------------------------------------------------------------- #
    # Low-level spatial predicates                                           #
    # --------------------------------------------------------------------- #
    def is_free(self, x: int, y: int) -> bool:
        """Return True iff (x, y) is inside the map and not a static obstacle."""
        return (0 <= x < self.width and
                0 <= y < self.height and
                not self.obstacles[x, y])

    def guard_pos(self, guard_idx: int, t: int):
        """Deterministically compute the position of guard *guard_idx* at time t."""
        route = self.guard_routes[guard_idx]
        return route[t % len(route)]

    def is_safe(self, x: int, y: int, t: int) -> bool:
        """
        A cell is *safe* when it is neither a static obstacle nor within any
        guard’s Manhattan-distance vision radius at time t.
        """
        if not self.is_free(x, y):
            return False
        for i in range(len(self.guard_routes)):
            gx, gy = self.guard_pos(i, t)
            if abs(gx - x) + abs(gy - y) <= self.vision_radius:
                return False
        return True

    def is_in_guard_view(self, guard_idx: int, x: int, y: int, t: int) -> bool:
        """
        Query whether cell (x, y) lies within the vision radius of guard
        *guard_idx* at simulation step t.
        """
        gx, gy = self.guard_pos(guard_idx, t)
        return abs(gx - x) + abs(gy - y) <= self.vision_radius

    # --------------------------------------------------------------------- #
    # Time-expanded graph construction                                       #
    # --------------------------------------------------------------------- #
    def build_time_expanded_graph(self):
        """
        Construct an adjacency list for the space–time graph in which nodes
        are (x, y, t) triples that are *safe*.  A directed edge connects a
        node at time t to all spatio-temporally adjacent nodes at time t+1,
        including the option of remaining stationary.
        """
        nodes = {}  # first collect all safe nodes
        for t in range(self.t_max):
            for x in range(self.width):
                for y in range(self.height):
                    if self.is_safe(x, y, t):
                        nodes[(x, y, t)] = []

        # add temporal edges (4-neighbourhood + wait action)
        for (x, y, t) in list(nodes.keys()):
            if t == self.t_max - 1:
                continue
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)]:
                nx, ny = x + dx, y + dy
                if (nx, ny, t + 1) in nodes:
                    nodes[(x, y, t)].append((nx, ny, t + 1))
        return nodes

    def build_static_time_expanded_graph(self):
        """
        Construct a space–time graph *ignoring* guard visibility; only static
        obstacles are treated as impassable.  The return format matches
        :meth:`build_time_expanded_graph`.
        """
        nodes = {}
        for t in range(self.t_max):
            for x in range(self.width):
                for y in range(self.height):
                    if self.is_free(x, y):          # static obstacles only
                        nodes[(x, y, t)] = []

        for (x, y, t) in list(nodes.keys()):
            if t == self.t_max - 1:
                continue
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)]:
                nx, ny = x + dx, y + dy
                if (nx, ny, t + 1) in nodes:
                    nodes[(x, y, t)].append((nx, ny, t + 1))
        return nodes

    def build_safe_time_expanded_graph(self):
        """
        Equivalent to :meth:`build_time_expanded_graph`; retained for semantic
        clarity when both static obstacles and guard vision are deemed unsafe.
        """
        return self.build_time_expanded_graph()

# ----------------------------------------------------------------------------- #
# Convenience constructors for exemplar testbeds                                #
# ----------------------------------------------------------------------------- #
def build_complex_env() -> GridEnvironment:
    """
    Create a 20 × 20 benchmark map with three patrolling guards.
    Cells with value 1 represent obstacles; 0 denotes traversable space.
    Guard patrol paths are specified explicitly as ordered lists of grid
    coordinates.
    """
    # --- 20 × 20 maze: 1 = obstacle, 0 = free space ----------------------
    maze = [
        [1]*20,
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,1,1,1,0,0,0,1,0,0,1,1,1,0,0,1,0,1],
        [1,0,0,1,1,1,0,0,0,1,0,0,1,1,1,0,0,1,0,1],
        [1,0,0,1,1,1,0,0,0,1,0,0,1,1,1,0,0,1,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1],
        [1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,1,1,1,0,0,0,1,0,0,1,1,1,0,0,1,0,1],
        [1,0,0,1,1,1,0,0,0,1,0,0,1,1,1,0,0,1,0,1],
        [1,0,0,1,1,1,0,0,0,1,0,0,1,1,1,0,0,1,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1]*20
    ]
    obstacles = (np.array(maze, dtype=int) == 1)

    # --- Guard patrol definitions (cyclic) --------------------------------
    # Guard 1: (9,1) → (9,15) → (11,15) → (11,1) → …
    guard1 = []
    guard1 += [(9,  y) for y in range( 1, 16)]
    guard1 += [(x, 15) for x in range(10, 12)]
    guard1 += [(11, y) for y in range(14,  0, -1)]
    guard1 += [(x,  1) for x in range(10,  8, -1)]

    # Guard 2: (13,2) → (13,6) → (17,6) → (17,2) → …
    guard2 = []
    guard2 += [(13, y) for y in range(2, 7)]
    guard2 += [(x,  6) for x in range(14, 18)]
    guard2 += [(17, y) for y in range(5, 1, -1)]
    guard2 += [(x,  2) for x in range(16, 12, -1)]

    # Guard 3: (3,18) → (16,18) → (3,18) (back-and-forth)
    guard3 = []
    guard3 += [(x, 18) for x in range( 3, 17)]
    guard3 += [(x, 18) for x in range(16,  2, -1)]

    guard_routes   = [guard1, guard2, guard3]
    vision_radius  = 2
    t_max          = 80
    W, H           = obstacles.shape
    return GridEnvironment(W, H, obstacles, guard_routes,
                           vision_radius, t_max)

def build_large_env() -> GridEnvironment:
    """
    Construct a 100 × 100 environment with four extensive guard patrols,
    enlarged guard vision, and an extended planning horizon (t_max = 300).
    The maze layout consists of nested rectangular corridors and central
    blocking structures.
    """
    # initialise empty maze (0 = free space)
    maze = [[0 for _ in range(100)] for _ in range(100)]

    # outer boundary walls
    for i in range(1, 99):
        maze[i][0]  = maze[i][99] = 1
        maze[0][i]  = maze[99][i] = 1

    # secondary rectangular frame
    for i in range(19, 79):
        maze[9][i]  = maze[89][i] = 1
        maze[i][9]  = maze[i][89] = 1

    # four interior obstacle blocks
    for i in range(19, 39):
        maze[i][49] = maze[49][i] = 1
        for j in range(19, 39):
            maze[i][j] = 1

    for i in range(19, 39):
        for j in range(59, 79):
            maze[i][j] = 1

    for i in range(59, 79):
        maze[i][49] = maze[49][i] = 1
        for j in range(19, 39):
            maze[i][j] = 1

    for i in range(59, 79):
        for j in range(59, 79):
            maze[i][j] = 1

    # central solid block
    for i in range(44, 54):
        for j in range(44, 54):
            maze[i][j] = 1

    obstacles = (np.array(maze, dtype=int) == 1)

    # Guard patrols along rectangular loops of varying size
    guard1 = []
    guard1 += [(43, y) for y in range(43, 55)]
    guard1 += [(x, 55) for x in range(43, 55)]
    guard1 += [(55, y) for y in range(55, 43, -1)]
    guard1 += [(x, 43) for x in range(55, 43, -1)]

    guard2 = []
    guard2 += [( 8, y) for y in range( 8, 90)]
    guard2 += [(x, 90) for x in range( 8, 90)]
    guard2 += [(90, y) for y in range(90,  8, -1)]
    guard2 += [(x,  8) for x in range(90,  8, -1)]

    guard3 = []
    guard3 += [(18, y) for y in range(18, 40)]
    guard3 += [(x, 40) for x in range(18, 40)]
    guard3 += [(40, y) for y in range(40, 18, -1)]
    guard3 += [(x, 18) for x in range(40, 18, -1)]

    guard4 = []
    guard4 += [(58, y) for y in range(58, 80)]
    guard4 += [(x, 80) for x in range(58, 80)]
    guard4 += [(80, y) for y in range(80, 58, -1)]
    guard4 += [(x, 58) for x in range(80, 58, -1)]

    guard_routes = [guard1, guard2, guard3, guard4]

    return GridEnvironment(
        width=100,
        height=100,
        obstacles=obstacles,
        guard_routes=guard_routes,
        vision_radius=5,   # enlarged vision radius
        t_max=300          # extended temporal horizon
    )
