"""
Pathfinding Algorithms in GridWorld
-----------------------------------
This script is kind of a sandbox to compare BFS, UCS, A*, and
a dynamic replanning approach when obstacles are moving.

Features:
- Custom GridWorld with static + moving obstacles
- Multiple search algorithms tested
- Auto replan when dynamic blocks show up
- Basic runtime + cost + expansion stats
- Quick plots + CSV output for results

Author: (Your Name)  <-- yeah, don’t forget to actually fill this in ;)
"""

import heapq
import random
import time
from collections import deque

# I like pandas/matplotlib for quick analysis
import pandas as pd
import matplotlib.pyplot as plt


# ==============================================================
#  ENVIRONMENT: GRID WORLD
# ==============================================================

class GridWorld:
    def __init__(self, grid, start, goal, dynamic_obstacles=None):
        """
        grid: 2D list (1=normal free cell, -1=wall, >1=extra cost)
        start: (row, col)
        goal: (row, col)
        dynamic_obstacles: list of cells that will "move around"
        """
        self.grid = grid
        self.start = start
        self.goal = goal
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.dynamic_obstacles = dynamic_obstacles if dynamic_obstacles else []

    # --- Helpers ---
    def in_bounds(self, pos):
        r, c = pos
        return (0 <= r < self.rows) and (0 <= c < self.cols)

    def passable(self, pos):
        # Important: avoid walking into walls or into moving blocks
        return (pos not in self.dynamic_obstacles) and (self.grid[pos[0]][pos[1]] != -1)

    def cost(self, pos):
        # Default cost is whatever’s in the grid (usually 1)
        return self.grid[pos[0]][pos[1]]

    def neighbors(self, pos):
        """Check N/S/E/W moves (no diagonals here)."""
        r, c = pos
        steps = [(1,0), (-1,0), (0,1), (0,-1)]
        out = []
        for dr, dc in steps:
            nxt = (r+dr, c+dc)
            if self.in_bounds(nxt) and self.passable(nxt):
                out.append(nxt)
        return out

    def move_obstacles(self):
        """Shift dynamic obstacles randomly. Note: this can trap the agent sometimes."""
        new_obs = []
        for (r, c) in self.dynamic_obstacles:
            candidates = [(r+1,c), (r-1,c), (r,c+1), (r,c-1)]
            candidates = [p for p in candidates if self.in_bounds(p) and self.grid[p[0]][p[1]] != -1]
            if candidates:
                new_obs.append(random.choice(candidates))
            else:
                new_obs.append((r, c))  # stuck in place
        self.dynamic_obstacles = new_obs


# ==============================================================
#  SEARCH ALGORITHMS
# ==============================================================

def bfs(world):
    start_time = time.time()
    start, goal = world.start, world.goal

    frontier = deque([start])
    came_from = {start: None}
    expanded = 0

    while frontier:
        current = frontier.popleft()
        expanded += 1
        if current == goal:
            break
        for nxt in world.neighbors(current):
            if nxt not in came_from:
                frontier.append(nxt)
                came_from[nxt] = current

    path = reconstruct_path(came_from, start, goal)
    return metrics("BFS", path, expanded, time.time() - start_time, world)


def uniform_cost(world):
    start_time = time.time()
    start, goal = world.start, world.goal

    # Note: using heapq as priority queue (classic Dijkstra style)
    frontier = [(0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}
    expanded = 0

    while frontier:
        curr_cost, current = heapq.heappop(frontier)
        expanded += 1
        if current == goal:
            break
        for nxt in world.neighbors(current):
            new_cost = curr_cost + world.cost(nxt)
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                heapq.heappush(frontier, (new_cost, nxt))
                came_from[nxt] = current

    path = reconstruct_path(came_from, start, goal)
    return metrics("UCS", path, expanded, time.time() - start_time, world)


def heuristic(a, b):
    # Manhattan distance (could swap to Euclidean for fun later)
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(world, start=None, goal=None):
    start_time = time.time()
    start = start or world.start
    goal = goal or world.goal

    frontier = [(0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}
    expanded = 0

    while frontier:
        _, current = heapq.heappop(frontier)
        expanded += 1
        if current == goal:
            break
        for nxt in world.neighbors(current):
            new_cost = cost_so_far[current] + world.cost(nxt)
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                priority = new_cost + heuristic(nxt, goal)
                heapq.heappush(frontier, (priority, nxt))
                came_from[nxt] = current

    path = reconstruct_path(came_from, start, goal)
    return metrics("A*", path, expanded, time.time() - start_time, world)


def dynamic_replan(world):
    start_time = time.time()
    current = world.start
    full_path = [current]
    expanded_total = 0

    while current != world.goal:
        # Always re-run A* from wherever we are
        result = astar(world, start=current)
        path = result["path"]
        expanded_total += result["nodes_expanded"]

        if not path:
            print("No path! Agent trapped :(")
            return metrics("Dynamic Replan", full_path, expanded_total, time.time() - start_time, world)

        for step in path[1:]:
            world.move_obstacles()  # world changes before moving
            if step in world.dynamic_obstacles:
                print(f"Blocked at {step}, need to replan...")
                break
            else:
                current = step
                full_path.append(current)
            if current == world.goal:
                return metrics("Dynamic Replan", full_path, expanded_total, time.time() - start_time, world)

    return metrics("Dynamic Replan", full_path, expanded_total, time.time() - start_time, world)


# ==============================================================
#  UTILITIES
# ==============================================================

def reconstruct_path(came_from, start, goal):
    if goal not in came_from:
        return None
    node = goal
    path = []
    while node != start:
        path.append(node)
        node = came_from[node]
    path.append(start)
    path.reverse()
    return path

def path_cost(path, world):
    if not path:
        return None
    total = 0
    for p in path:   # less optimal than sum() but more explicit
        total += world.cost(p)
    return total

def metrics(name, path, expanded, runtime, world):
    return {
        "algorithm": name,
        "path": path,
        "path_cost": path_cost(path, world),
        "nodes_expanded": expanded,
        "runtime": round(runtime, 6)
    }


# ==============================================================
#  RESULTS + PLOTS
# ==============================================================

def analyze_results(all_results, save_csv=True):
    df = pd.DataFrame(all_results)
    print("\n=== Results Table ===")
    print(df[["map", "algorithm", "path_cost", "nodes_expanded", "runtime"]])

    if save_csv:
        df.to_csv("results.csv", index=False)
        print("Saved results to results.csv")

    # very simple bar plots
    for metric in ["runtime", "path_cost", "nodes_expanded"]:
        df.pivot(index="algorithm", columns="map", values=metric).plot(kind="bar", title=metric)
        plt.ylabel(metric)
        plt.show()


# ==============================================================
#  MAIN
# ==============================================================

if __name__ == "__main__":
    maps = {}

    # Small test map
    maps["Small"] = GridWorld(
        grid=[[1,1,1],[1,-1,1],[1,1,1]],
        start=(0,0), goal=(2,2)
    )

    # Medium 5x5
    maps["Medium"] = GridWorld(
        grid=[
            [1,1,1,1,1],
            [1,-1,-1,1,1],
            [1,1,1,-1,1],
            [1,-1,1,1,1],
            [1,1,1,1,1],
        ],
        start=(0,0), goal=(4,4)
    )

    # Bigger 10x10 with a wall row (but a gap at the end)
    big = [[1]*10 for _ in range(10)]
    for i in range(10):
        big[5][i] = -1
    big[5][9] = 1
    maps["Large"] = GridWorld(grid=big, start=(0,0), goal=(9,9))

    # Dynamic obstacle map
    maps["Dynamic"] = GridWorld(
        grid=[
            [1,1,1,1,1],
            [1,-1,1,-1,1],
            [1,1,1,1,1],
            [1,-1,1,-1,1],
            [1,1,1,1,1],
        ],
        start=(0,0), goal=(4,4),
        dynamic_obstacles=[(2,2)]
    )

    results = run_experiments(maps)
    analyze_results(results)
