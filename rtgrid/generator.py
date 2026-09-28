"""Procedural layouts built with MiniGrid's grid/objects (rooms + doorways + obstacles).

Object mapping: Wall -> '#', Goal -> 'G', Lava -> 'L', Ball(yellow) -> coin 'c', Box(purple) -> treasure 'T'.
`make_layout(spec, seed)` deterministically returns the first candidate (sub-seed attempt) that passes the
map-quality filters, so a map seed identifies a layout.
"""
from __future__ import annotations

from collections import deque

import numpy as np
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Box, Goal, Lava, Wall
from minigrid.minigrid_env import MiniGridEnv

from . import config as C
from .config import EnvSpec
from .layout import COINC, FLOOR, GOALC, LAVAC, TREAC, WALL, Layout
from .planner import Planner

_TYPE2CH = {"wall": WALL, "goal": GOALC, "lava": LAVAC, "ball": COINC, "box": TREAC}


def _bfs(grid: Grid, src, blocked_types=("wall", "lava")):
    n = grid.width
    d = {src: 0}
    q = deque([src])
    while q:
        x, y = q.popleft()
        for dx, dy in C.ACTION_DELTAS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < n and (nx, ny) not in d:
                o = grid.get(nx, ny)
                if o is not None and o.type in blocked_types:
                    continue
                d[(nx, ny)] = d[(x, y)] + 1
                q.append((nx, ny))
    return d


class RoomsEnv(MiniGridEnv):
    """Candidate-layout generator. Only `_gen_grid` matters; stepping is done by rtgrid.env."""

    def __init__(self, spec: EnvSpec, **kwargs):
        self.spec = spec
        super().__init__(
            mission_space=MissionSpace(mission_func=lambda: "reach the goal"),
            grid_size=spec.size, max_steps=spec.horizon, see_through_walls=True, **kwargs,
        )

    # -- helpers -------------------------------------------------------------------------------
    def _ri(self, lo, hi):
        return int(self.np_random.integers(lo, hi + 1))

    def _free(self, ex=()):
        n = self.spec.size
        return [(x, y) for y in range(1, n - 1) for x in range(1, n - 1)
                if self.grid.get(x, y) is None and (x, y) not in ex]

    def _gen_grid(self, width, height):
        sp, rng = self.spec, self.np_random
        n = sp.size
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        lx = sp.lines_x_options[self._ri(0, len(sp.lines_x_options) - 1)]
        ly = sp.lines_y_options[self._ri(0, len(sp.lines_y_options) - 1)]
        for x in lx:
            for y in range(height):
                self.grid.set(x, y, Wall())
        for y in ly:
            for x in range(width):
                self.grid.set(x, y, Wall())
        bx = [0, *lx, n - 1]
        by = [0, *ly, n - 1]
        xr = [(bx[i] + 1, bx[i + 1] - 1) for i in range(len(bx) - 1)]
        yr = [(by[j] + 1, by[j + 1] - 1) for j in range(len(by) - 1)]
        nx_, ny_ = len(xr), len(yr)
        # edges between adjacent rooms: (roomA, roomB, candidate door cells)
        edges = []
        for j in range(ny_):
            for i in range(nx_ - 1):
                edges.append(((i, j), (i + 1, j), [(lx[i], y) for y in range(yr[j][0], yr[j][1] + 1)]))
        for j in range(ny_ - 1):
            for i in range(nx_):
                edges.append(((i, j), (i, j + 1), [(x, ly[j]) for x in range(xr[i][0], xr[i][1] + 1)]))
        order = rng.permutation(len(edges))
        parent = {(i, j): (i, j) for i in range(nx_) for j in range(ny_)}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        opened, rest = [], []
        for idx in order:
            a, b, cand = edges[idx]
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
                opened.append(idx)
            else:
                rest.append(idx)
        n_extra = min(self._ri(*sp.n_extra_doors), len(rest))
        opened += rest[:n_extra]
        for idx in opened:
            cand = edges[idx][2]
            x, y = cand[int(rng.integers(len(cand)))]
            self.grid.set(x, y, None)
        # obstacles
        for _ in range(self._ri(*sp.n_obstacles)):
            free = self._free()
            if free:
                x, y = free[int(rng.integers(len(free)))]
                self.grid.set(x, y, Wall())
        free = self._free()
        start = free[int(rng.integers(len(free)))]
        ds = _bfs(self.grid, start)
        far = max(ds.values())
        goal_c = [c for c, d in ds.items() if c != start and d >= sp.min_goal_dist_frac * far
                  and self.grid.get(*c) is None]
        goal = goal_c[int(rng.integers(len(goal_c)))] if goal_c else start
        self.grid.set(*goal, Goal())
        # lava: not adjacent to start
        n_lava = self._ri(*sp.n_lava)
        used = {start, goal}
        for _ in range(n_lava):
            free = [c for c in self._free(used) if abs(c[0] - start[0]) + abs(c[1] - start[1]) > 1 and c in ds]
            if not free:
                break
            c = free[int(rng.integers(len(free)))]
            self.grid.set(*c, Lava())
            used.add(c)
        ds2 = _bfs(self.grid, start)
        dg = _bfs(self.grid, goal, blocked_types=("wall", "lava"))
        base = ds2.get(goal, 10 ** 6)

        def place(k, max_detour, obj):
            for _ in range(k):
                free = [c for c in self._free(used) if c in ds2 and c in dg]
                near = [c for c in free if ds2[c] + dg[c] - base <= max_detour]
                pool = near or free
                if not pool:
                    return
                c = pool[int(rng.integers(len(pool)))]
                self.grid.set(*c, obj())
                used.add(c)

        place(self._ri(*sp.n_coins), 14, lambda: Ball("yellow"))
        place(self._ri(*sp.n_treasures), 40, lambda: Box("purple"))
        self.agent_pos = start
        self.agent_dir = 0


def _extract(env: RoomsEnv) -> Layout:
    n = env.spec.size
    g = np.full((n, n), FLOOR, dtype="<U1")
    for y in range(n):
        for x in range(n):
            o = env.grid.get(x, y)
            if o is not None:
                g[y, x] = _TYPE2CH[o.type]
    return Layout(g, (int(env.agent_pos[0]), int(env.agent_pos[1])))


def levels(frontier: dict, min_gap: int) -> list:
    """Greedy set of feasible returns pairwise separated by >= min_gap."""
    out = []
    for v in sorted(set(frontier.values())):
        if not out or v - out[-1] >= min_gap:
            out.append(v)
    return out


def accept(layout: Layout, spec: EnvSpec):
    """Return (ok, Planner|None). Applies the RTG-design filters from the experiment plan."""
    if len(layout.cells(GOALC)) != 1:
        return False, None
    p = Planner(layout)
    fr = p.frontier()
    if p.full not in fr:  # collect-all route must exist
        return False, None
    if len(levels(fr, spec.min_level_gap)) < spec.min_levels:
        return False, None
    for i in range(p.k):  # every bonus object must matter for some strategy
        ok = any((F | (1 << i)) in fr and fr[F | (1 << i)] - fr[F] >= spec.min_net_gain
                 for F in fr if not (F >> i) & 1)
        if not ok:
            return False, None
    return True, p


_ENVS = {}


def make_layout(spec: EnvSpec, seed: int, max_attempts: int = 500):
    """Deterministic: first accepted candidate for sub-seeds seed*10007 + attempt."""
    env = _ENVS.get(spec.name)
    if env is None:
        env = _ENVS[spec.name] = RoomsEnv(spec)
    for attempt in range(max_attempts):
        env.reset(seed=seed * 10007 + attempt)
        lay = _extract(env)
        ok, _ = accept(lay, spec)
        if ok:
            return lay, attempt
    raise RuntimeError(f"no acceptable layout for seed {seed} within {max_attempts} attempts")
