"""Behavior policies. Each has reset(layout, planner, rng) and act(env) -> action."""
from __future__ import annotations

from collections import deque

import numpy as np

from . import config as C
from .generator import levels
from .layout import WALL
from .planner import Committed


def _rand_from(rng, s):
    s = sorted(s)
    return s[int(rng.integers(len(s)))]


def choose_target(planner, rng, gap: int = 10):
    """Uniformly pick one of the well separated efficient return levels from the start."""
    fr = planner.frontier()
    lv = levels(fr, gap)
    return lv[int(rng.integers(len(lv)))]


class TargetPlanner:
    """Follows the exact RTG-aware planner (strategy-committed) toward a planned return.
    eps: prob. of uniform random action. temp: Boltzmann temperature over per-action return error (reward units;
    one wrong step costs ~0.2, so temp ~0.05-0.1 gives mildly sub-optimal behavior)."""

    def __init__(self, eps=0.0, temp=None, target=None):
        self.eps, self.temp, self.fixed_target = eps, temp, target
        self.name = f"eps{eps:g}" if eps else f"boltz{temp:g}" if temp else "clean"

    def reset(self, layout, planner, rng):
        self.planner, self.rng = planner, rng
        self.router = Committed(planner, rng)
        self.R = self.fixed_target if self.fixed_target is not None else choose_target(planner, rng)

    def replan(self, env, rng=None):
        rets = sorted(set(self.planner.returns_after(env.pos, env.mask).values()))
        if rets:
            r = (rng or self.rng).integers(len(rets))
            self.R = env.cum + rets[int(r)]

    def act(self, env):
        rng = self.rng
        if self.eps and rng.random() < self.eps:
            return int(rng.integers(4))
        if self.temp:
            A, errs = self.planner.action_set(env.pos, env.mask, self.R - env.cum)
            e = np.array([(x if x is not None else 200) for x in errs], dtype=float) / 10.0
            p = np.exp(-(e - e.min()) / self.temp)
            return int(rng.choice(4, p=p / p.sum()))
        a, _ = self.router.act(env.pos, env.mask, self.R - env.cum)
        return a


def _descend(planner, dist, pos, rng):
    """Random action that strictly decreases dist (falls back to random)."""
    best = []
    cur = dist.get(pos, 10 ** 6)
    for a, (dx, dy) in enumerate(C.ACTION_DELTAS):
        nb = (pos[0] + dx, pos[1] + dy)
        if dist.get(nb, 10 ** 6) < cur:
            best.append(a)
    return best[int(rng.integers(len(best)))] if best else int(rng.integers(4))


class Detour:
    """Walk to a random waypoint, then behave like TargetPlanner from there."""
    name = "detour"

    def reset(self, layout, planner, rng):
        self.planner, self.rng, self.inner = planner, rng, TargetPlanner()
        self.inner.reset(layout, planner, rng)
        cells = [c for c in planner.walk_dist(tuple(layout.start)) if c != planner.goal]
        self.w = cells[int(rng.integers(len(cells)))]
        self.wd = planner.walk_dist(self.w)
        self.arrived = False

    def act(self, env):
        if not self.arrived and env.pos == self.w:
            self.arrived = True
            self.inner.replan(env)
        if not self.arrived:
            return _descend(self.planner, self.wd, env.pos, self.rng)
        return self.inner.act(env)


class Myopic:
    """Greedy nearest-object-first over a random subset, then goal (no global route optimization)."""
    name = "myopic"

    def reset(self, layout, planner, rng):
        self.planner, self.rng = planner, rng
        k = planner.k
        self.subset = [i for i in range(k) if rng.random() < 0.6]

    def act(self, env):
        pl = self.planner
        remaining = [pl.bonus[i] for i in self.subset if not (env.mask >> i) & 1]
        dcur = pl.walk_dist(env.pos)
        if remaining:
            tgt = min(remaining, key=lambda c: dcur.get(c, 10 ** 6))
        else:
            tgt = pl.goal
        return _descend(pl, pl.walk_dist(tgt), env.pos, self.rng)


class Partial:
    """Collect a random non-empty object set via a committed route, then random-walk (never deliberately finishes)."""
    name = "partial"

    def reset(self, layout, planner, rng):
        self.planner, self.rng = planner, rng
        fr = planner.frontier()
        Fs = sorted(F for F in fr if F)
        self.F = Fs[int(rng.integers(len(Fs)))] if Fs else 0
        self.router = Committed(planner, rng)

    def act(self, env):
        if (env.mask & self.F) == self.F:
            return int(self.rng.integers(4))
        acts = self.router.route_actions(self.F, env.pos, env.mask)
        return acts[int(self.rng.integers(len(acts)))] if acts else int(self.rng.integers(4))


class RandomWalk:
    name = "random"

    def reset(self, layout, planner, rng):
        self.rng = rng

    def act(self, env):
        return int(self.rng.integers(4))


class WeakGreedy:
    """Manhattan-greedy toward the goal ignoring walls; 10% random."""
    name = "weak_greedy"

    def reset(self, layout, planner, rng):
        self.goal, self.rng = planner.goal, rng

    def act(self, env):
        if self.rng.random() < 0.1:
            return int(self.rng.integers(4))
        dx, dy = self.goal[0] - env.pos[0], self.goal[1] - env.pos[1]
        if abs(dx) >= abs(dy):
            return C.RIGHT if dx > 0 else C.LEFT
        return C.DOWN if dy > 0 else C.UP


class LavaBlind:
    """Shortest path to the goal that treats lava as ordinary floor."""
    name = "lava_blind"

    def reset(self, layout, planner, rng):
        self.rng = rng
        n, g = layout.size, layout.grid
        goal = planner.goal
        d = {goal: 0}
        q = deque([goal])
        while q:
            c = q.popleft()
            for dx, dy in C.ACTION_DELTAS:
                nb = (c[0] + dx, c[1] + dy)
                if 0 <= nb[0] < n and 0 <= nb[1] < n and nb not in d and g[nb[1], nb[0]] != WALL:
                    d[nb] = d[c] + 1
                    q.append(nb)
        self.d, self.planner = d, planner

    def act(self, env):
        return _descend(self.planner, self.d, env.pos, self.rng)


def make_policy(kind: str, rng) -> object:
    if kind == "clean":
        return TargetPlanner()
    if kind == "eps":
        return TargetPlanner(eps=float(rng.choice([0.05, 0.1, 0.2, 0.3])))
    if kind == "boltz":
        return TargetPlanner(temp=float(rng.choice([0.03, 0.05, 0.1])))
    if kind == "detour":
        return Detour()
    if kind == "myopic":
        return Myopic()
    if kind == "partial":
        return Partial()
    if kind == "random":
        return RandomWalk()
    if kind == "weak_greedy":
        return WeakGreedy()
    if kind == "lava_blind":
        return LavaBlind()
    raise KeyError(kind)
