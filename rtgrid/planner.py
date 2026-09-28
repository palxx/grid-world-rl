"""Exact planner for efficient (shortest-route-per-object-set) behavior.

An *efficient continuation* from state s=(pos, mask) is defined by a final collected set F ⊇ mask:
walk the shortest route that collects exactly the objects in F\\mask (objects outside F are avoided,
lava is avoided, the goal is only entered last, no wall bumps). Its return-to-go is

    ret_F(s) = value(F \\ mask) + GOAL + STEP * steps_F(s)

The set of achievable efficient returns from s is G(s) = {ret_F(s)}; per action G(s, a) is defined
analogously. A(s, R) is the set of actions whose best efficient return is closest to R.
"""
from __future__ import annotations

from collections import deque
from functools import lru_cache

from . import config as C
from .layout import COINC, GOALC, LAVAC, TREAC, WALL, Layout

INF = 10 ** 6


class Planner:
    def __init__(self, layout: Layout):
        self.layout = layout
        self.n = layout.size
        self.goal = layout.goal
        self.bonus = layout.bonus
        self.k = len(self.bonus)
        self.vals = [C.COIN if layout.grid[y, x] == COINC else C.TREASURE for x, y in self.bonus]
        self.bonus_idx = {p: i for i, p in enumerate(self.bonus)}
        self.lava = set(layout.lava)
        g = layout.grid
        n = self.n
        self._walk = {}
        for y in range(n):
            for x in range(n):
                if g[y, x] == WALL or (x, y) in self.lava:
                    continue
                nb = []
                for dx, dy in C.ACTION_DELTAS:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < n and 0 <= ny < n and g[ny, nx] != WALL and (nx, ny) not in self.lava:
                        nb.append((nx, ny))
                self._walk[(x, y)] = nb
        self._dist_cache = {}
        self._W_cache = {}
        self.full = (1 << self.k) - 1

    # ---- distances -------------------------------------------------------------------------
    def _dist(self, F: int, src: tuple) -> dict:
        key = (F, src)
        d = self._dist_cache.get(key)
        if d is not None:
            return d
        blocked = {p for i, p in enumerate(self.bonus) if not (F >> i) & 1}
        d = {src: 0}
        q = deque([src])
        while q:
            c = q.popleft()
            if c == self.goal and c != src:
                continue  # goal is terminal: never traverse through it
            for nb in self._walk[c]:
                if nb in d or nb in blocked:
                    continue
                d[nb] = d[c] + 1
                q.append(nb)
        self._dist_cache[key] = d
        return d

    def _W(self, F: int, S: int, j: int) -> int:
        """Min steps from object j, visiting all objects in S (⊆F, j∉S), then ending at the goal."""
        key = (F, S, j)
        v = self._W_cache.get(key)
        if v is not None:
            return v
        dj = self._dist(F, self.bonus[j])
        if S == 0:
            v = dj.get(self.goal, INF)
        else:
            v = INF
            for i in range(self.k):
                if (S >> i) & 1:
                    step = dj.get(self.bonus[i], INF)
                    if step >= INF:
                        continue
                    v = min(v, step + self._W(F, S ^ (1 << i), i))
        self._W_cache[key] = v
        return v

    def steps_F(self, F: int, pos: tuple, mask: int) -> int:
        rem = F & ~mask
        if rem == 0:
            return self._dist(F, self.goal).get(pos, INF)
        best = INF
        for j in range(self.k):
            if (rem >> j) & 1:
                dj = self._dist(F, self.bonus[j]).get(pos, INF)
                if dj >= INF:
                    continue
                best = min(best, dj + self._W(F, rem ^ (1 << j), j))
        return best

    def value(self, S: int) -> int:
        return sum(self.vals[i] for i in range(self.k) if (S >> i) & 1)

    # ---- returns ---------------------------------------------------------------------------
    def returns_after(self, pos: tuple, mask: int) -> dict:
        """{F: ret_F(s)} over efficient continuations from state (pos, mask)."""
        out = {}
        for F in range(1 << self.k):
            if F & mask != mask:
                continue
            st = self.steps_F(F, pos, mask)
            if st < INF:
                out[F] = self.value(F & ~mask) + C.GOAL + C.STEP * st
        return out

    def frontier(self) -> dict:
        """Efficient feasible returns from the start: {F: return}."""
        return self.returns_after(tuple(self.layout.start), 0)

    def action_returns(self, pos: tuple, mask: int) -> list:
        """Per action a: {F: return-to-go incl. this step's reward} of efficient continuations
        (empty dict if the action bumps, hits lava, or leaves no route)."""
        res = []
        for a in range(4):
            dx, dy = C.ACTION_DELTAS[a]
            nxt = (pos[0] + dx, pos[1] + dy)
            if nxt not in self._walk:  # wall / oob / lava
                res.append({})
                continue
            if nxt == self.goal:
                res.append({mask: C.STEP + C.GOAL})
                continue
            m2 = mask
            r = C.STEP
            if nxt in self.bonus_idx and not (mask >> self.bonus_idx[nxt]) & 1:
                i = self.bonus_idx[nxt]
                m2 |= 1 << i
                r += self.vals[i]
            res.append({F: r + g for F, g in self.returns_after(nxt, m2).items()})
        return res

    def action_set(self, pos: tuple, mask: int, rtg: int):
        """(A(s,R), per-action error) with A = actions whose closest efficient return is nearest R.
        Returns (empty set, errs) if no efficient action exists."""
        errs = []
        for gs in self.action_returns(pos, mask):
            errs.append(min((abs(g - rtg) for g in gs.values()), default=None))
        valid = [e for e in errs if e is not None]
        if not valid:
            return set(), errs
        m = min(valid)
        return {a for a, e in enumerate(errs) if e == m}, errs

    def walk_dist(self, src: tuple) -> dict:
        """Plain BFS over non-wall, non-lava cells (goal is leaf, objects passable)."""
        d = {src: 0}
        q = deque([src])
        while q:
            c = q.popleft()
            if c == self.goal and c != src:
                continue
            for nb in self._walk[c]:
                if nb not in d:
                    d[nb] = d[c] + 1
                    q.append(nb)
        return d


class Committed:
    """Strategy-committed RTG-aware router (avoids oscillation between tied strategies).

    Keeps one planned final object set F and follows a shortest route for it (steps_F strictly decreases, so no
    cycles). F is only changed if it becomes invalid (an object outside F was collected) or a strictly closer
    match to the commanded return exists (e.g. after noise/bumps). Every action it returns lies in
    Planner.action_set(s, R) (the union over all tied strategies used for scoring)."""

    def __init__(self, planner: Planner, rng):
        self.pl, self.rng, self.F = planner, rng, None

    def route_actions(self, F: int, pos: tuple, mask: int) -> list:
        pl = self.pl
        cur = pl.steps_F(F, pos, mask)
        out = []
        for a in range(4):
            dx, dy = C.ACTION_DELTAS[a]
            nxt = (pos[0] + dx, pos[1] + dy)
            if nxt not in pl._walk:
                continue
            if nxt == pl.goal:
                if (F & ~mask) == 0 and cur == 1:
                    out.append(a)
                continue
            m2 = mask | (1 << pl.bonus_idx[nxt]) if nxt in pl.bonus_idx else mask
            if m2 & ~F:
                continue
            if pl.steps_F(F, nxt, m2) == cur - 1:
                out.append(a)
        return out

    def select(self, pos: tuple, mask: int, rtg: int):
        rets = self.pl.returns_after(pos, mask)
        if not rets:
            return None
        best = min(abs(g - rtg) for g in rets.values())
        if self.F in rets and abs(rets[self.F] - rtg) == best:
            return self.F
        cands = sorted(F for F, g in rets.items() if abs(g - rtg) == best)
        self.F = cands[int(self.rng.integers(len(cands)))]
        return self.F

    def act(self, pos: tuple, mask: int, rtg: int):
        """(action, ok). ok=False if no efficient route exists (caller falls back to random)."""
        F = self.select(pos, mask, rtg)
        if F is None:
            return int(self.rng.integers(4)), False
        acts = self.route_actions(F, pos, mask)
        if not acts:
            return int(self.rng.integers(4)), False
        return acts[int(self.rng.integers(len(acts)))], True
