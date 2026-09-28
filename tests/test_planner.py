"""Planner vs. an independent brute-force BFS over the real environment dynamics."""
from collections import deque

import numpy as np

from rtgrid import config as C
from rtgrid.env import CardinalEnv
from rtgrid.policies import TargetPlanner
from rtgrid.trajectory import rollout_hindsight


def brute_returns(lay, pos, mask, horizon=10 ** 4):
    """{F: return-to-go} by BFS over (pos, mask) using CardinalEnv.step, forbidding bumps and lava."""
    env = CardinalEnv(lay, horizon)
    seen = {(pos, mask): 0}
    q = deque([(pos, mask)])
    best = {}
    while q:
        p, m = q.popleft()
        d = seen[(p, m)]
        for a in range(4):
            env.pos, env.mask, env.t, env.done, env.cum = p, m, 0, False, 0
            res = env.step(a)
            if res.bump or res.event == "lava":
                continue
            if res.event == "goal":
                F = env.mask
                v = sum(C.COIN if lay.grid[y, x] == "c" else C.TREASURE
                        for i, (x, y) in enumerate(lay.bonus) if (F >> i) & 1 and not (mask >> i) & 1)
                # return-to-go = bonus collected on the way + goal + step costs
                best[F] = min(best.get(F, 10 ** 9), d + 1)
                continue
            key = (env.pos, env.mask)
            if key not in seen:
                seen[key] = d + 1
                q.append(key)
    out = {}
    for F, steps in best.items():
        val = sum(C.COIN if lay.grid[y, x] == "c" else C.TREASURE
                  for i, (x, y) in enumerate(lay.bonus) if (F >> i) & 1 and not (mask >> i) & 1)
        out[F] = val + C.GOAL + C.STEP * steps
    return out


def test_frontier_matches_bruteforce(maps):
    for s, lay, pl in maps:
        assert pl.frontier() == brute_returns(lay, tuple(lay.start), 0), s


def test_returns_and_action_returns_from_random_states(maps):
    rng = np.random.default_rng(0)
    for s, lay, pl in maps[:6]:
        cells = [c for c in pl.walk_dist(tuple(lay.start)) if c != pl.goal]  # goal is terminal
        for _ in range(6):
            pos = cells[int(rng.integers(len(cells)))]
            mask = int(rng.integers(1 << pl.k)) if pl.k else 0
            # mask must be consistent with position (agent stands on a collected object)
            if pos in pl.bonus_idx:
                mask |= 1 << pl.bonus_idx[pos]
            assert pl.returns_after(pos, mask) == brute_returns(lay, pos, mask), (s, pos, mask)
            # per-action returns == step then brute force from the next state
            env = CardinalEnv(lay, 10 ** 4)
            for a, gs in enumerate(pl.action_returns(pos, mask)):
                env.pos, env.mask, env.t, env.done, env.cum = pos, mask, 0, False, 0
                res = env.step(a)
                if res.bump or res.event == "lava":
                    assert gs == {}
                elif res.event == "goal":
                    assert gs == {mask: C.STEP + C.GOAL}
                else:
                    want = {F: res.reward + g for F, g in brute_returns(lay, env.pos, env.mask).items()}
                    assert gs == want, (s, pos, mask, a)


def test_clean_planner_hits_target_exactly(maps):
    for s, lay, pl in maps:
        rng = np.random.default_rng(s)
        for R in sorted(set(pl.frontier().values())):  # ALL efficient levels, incl. ties between strategies
            tr = rollout_hindsight(lay, pl, 10 ** 4, TargetPlanner(target=R), rng, s)
            assert tr.end == "goal" and tr.ret == R
            assert all(st.opt >> st.action & 1 for st in tr.steps)  # executed action is RTG-optimal
            assert tr.ret == sum(st.reward for st in tr.steps)
