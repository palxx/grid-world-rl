#!/usr/bin/env python
"""Empirical horizon selection: H = ceil(1.5 * p99 collect-all route length)."""
import argparse, math
import numpy as np
from rtgrid.config import SPECS
from rtgrid.generator import make_layout
from rtgrid.planner import Planner, INF

ap = argparse.ArgumentParser()
ap.add_argument("--spec", default="mg19")
ap.add_argument("--n", type=int, default=500)
a = ap.parse_args()
sp = SPECS[a.spec]
goal_only, all_route, ks = [], [], []
for s in range(a.n):
    lay, _ = make_layout(sp, s)
    p = Planner(lay)
    st = tuple(lay.start)
    goal_only.append(p.steps_F(0, st, 0))
    all_route.append(p.steps_F(p.full, st, 0))
    ks.append(p.k)
frac_inf = np.mean([g >= INF for g in goal_only])
goal_only = [g for g in goal_only if g < INF]
q = lambda x: {f"p{k}": float(np.percentile(x, k)) for k in (5, 50, 90, 99)} | {"max": float(max(x))}
print("spec", a.spec, "maps", a.n, "mean #bonus", np.mean(ks))
print(f"goal-only shortest steps : {q(goal_only)}  (infeasible on {frac_inf:.1%} of maps: objects block all routes)")
print("collect-all route steps  :", q(all_route))
print("suggested horizon        :", math.ceil(1.5 * np.percentile(all_route, 99)), f"(config: {sp.horizon})")
