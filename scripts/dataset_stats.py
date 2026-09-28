#!/usr/bin/env python
"""Dataset diagnostics: outcomes, return spread, RTG-sensitive decision fraction, token budget."""
import argparse, collections, json
import numpy as np
from rtgrid.dataset import read_jsonl
from rtgrid.generator import levels
from rtgrid.layout import Layout
from rtgrid.planner import Planner
from rtgrid.trajectory import Trajectory

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--split", default="train")
ap.add_argument("--max-trajs", type=int, default=3000)
ap.add_argument("--sens-sample", type=int, default=400, help="#supervised decisions to test for RTG sensitivity")
a = ap.parse_args()
maps = {m["seed"]: m for m in read_jsonl(f"{a.data}/maps_{a.split}.jsonl")}
planners = {}
by = collections.defaultdict(lambda: collections.Counter())
rets, lens, n_dec, sup = [], [], 0, 0
per_map_levels = collections.defaultdict(set)
sample = []
rng = np.random.default_rng(0)
for i, d in enumerate(read_jsonl(f"{a.data}/trajs_{a.split}.jsonl")):
    if i >= a.max_trajs:
        break
    t = Trajectory.from_dict(d)
    by[(t.source, t.policy if t.source == "hindsight" else "")][t.end] += 1
    rets.append(t.ret / 10); lens.append(len(t.steps))
    per_map_levels[t.map_seed].add(t.steps[0].rtg)
    for k, s in enumerate(t.steps):
        n_dec += 1
        if s.supervise:
            sup += 1
            if len(sample) < 20000:
                sample.append((t.map_seed, s.pos, s.mask, t.steps[0].rtg - 0, t, k))
print(f"trajs {len(lens)}  mean steps {np.mean(lens):.1f}  p95 {np.percentile(lens,95):.0f}  supervised decisions {sup}/{n_dec}")
print(f"return: mean {np.mean(rets):.2f} std {np.std(rets):.2f}  p5/p50/p95 {np.percentile(rets,5):.1f}/{np.percentile(rets,50):.1f}/{np.percentile(rets,95):.1f}")
tot = collections.Counter()
print("outcomes by (source, policy):")
for k, c in sorted(by.items()):
    n = sum(c.values()); tot.update(c)
    print(f"  {k[0]:9s}{k[1]:12s} n={n:5d}  " + "  ".join(f"{e}={v/n:.0%}" for e, v in sorted(c.items())))
n = sum(tot.values()); print("overall:", {e: f"{v/n:.1%}" for e, v in tot.items()})
print(f"distinct initial RTGs per map: mean {np.mean([len(v) for v in per_map_levels.values()]):.1f}")
# RTG-sensitive decisions: exist two well-separated levels with disjoint optimal action sets at the same state/history offset
hits = tested = 0
for j in rng.permutation(len(sample))[:a.sens_sample]:
    seed, pos, mask, _, t, k = sample[j]
    if seed not in planners:
        lay = Layout.from_dict(maps[seed]["layout"]); planners[seed] = (Planner(lay), maps[seed]["levels"])
    pl, lv = planners[seed]
    cum = t.steps[0].rtg - t.steps[k].rtg  # return collected so far under this trajectory's RTG bookkeeping
    sets = [pl.action_set(pos, mask, L - cum)[0] for L in lv]
    sets = [s for s in sets if s]
    tested += 1
    hits += any(not (x & y) for i, x in enumerate(sets) for y in sets[i + 1:])
print(f"RTG-sensitive supervised decisions (some pair of levels needs disjoint actions): {hits}/{tested} = {hits/max(tested,1):.0%}")
