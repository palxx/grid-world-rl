#!/usr/bin/env python
import argparse, json
from rtgrid.dataset import write_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--spec", default="mg8", choices=["mg8", "mg19"])
ap.add_argument("--out", required=True)
ap.add_argument("--n-train", type=int, default=300)
ap.add_argument("--n-val", type=int, default=50)
ap.add_argument("--n-test", type=int, default=100)
ap.add_argument("--traj-per-map", type=int, default=8)
ap.add_argument("--test-traj-per-map", type=int, default=0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--holdout-band", type=float, nargs=2, action="append", default=None,
                help="hold out train trajectories whose initial RTG is in [lo, hi] (reward units); repeatable")
a = ap.parse_args()
bands = [(round(lo * 10), round(hi * 10)) for lo, hi in a.holdout_band] if a.holdout_band else None
print(json.dumps(write_dataset(a.spec, a.out, a.n_train, a.n_val, a.n_test, a.traj_per_map, a.seed,
                               a.workers, a.test_traj_per_map, holdout_bands=bands), indent=1))
