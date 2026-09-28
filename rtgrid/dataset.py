"""Split-by-map dataset construction with dihedral-hash dedupe and optional RTG-band holdout."""
from __future__ import annotations

import json
import os
from multiprocessing import Pool

import numpy as np

from .config import SPECS
from .generator import levels, make_layout
from .layout import canonical_hash
from .planner import Planner
from .trajectory import MIXTURE, generate_for_layout

SEED_BASE = {"train": 0, "val": 10_000_000, "test": 20_000_000}


def build_splits(spec, n_train: int, n_val: int, n_test: int):
    """Disjoint seed ranges AND disjoint canonical layout hashes (rotations/reflections count as duplicates)."""
    seen, out = set(), {}
    for split, n in (("train", n_train), ("val", n_val), ("test", n_test)):
        recs, s = [], SEED_BASE[split]
        while len(recs) < n:
            lay, _ = make_layout(spec, s)
            h = canonical_hash(lay)
            if h not in seen:
                seen.add(h)
                recs.append((s, h))
            s += 1
        out[split] = recs
    return out


def _work(args):
    spec_name, seed, h, n_traj, data_seed, mixture, bands = args
    spec = SPECS[spec_name]
    lay, _ = make_layout(spec, seed)
    pl = Planner(lay)
    fr = pl.frontier()
    mrec = {"seed": seed, "hash": h, "layout": lay.to_dict(),
            "frontier": sorted(fr.values()), "levels": levels(fr, spec.min_level_gap)}
    trajs = []
    if n_traj:
        rng = np.random.default_rng([data_seed, seed])
        for t in generate_for_layout(lay, pl, spec.horizon, rng, n_traj, seed, mixture):
            r0 = t.steps[0].rtg
            if bands and any(lo <= r0 <= hi for lo, hi in bands):
                continue
            trajs.append(t.to_dict())
    return mrec, trajs


def write_dataset(spec_name, out_dir, n_train, n_val, n_test, traj_per_map, data_seed=0, workers=4,
                  test_traj_per_map=0, val_traj_per_map=None, mixture=MIXTURE, holdout_bands=None):
    spec = SPECS[spec_name]
    os.makedirs(out_dir, exist_ok=True)
    splits = build_splits(spec, n_train, n_val, n_test)
    tpm = {"train": traj_per_map, "val": traj_per_map if val_traj_per_map is None else val_traj_per_map,
           "test": test_traj_per_map}
    summary = {"spec": spec_name, "horizon": spec.horizon, "holdout_bands": holdout_bands, "splits": {}}
    with Pool(workers) as pool:
        for split, recs in splits.items():
            jobs = [(spec_name, s, h, tpm[split], data_seed, mixture,
                     holdout_bands if split == "train" else None) for s, h in recs]
            n_tr = 0
            with open(f"{out_dir}/maps_{split}.jsonl", "w") as fm, open(f"{out_dir}/trajs_{split}.jsonl", "w") as ft:
                for mrec, trajs in pool.imap(_work, jobs, chunksize=4):
                    fm.write(json.dumps(mrec) + "\n")
                    for t in trajs:
                        ft.write(json.dumps(t) + "\n")
                    n_tr += len(trajs)
            summary["splits"][split] = {"maps": len(recs), "trajs": n_tr}
    with open(f"{out_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    return summary


def read_jsonl(path):
    with open(path) as f:
        for line in f:
            yield json.loads(line)
