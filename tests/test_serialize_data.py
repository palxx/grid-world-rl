import json

import numpy as np

from rtgrid import config as C
from rtgrid.dataset import build_splits, read_jsonl, write_dataset
from rtgrid.generator import make_layout
from rtgrid.planner import Planner
from rtgrid.serialize import (ACT, ACTION_TEXT, RTG, encode_trajectory, fmt_rtg, parse_rtg, parse_text,
                              render_text, trajectory_segments, action_token_ids)
from rtgrid.stubtok import StubTokenizer
from rtgrid.trajectory import Trajectory, generate_for_layout, rollout_dart, rollout_fork


def test_rtg_format_roundtrip_and_width():
    for t in range(-999, 1000):
        s = fmt_rtg(t)
        assert len(s) == 5 and parse_rtg(s) == t or t == 0 and s == "+00.0"
    assert fmt_rtg(151) == "+15.1" and fmt_rtg(-35) == "-03.5"


def test_rtg_invariants_and_masks(maps):
    rng = np.random.default_rng(0)
    for s, lay, pl in maps[:8]:
        for tr in generate_for_layout(lay, pl, 150, rng, 20, s):
            st = tr.steps
            for a, b in zip(st, st[1:]):
                assert b.rtg == a.rtg - a.reward
            if tr.source == "hindsight":
                assert st[0].rtg == tr.ret and all(x.supervise for x in st)
                assert st[-1].rtg == st[-1].reward
            else:
                assert st[0].rtg == tr.target
            assert tr.ret == sum(x.reward for x in st)
            # only the last step may terminate; nothing after goal/lava
            assert all(x.reward < C.GOAL for x in st[:-1])
            assert (tr.end == "goal") == (st[-1].reward >= C.GOAL + C.STEP)


def test_serialization_tokens(maps):
    tok = StubTokenizer()
    ids = action_token_ids(tok)
    rng = np.random.default_rng(1)
    for s, lay, pl in maps[:6]:
        for tr in generate_for_layout(lay, pl, 150, rng, 10, s):
            enc = encode_trajectory(tok, lay, tr)
            # constant RTG token width
            assert all(b - a == 5 for a, b in enc.rtg_spans) and len(enc.rtg_spans) == len(tr.steps)
            # loss applies to exactly the supervised action tokens
            assert len(enc.readout) == sum(x.supervise for x in tr.steps) == len(enc.target)
            for r, t in zip(enc.readout, enc.target):
                assert enc.ids[r + 1] == ids[t]
            # text round trip
            txt = render_text(trajectory_segments(lay, tr))
            parsed = parse_text(txt)
            assert [(p[0], p[1], p[3]) for p in parsed] == [(x.rtg, x.pos, x.action) for x in tr.steps]


def test_fork_pairs(maps):
    found = 0
    rng = np.random.default_rng(3)
    for s, lay, pl in maps:
        out = rollout_fork(lay, pl, 150, rng, s)
        if not out:
            continue
        found += 1
        a, b = out
        t = a.meta["fork_t"]
        assert a.meta["pair"][0] != a.meta["pair"][1]
        for i in range(t):  # identical (state, action, reward) prefix, different RTG, context-only
            sa, sb = a.steps[i], b.steps[i]
            assert (sa.pos, sa.mask, sa.action, sa.reward) == (sb.pos, sb.mask, sb.action, sb.reward)
            assert not sa.supervise and not sb.supervise
            assert sa.rtg != sb.rtg
        sa, sb = a.steps[t], b.steps[t]
        assert (sa.pos, sa.mask) == (sb.pos, sb.mask) and sa.rtg != sb.rtg
        assert sa.opt & sb.opt == 0 and sa.opt and sb.opt  # disjoint RTG-optimal action sets
        assert sa.supervise and sb.supervise
    assert found >= 3


def test_dart_masks_noise_only(maps):
    rng = np.random.default_rng(0)
    s, lay, pl = maps[1]
    tr = rollout_dart(lay, pl, 150, rng, 0.3, map_seed=s)
    for x in tr.steps:
        if x.supervise:
            assert x.opt >> x.action & 1


def test_splits_disjoint(spec, tmp_path):
    sp = build_splits(spec, 12, 4, 4)
    hs = [{h for _, h in v} for v in sp.values()]
    assert not (hs[0] & hs[1]) and not (hs[0] & hs[2]) and not (hs[1] & hs[2])
    seeds = [{s for s, _ in v} for v in sp.values()]
    assert not (seeds[0] & seeds[1]) and not (seeds[0] & seeds[2])


def test_write_dataset_and_no_leakage(tmp_path):
    summ = write_dataset("mg8", str(tmp_path), 6, 3, 3, traj_per_map=4, workers=2, test_traj_per_map=2,
                         holdout_bands=[(0, 0)])
    assert summ["splits"]["train"]["maps"] == 6
    hashes = {sp: {m["hash"] for m in read_jsonl(f"{tmp_path}/maps_{sp}.jsonl")} for sp in ("train", "val", "test")}
    assert not (hashes["train"] & hashes["test"]) and not (hashes["train"] & hashes["val"])
    train_seeds = {m["seed"] for m in read_jsonl(f"{tmp_path}/maps_train.jsonl")}
    for t in read_jsonl(f"{tmp_path}/trajs_train.jsonl"):
        assert t["map_seed"] in train_seeds and len(t["steps"]) <= 48
