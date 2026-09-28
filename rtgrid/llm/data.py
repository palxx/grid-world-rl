"""On-the-fly tokenized SFT dataset (byte-offset indexed JSONL) with RTG control modes."""
from __future__ import annotations

import json

import numpy as np
import torch
from torch.utils.data import Dataset

from ..layout import Layout
from ..serialize import encode_segments, trajectory_segments
from ..trajectory import Trajectory


class CachedTok:
    """Segment-level tokenization cache (segments repeat heavily: RTG strings, positions, masks)."""

    def __init__(self, tok):
        self.tok, self._c = tok, {}

    def encode(self, text, add_special_tokens=False):
        v = self._c.get(text)
        if v is None:
            v = self._c[text] = self.tok.encode(text, add_special_tokens=False)
        return v


class JsonlIndex:
    def __init__(self, path):
        self.path, self.offsets = path, []
        with open(path, "rb") as f:
            pos = 0
            for line in f:
                self.offsets.append(pos)
                pos += len(line)
        self._f = None

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, i):
        if self._f is None:
            self._f = open(self.path, "rb")
        self._f.seek(self.offsets[i])
        return json.loads(self._f.readline())


def rtg_transform(mode: str, traj: Trajectory, rng: np.random.Generator, pool_r0: list):
    """Returns rtg_of(i, step) for control models. modes: correct | none | shuffled."""
    if mode == "correct":
        return None
    if mode == "none":
        return lambda i, s: None
    if mode == "shuffled":  # shift the whole RTG sequence to another trajectory's initial RTG (keeps decrement consistency)
        delta = int(pool_r0[int(rng.integers(len(pool_r0)))]) - traj.steps[0].rtg
        return lambda i, s: max(-999, min(999, s.rtg + delta))
    raise ValueError(mode)


class TrajDataset(Dataset):
    def __init__(self, data_dir: str, split: str, tok, rtg_mode: str = "correct", preamble: bool = True,
                 max_trajs: int | None = None, seed: int = 0):
        self.maps = {m["seed"]: Layout.from_dict(m["layout"]) for m in JsonlIndex(f"{data_dir}/maps_{split}.jsonl")}
        self.trajs = JsonlIndex(f"{data_dir}/trajs_{split}.jsonl")
        self.n = len(self.trajs) if max_trajs is None else min(max_trajs, len(self.trajs))
        self.tok, self.rtg_mode, self.preamble, self.seed = CachedTok(tok), rtg_mode, preamble, seed
        self._hdr = {}
        self.pool_r0 = [self.trajs[i]["steps"][0][3] for i in range(min(self.n, 2000))]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        d = self.trajs[i]
        traj = Trajectory.from_dict(d)
        layout = self.maps[traj.map_seed]
        rng = np.random.default_rng([self.seed, i])
        rtg_of = rtg_transform(self.rtg_mode, traj, rng, self.pool_r0)
        segs = trajectory_segments(layout, traj, self.preamble, rtg_of)
        enc = encode_segments(self.tok, segs, [s.action for s in traj.steps])
        sup_steps = [k for k, s in enumerate(traj.steps) if s.supervise]
        return {
            "ids": enc.ids, "readout": enc.readout, "target": enc.target,
            "opt": [traj.steps[k].opt for k in sup_steps],
            "rtg": [traj.steps[k].rtg for k in sup_steps],
        }


def collate(batch, pad_id: int = 0):
    B = len(batch)
    T = max(len(b["ids"]) for b in batch)
    P = max(1, max(len(b["readout"]) for b in batch))
    ids = torch.full((B, T), pad_id, dtype=torch.long)
    att = torch.zeros((B, T), dtype=torch.long)
    pos = torch.full((B, P), -1, dtype=torch.long)
    tgt = torch.full((B, P), -100, dtype=torch.long)
    opt = torch.zeros((B, P), dtype=torch.long)
    for i, b in enumerate(batch):
        n, p = len(b["ids"]), len(b["readout"])
        ids[i, :n] = torch.tensor(b["ids"])
        att[i, :n] = 1
        if p:
            pos[i, :p] = torch.tensor(b["readout"])
            tgt[i, :p] = torch.tensor(b["target"])
            opt[i, :p] = torch.tensor(b["opt"])
    return {"input_ids": ids, "attention_mask": att, "positions": pos, "targets": tgt, "opt": opt}
