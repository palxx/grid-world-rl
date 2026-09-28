"""Batched closed-loop evaluation with KV cache.

Envs are grouped by number of bonus objects k (mask width) so every step appends the same number of tokens;
only the header differs in length and is left-padded. The full 4-way action distribution is recorded at
every step. Prompts are built from the same segments as training (piecewise tokenization).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from transformers import DynamicCache

from .. import config as C
from ..env import CardinalEnv
from ..layout import Layout
from ..planner import Planner
from ..serialize import ACTION_TEXT, header_segments, step_prefix_segments, action_token_ids
from .data import CachedTok
from .readout import action_logits_from_hidden, backbone_and_head


@dataclass
class Job:
    layout: Layout
    planner: Planner
    target: int  # commanded return (tenths)
    horizon: int
    tag: dict = field(default_factory=dict)  # free-form (map seed, rtg label, ...)
    shown_target: int | None = None  # RTG actually shown (default = target); use for shuffled/wrong-RTG ablations


@dataclass
class EpisodeResult:
    job: Job
    steps: list  # dicts: pos, mask, rtg_shown, probs(4), action, reward, opt
    ret: int
    end: str
    mask: int

    @property
    def success(self):
        return self.end == "goal"


def _ids(tok, segs):
    out = []
    for text, _ in segs:
        out.extend(tok.encode(text, add_special_tokens=False))
    return out


def _prefix_ids(tok, rtg, pos, mask, k):
    return _ids(tok, step_prefix_segments(rtg, pos, mask, k))


@torch.no_grad()
def run_episodes(model, tok, jobs, action_ids, device, rtg_mode="correct", greedy=True, temperature=1.0,
                 batch_size=32, preamble=True, seed=0, compact=True):
    """rtg_mode: 'correct' (shown RTG = shown_target - return so far, decremented by observed reward),
    'none' (RTG field omitted)."""
    tok = CachedTok(tok)
    rng = np.random.default_rng(seed)
    dec, _ = backbone_and_head(model)
    act_tok = [tok.encode(t)[0] for t in ACTION_TEXT]
    nl = tok.encode("\n")
    results = [None] * len(jobs)
    groups = {}
    for i, j in enumerate(jobs):
        groups.setdefault(len(j.layout.bonus), []).append(i)
    model.eval()
    for k, idxs in groups.items():
        for c in range(0, len(idxs), batch_size):
            chunk = idxs[c:c + batch_size]
            _run_chunk(model, dec, tok, [jobs[i] for i in chunk], chunk, results, k, action_ids, act_tok, nl,
                       device, rtg_mode, greedy, temperature, preamble, rng, compact)
    return results


def _run_chunk(model, dec, tok, jobs, slots, results, k, action_ids, act_tok, nl, device, rtg_mode, greedy, temp,
               preamble, rng, compact):
    B = len(jobs)
    envs = [CardinalEnv(j.layout, j.horizon) for j in jobs]
    shown0 = [j.target if j.shown_target is None else j.shown_target for j in jobs]
    recs = [[] for _ in range(B)]
    ends = [""] * B
    live = list(range(B))  # original indices of envs still in the batch (rows of the cache)
    headers = [_ids(tok, header_segments(j.layout, preamble)) for j in jobs]

    def rtg_shown(b):
        return None if rtg_mode == "none" else shown0[b] - envs[b].cum

    new = [headers[b] + _prefix_ids(tok, rtg_shown(b), envs[b].pos, envs[b].mask, k) for b in live]
    cache = DynamicCache()
    att = None
    last_actions = None
    horizon = max(j.horizon for j in jobs)
    for t in range(horizon):
        # ---- assemble the input for the live rows (left pad at t==0) ----
        L = max(len(x) for x in new)
        ids = torch.zeros((len(live), L), dtype=torch.long)
        m_new = torch.zeros((len(live), L), dtype=torch.long)
        for r, x in enumerate(new):
            ids[r, L - len(x):] = torch.tensor(x)
            m_new[r, L - len(x):] = 1
        ids, m_new = ids.to(device), m_new.to(device)
        att = m_new if att is None else torch.cat([att, m_new], 1)
        pos_ids = (att.cumsum(1) - 1).clamp(min=0)[:, -L:]
        out = dec(input_ids=ids, attention_mask=att, position_ids=pos_ids, past_key_values=cache, use_cache=True)
        h = out.last_hidden_state[:, -1]
        logits = action_logits_from_hidden(model, h, action_ids)
        probs = logits.softmax(-1).cpu().numpy()
        # ---- choose actions & step envs ----
        acts = []
        for r, b in enumerate(live):
            p = probs[r]
            if greedy:
                a = int(p.argmax())
            else:
                q = np.exp(np.log(np.clip(p, 1e-12, 1)) / temp)
                a = int(rng.choice(4, p=q / q.sum()))
            env = envs[b]
            rshown = rtg_shown(b)
            opt = 0
            if rshown is not None:
                A, _ = jobs[b].planner.action_set(env.pos, env.mask, rshown)
                opt = sum(1 << x for x in A)
            pos, mask = env.pos, env.mask
            res = env.step(a)
            recs[b].append({"pos": pos, "mask": mask, "rtg": rshown, "probs": p.tolist(), "action": a,
                            "reward": res.reward, "opt": opt})
            if env.done:
                ends[b] = "goal" if res.event == "goal" else "lava" if res.event == "lava" else "timeout"
            acts.append(a)
        # ---- drop finished envs (compaction) and build next inputs ----
        keep = [r for r, b in enumerate(live) if not envs[b].done]
        if not keep:
            break
        if len(keep) < len(live):
            sel = torch.tensor(keep, device=device)
            cache.batch_select_indices(sel)
            att = att[sel]
        live = [live[r] for r in keep]
        new = [[act_tok[acts[r]]] + nl + _prefix_ids(tok, rtg_shown(b), envs[b].pos, envs[b].mask, k)
               for r, b in zip(keep, live)]
    for b in range(B):
        results[slots[b]] = EpisodeResult(jobs[b], recs[b], envs[b].cum, ends[b], envs[b].mask)
