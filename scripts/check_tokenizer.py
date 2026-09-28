#!/usr/bin/env python
"""Verify the serialization is tokenizer-safe. Run on the machine that has the Qwen tokenizer:
    python scripts/check_tokenizer.py --tokenizer Qwen/Qwen3-8B-Base
Use --stub to run the same checks with the offline stub."""
import argparse
import numpy as np

from rtgrid.config import SPECS
from rtgrid.generator import make_layout
from rtgrid.serialize import (ACTION_TEXT, fmt_mask, fmt_pos, fmt_rtg, header_segments, step_prefix_segments,
                              action_token_ids)

ap = argparse.ArgumentParser()
ap.add_argument("--tokenizer", default="Qwen/Qwen3-8B-Base")
ap.add_argument("--stub", action="store_true")
a = ap.parse_args()
if a.stub:
    from rtgrid.stubtok import StubTokenizer
    tok = StubTokenizer()
else:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
enc = lambda t: tok.encode(t, add_special_tokens=False)

ids = action_token_ids(tok)  # asserts single, distinct tokens
print("action token ids:", dict(zip(ACTION_TEXT, ids)))
widths = {len(enc(fmt_rtg(t))) for t in range(-999, 1000)}
assert widths == {5}, f"RTG token widths vary: {widths}  -> use per-digit spacing or another format"
print("RTG: fixed 5 tokens for every value in [-99.9, 99.9]")
for name, f, rng_ in (("pos", fmt_pos, [(x, y) for x in range(25) for y in range(25)]),):
    w = {len(enc(f(v))) for v in rng_}
    assert len(w) == 1, (name, w)
    print(f"{name}: fixed {w.pop()} tokens")
for k in range(0, 7):
    w = {len(enc(fmt_mask(m, k))) for m in range(1 << k)} if k else {0}
    assert len(w) == 1, ("mask", k, w)
per_step = {}
for spec in SPECS.values():
    lay, _ = make_layout(spec, 0)
    k = len(lay.bonus)
    hdr = sum(len(enc(t)) for t, _ in header_segments(lay))
    rows = [len(enc(t)) for t, kind in header_segments(lay, preamble=False)[1:1 + lay.size]]
    st = sum(len(enc(t)) for t, _ in step_prefix_segments(0, (1, 1), 0, k)) + 2  # + action + newline
    print(f"{spec.name}: header {hdr} tokens, map row tokens {set(rows)}, per-step {st} tokens (k={k})")
    assert len(set(rows)) == 1
    print(f"   est. tokens/traj @ 60 steps: {hdr + 60 * st}")
print("OK")
