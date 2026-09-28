"""Closed-loop evaluation on held-out maps: every efficient return level per map is commanded.

python -m rtgrid.llm.eval_cli --data data/mg8 --split test --model Qwen/Qwen3-8B-Base --adapter runs/x/ckpt_1000 \
       --rtg-mode correct --out results/sft_correct.json
Identical maps/levels/prompt/readout/horizon for every checkpoint; only --adapter / --rtg-mode change.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from ..config import SPECS
from ..dataset import read_jsonl
from ..layout import Layout
from ..planner import Planner
from ..serialize import action_token_ids
from .metrics import desired_vs_achieved, episode_row, summarize
from .rollout import Job, run_episodes


def build_jobs(data_dir, split, spec_name, n_maps, rtg_mode, seed=0):
    spec = SPECS[spec_name]
    recs = list(read_jsonl(f"{data_dir}/maps_{split}.jsonl"))[:n_maps]
    rng = np.random.default_rng(seed)
    all_levels = [m["levels"] for m in recs]
    jobs = []
    for i, m in enumerate(recs):
        lay = Layout.from_dict(m["layout"])
        pl = Planner(lay)
        for R in m["levels"]:
            shown = None
            if rtg_mode == "shuffled":  # command from a different map's feasible set
                other = all_levels[int(rng.integers(len(all_levels)))]
                shown = other[int(rng.integers(len(other)))]
            elif rtg_mode.startswith("wrong:"):  # constant shown RTG in reward units, e.g. wrong:50
                shown = int(float(rtg_mode.split(":")[1]) * 10)
            jobs.append(Job(lay, pl, R, spec.horizon, {"map": m["seed"]}, shown))
    return jobs


def load_model(a):
    if a.tiny:
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from ..stubtok import StubTokenizer
        torch.manual_seed(0)
        cfg = Qwen3Config(vocab_size=4096, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2, head_dim=16, tie_word_embeddings=False)
        return Qwen3ForCausalLM(cfg).eval(), StubTokenizer(), "cpu"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter)
    return model.to("cuda").eval(), tok, "cuda"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--spec", default="mg8")
    ap.add_argument("--model", default="Qwen/Qwen3-8B-Base")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--rtg-mode", default="correct", help="correct | none | shuffled | wrong:<value>")
    ap.add_argument("--n-maps", type=int, default=100)
    ap.add_argument("--sample", action="store_true", help="sample at T=1 instead of greedy")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump-steps", action="store_true", help="also write per-step action distributions")
    ap.add_argument("--tiny", action="store_true")
    a = ap.parse_args()
    model, tok, dev = load_model(a)
    jobs = build_jobs(a.data, a.split, a.spec, a.n_maps, a.rtg_mode)
    engine_mode = "none" if a.rtg_mode == "none" else "correct"
    res = run_episodes(model, tok, jobs, action_token_ids(tok), dev, rtg_mode=engine_mode, greedy=not a.sample,
                       batch_size=a.batch_size)
    out = {"args": vars(a), "summary": summarize(res), "desired_vs_achieved": desired_vs_achieved(res),
           "episodes": [episode_row(r) for r in res]}
    if a.dump_steps:
        out["steps"] = [[{k: v for k, v in s.items()} for s in r.steps] for r in res]
    with open(a.out, "w") as f:
        json.dump(out, f)
    s = out["summary"]
    print(json.dumps({k: (round(v["mean"], 3) if isinstance(v, dict) else v) for k, v in s.items()}, indent=1))
    print("slope", out["desired_vs_achieved"]["slope"], "r2", out["desired_vs_achieved"]["r2"])


if __name__ == "__main__":
    main()
