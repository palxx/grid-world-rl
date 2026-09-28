"""LoRA SFT with loss only on action tokens (restricted 4-way softmax at the readout position)."""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..serialize import action_token_ids
from .data import TrajDataset, collate
from .readout import action_logits


def loss_and_stats(model, batch, action_ids, device):
    b = {k: v.to(device) for k, v in batch.items()}
    logits = action_logits(model, b["input_ids"], b["attention_mask"], b["positions"], action_ids)  # [B,P,4]
    valid = b["targets"] != -100
    loss = F.cross_entropy(logits[valid], b["targets"][valid])  # mean over supervised action tokens
    with torch.no_grad():
        pred = logits[valid].argmax(-1)
        acc = (pred == b["targets"][valid]).float().mean()
        in_opt = ((b["opt"][valid] >> pred) & 1).float().mean()  # prediction in RTG-conditioned optimal set
    return loss, {"acc": acc.item(), "opt_acc": in_opt.item(), "n": int(valid.sum())}


@torch.no_grad()
def evaluate(model, loader, action_ids, device, max_batches=50):
    model.eval()
    tot = {"loss": 0.0, "acc": 0.0, "opt_acc": 0.0, "n": 0}
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        loss, st = loss_and_stats(model, batch, action_ids, device)
        n = st["n"]
        tot["loss"] += loss.item() * n
        tot["acc"] += st["acc"] * n
        tot["opt_acc"] += st["opt_acc"] * n
        tot["n"] += n
    model.train()
    n = max(tot["n"], 1)
    return {k: (v / n if k != "n" else v) for k, v in tot.items()}


def ckpt_steps(total, dense=(0, 25, 50, 100, 200, 500, 1000, 2000, 5000, 10000)):
    return {s for s in dense if s <= total} | {total}


def train(model, train_ds, val_ds, action_ids, out_dir, steps, lr=1e-4, bs=8, accum=1, warmup=0.03,
          device="cuda", log_every=10, eval_every=100, num_workers=0, pad_id=0, save_adapters=True, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    g = torch.Generator().manual_seed(seed)
    tl = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=lambda b: collate(b, pad_id),
                    num_workers=num_workers, drop_last=True, generator=g)
    vl = DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=lambda b: collate(b, pad_id)) if val_ds else None
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0, betas=(0.9, 0.999))
    wsteps = max(1, int(warmup * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / wsteps) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    saves = ckpt_steps(steps)
    log = open(f"{out_dir}/log.jsonl", "a")
    model.train()
    step, it, t0 = 0, iter(tl), time.time()

    def save(tag):
        if save_adapters and hasattr(model, "save_pretrained"):
            model.save_pretrained(f"{out_dir}/ckpt_{tag}")

    if 0 in saves:
        save(0)
    while step < steps:
        acc_loss = 0.0
        for _ in range(accum):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(tl)
                batch = next(it)
            loss, st = loss_and_stats(model, batch, action_ids, device)
            (loss / accum).backward()
            acc_loss += loss.item() / accum
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        rec = {"step": step, "loss": acc_loss, "acc": st["acc"], "lr": sched.get_last_lr()[0], "t": time.time() - t0}
        if vl is not None and (step % eval_every == 0 or step == steps):
            rec.update({f"val_{k}": v for k, v in evaluate(model, vl, action_ids, device).items()})
        if step % log_every == 0 or "val_loss" in rec:
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(rec, flush=True)
        if step in saves:
            save(step)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-8B-Base")
    ap.add_argument("--rtg-mode", default="correct", choices=["correct", "none", "shuffled"])
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--alpha", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    action_ids = action_token_ids(tok)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.to("cuda")
    if not a.no_grad_ckpt:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    cfg = LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.05, task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    tr = TrajDataset(a.data, "train", tok, a.rtg_mode, seed=a.seed)
    va = TrajDataset(a.data, "val", tok, "none" if a.rtg_mode == "none" else "correct", seed=a.seed, max_trajs=512)
    train(model, tr, va, action_ids, a.out, a.steps, a.lr, a.bs, a.accum, num_workers=a.workers, pad_id=pad, seed=a.seed)


if __name__ == "__main__":
    main()
