"""Mechanistic-interpretability utilities on TransformerLens' TransformerBridge (Qwen3).

Workflow: merge a LoRA checkpoint into the base weights (merge_and_unload), boot a bridge with hf_model=merged,
then run patching / steering / ablation on prompts from rtgrid.llm.prompts.build_prompt. All effects are
measured on the restricted 4-way action distribution (identical to the policy readout).
"""
from __future__ import annotations

import numpy as np
import torch


def load_bridge(hf_model, model_name_or_path: str, device="cuda", dtype=torch.bfloat16):
    from transformer_lens.model_bridge import TransformerBridge
    return TransformerBridge.boot_transformers(model_name_or_path, hf_model=hf_model, device=device,
                                               dtype=dtype, tokenizer=None)


def merge_lora(peft_model):
    return peft_model.merge_and_unload()


def resid(layer: int, where: str = "pre") -> str:
    return f"blocks.{layer}.hook_resid_{where}"


def _probs(logits, action_ids):
    return logits[:, -1, action_ids].float().softmax(-1)


@torch.no_grad()
def run_probs(bridge, ids, action_ids, hooks=None):
    x = torch.as_tensor(ids, device=next(bridge.parameters()).device)
    if x.ndim == 1:
        x = x[None]
    logits = bridge.run_with_hooks(x, fwd_hooks=hooks or [])
    return _probs(logits, action_ids)


@torch.no_grad()
def cache_resid(bridge, ids, layers, where="pre"):
    """{layer: [T, D]} residual stream for a single prompt."""
    x = torch.as_tensor(ids, device=next(bridge.parameters()).device)[None]
    names = {resid(l, where) for l in layers}
    _, cache = bridge.run_with_cache(x, names_filter=lambda n: n in names)
    return {l: cache[resid(l, where)][0].detach() for l in layers}


def patch_hook(layer, clean_resid, positions, where="pre"):
    """Replace the residual stream at `positions` with the clean run's values (denoising direction)."""
    pos = torch.as_tensor(list(positions), device=clean_resid.device)

    def fn(act, hook):
        act = act.clone()
        act[:, pos] = clean_resid[pos].to(act.dtype)
        return act
    return (resid(layer, where), fn)


def steer_hook(layer, direction, alpha, positions, where="pre"):
    """h' = h + alpha * v at `positions`."""
    pos = torch.as_tensor(list(positions), device=direction.device)

    def fn(act, hook):
        act = act.clone()
        act[:, pos] = act[:, pos] + alpha * direction.to(act.dtype)
        return act
    return (resid(layer, where), fn)


def ablate_hook(layer, direction, positions=None, where="pre"):
    """Project out `direction` (unit-normalized) from the residual stream (all positions unless given)."""
    v = direction / direction.norm()

    def fn(act, hook):
        act = act.clone()
        sl = slice(None) if positions is None else torch.as_tensor(list(positions), device=act.device)
        x = act[:, sl]
        act[:, sl] = x - (x @ v.to(x.dtype))[..., None] * v.to(x.dtype)
        return act
    return (resid(layer, where), fn)


def normalized_effect(p_patched, p_low, p_high, high_set_mask: int) -> float:
    """(P_patched(A_high) - P_low(A_high)) / (P_high(A_high) - P_low(A_high)); 1 = full transfer, 0 = none."""
    m = torch.tensor([(high_set_mask >> a) & 1 for a in range(4)], dtype=torch.float32, device=p_low.device)
    pl, ph, pp = ((p @ m).item() for p in (p_low[0], p_high[0], p_patched[0]))
    den = ph - pl
    return float("nan") if abs(den) < 1e-6 else (pp - pl) / den


def patching_sweep(bridge, ids_low, ids_high, layers, position_groups: dict, action_ids, high_set_mask):
    """Denoising patch: run the *low-RTG* prompt, patch in *high-RTG* residuals at each (layer, position group).
    Prompts must have identical length. Returns {group: [effect per layer]} plus reference distributions."""
    assert len(ids_low) == len(ids_high)
    p_low = run_probs(bridge, ids_low, action_ids)
    p_high = run_probs(bridge, ids_high, action_ids)
    clean = cache_resid(bridge, ids_high, layers)
    out = {}
    for name, pos in position_groups.items():
        out[name] = [normalized_effect(run_probs(bridge, ids_low, action_ids, [patch_hook(l, clean[l], pos)]),
                                       p_low, p_high, high_set_mask) for l in layers]
    return out, p_low, p_high


def diff_in_means(bridge, id_pairs, layer, position_fn, where="pre"):
    """v = mean over pairs of resid(high)[pos] - resid(low)[pos]; position_fn(ids_high) -> index (e.g. last RTG digit)."""
    diffs = []
    for lo, hi, pos in id_pairs:
        rl = cache_resid(bridge, lo, [layer], where)[layer][pos]
        rh = cache_resid(bridge, hi, [layer], where)[layer][pos]
        diffs.append((rh - rl).float())
    return torch.stack(diffs).mean(0)


def ridge_probe(X: np.ndarray, y: np.ndarray, groups: np.ndarray, alpha=10.0, n_folds=5, classify=False, seed=0):
    """Linear probe with GROUP-wise CV (groups = map ids) so test maps are unseen. Returns mean held-out R^2
    (regression) or accuracy (classification, one-vs-all ridge) and the same for a label-shuffled control."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    rng.shuffle(ug)
    folds = np.array_split(ug, n_folds)

    def fit_eval(y_):
        scores = []
        for f in folds:
            te = np.isin(groups, f)
            tr = ~te
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
            Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
            if classify:
                cls = np.unique(y_[tr])
                Y = (y_[tr][:, None] == cls[None]).astype(float)
            else:
                Y = y_[tr][:, None].astype(float)
            W = np.linalg.solve(Xtr.T @ Xtr + alpha * np.eye(X.shape[1]), Xtr.T @ (Y - Y.mean(0)))
            pred = Xte @ W + Y.mean(0)
            if classify:
                scores.append(float((cls[pred.argmax(1)] == y_[te]).mean()))
            else:
                yt = y_[te].astype(float)
                scores.append(1 - float(((pred[:, 0] - yt) ** 2).sum() / (((yt - yt.mean()) ** 2).sum() + 1e-12)))
        return float(np.mean(scores))

    return {"score": fit_eval(y), "control": fit_eval(rng.permutation(y))}
