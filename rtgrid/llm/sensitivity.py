"""RTG sweeps on identical states, divergence metrics, forked-state discrimination margin."""
from __future__ import annotations

import numpy as np
import torch

from .prompts import build_prompt
from .readout import action_logits_from_hidden, backbone_and_head

EPS = 1e-12


def jsd(p, q) -> float:
    """Jensen-Shannon divergence in nats (bounded by ln 2)."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(np.where(a > 0, a * (np.log(a + EPS) - np.log(b + EPS)), 0.0)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def tv(p, q) -> float:
    return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())


@torch.no_grad()
def rtg_sweep(model, tok, layout, history, state, rtg_levels, action_ids, device, preamble=True, return_hidden=False):
    """P(a | s, R) for each commanded R at an identical (history, state).

    history: list of (pos, mask, action, reward) already executed; state: (pos, mask) current.
    The RTG shown at every history step is R - (return so far), i.e. the whole RTG sequence is shifted
    consistently. All levels have identical token length (fixed-width RTG) so no padding is needed.
    Returns probs [n_levels, 4] (and last-layer readout hidden if requested)."""
    dec, _ = backbone_and_head(model)
    rows = [build_prompt(tok, layout, history, state, R, preamble).ids for R in rtg_levels]
    assert len({len(r) for r in rows}) == 1, "RTG levels must yield equal-length prompts"
    x = torch.tensor(rows, device=device)
    h = dec(input_ids=x).last_hidden_state[:, -1]
    probs = action_logits_from_hidden(model, h, action_ids).softmax(-1).cpu().numpy()
    return (probs, h) if return_hidden else probs


def pairwise_divergence(probs, fn=jsd):
    n = len(probs)
    return np.array([[fn(probs[i], probs[j]) for j in range(n)] for i in range(n)])


def mass_on(probs_row, action_set_mask: int) -> float:
    return float(sum(probs_row[a] for a in range(4) if (action_set_mask >> a) & 1))


def discrimination_margin(p_a, p_b, opt_a: int, opt_b: int) -> float:
    """On a forked state (opt_a & opt_b == 0): mean over the two commands of
    P(A_own | own command) - P(A_own | other command). 1.0 = perfectly RTG-driven, 0 = RTG ignored."""
    assert opt_a & opt_b == 0 and opt_a and opt_b
    return 0.5 * ((mass_on(p_a, opt_a) - mass_on(p_b, opt_a)) + (mass_on(p_b, opt_b) - mass_on(p_a, opt_b)))


def equivalent_strategy_pairs(planner, pos, mask, rtg_levels, max_shift=1):
    """Pairs (R, R+delta) with |delta|<=max_shift tenths: same optimal action set -> control for irrelevant sensitivity."""
    out = []
    for R in rtg_levels:
        for d in (max_shift, -max_shift):
            A1, _ = planner.action_set(pos, mask, R)
            A2, _ = planner.action_set(pos, mask, R + d)
            if A1 == A2:
                out.append((R, R + d))
    return out
