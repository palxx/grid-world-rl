"""Single source of truth for building decision prompts (used by sweeps and interpretability)."""
from __future__ import annotations

from dataclasses import dataclass

from ..serialize import ACTION_TEXT, header_segments, step_prefix_segments
from .data import CachedTok


@dataclass
class Prompt:
    ids: list
    rtg_spans: list  # token span (start, end) of every RTG field, history steps then the current step
    pos_spans: list  # agent-position field spans (same order)
    readout: int  # index of the token whose hidden state predicts the action (last token)
    header_len: int


def build_prompt(tok, layout, history, state, R, preamble=True, show_rtg=True) -> Prompt:
    """history: [(pos, mask, action, reward)]; state: (pos, mask); R: commanded return-to-go at the *start*
    of the history (tenths). RTG shown at each step = R - return so far."""
    tok = CachedTok(tok)
    k = len(layout.bonus)
    act_tok = [tok.encode(t)[0] for t in ACTION_TEXT]
    nl = tok.encode("\n")
    ids, rspans, pspans = [], [], []

    def add(segs):
        for text, kind in segs:
            t = tok.encode(text)
            if kind == "rtg":
                rspans.append((len(ids), len(ids) + len(t)))
            elif kind == "pos":
                pspans.append((len(ids), len(ids) + len(t)))
            ids.extend(t)

    add(header_segments(layout, preamble))
    header_len = len(ids)
    cum = 0
    for pos, mask, a, r in history:
        add(step_prefix_segments(R - cum if show_rtg else None, pos, mask, k))
        ids += [act_tok[a]] + nl
        cum += r
    add(step_prefix_segments(R - cum if show_rtg else None, state[0], state[1], k))
    return Prompt(ids, rspans, pspans, len(ids) - 1, header_len)
