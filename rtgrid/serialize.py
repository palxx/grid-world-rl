"""Fixed-width text format + piecewise tokenization.

Every field is emitted as its own text *segment*, tokenized independently and concatenated. The rollout
harness builds prompts incrementally with the same segments, so training and inference tokenization are
identical by construction. Segment kinds let us locate RTG / position / readout tokens exactly.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import config as C
from .layout import Layout

PREAMBLE = (
    "Gridworld return-conditioned policy. MAP rows list cells: # wall, . floor, G goal, c coin, T treasure, "
    "L lava. Coordinates are x,y from the top-left (x=column, y=row). Each line gives R (return still to "
    "collect, +/-tens.tenths), P (agent position), M (bits of collected coins then treasures), then the "
    "action A: UP y-1, DOWN y+1, LEFT x-1, RIGHT x+1. Choose actions so the total return matches R."
)
ACTION_TEXT = tuple(" " + n for n in C.ACTION_NAMES)  # single-token check in scripts/check_tokenizer.py

# segment kinds
CTX, RTG, POS, MASK, ACT, ACT_CTX = "ctx", "rtg", "pos", "mask", "act", "act_ctx"


def fmt_rtg(t: int) -> str:
    """Integer tenths -> fixed-width '+15.1' / '-03.5' (5 chars). |t| must be < 1000."""
    assert abs(t) < 1000, t
    a = abs(t)
    return f"{'+' if t >= 0 else '-'}{a // 10:02d}.{a % 10}"


def parse_rtg(s: str) -> int:
    m = re.fullmatch(r"([+-])(\d{2})\.(\d)", s)
    assert m, s
    v = int(m.group(2)) * 10 + int(m.group(3))
    return v if m.group(1) == "+" else -v


def fmt_pos(p) -> str:
    return f"{p[0]:02d},{p[1]:02d}"


def fmt_mask(mask: int, k: int) -> str:
    return "".join("1" if (mask >> i) & 1 else "0" for i in range(k))


def obj_line(layout: Layout) -> str:
    g = layout.goal
    parts = [f"G={fmt_pos(g)}"]
    parts += [f"C{i + 1}={fmt_pos(p)}" for i, p in enumerate(layout.coins)]
    parts += [f"T{i + 1}={fmt_pos(p)}" for i, p in enumerate(layout.treasures)]
    parts += [f"L{i + 1}={fmt_pos(p)}" for i, p in enumerate(layout.lava)]
    return " ".join(parts)


def header_segments(layout: Layout, preamble: bool = True) -> list:
    segs = []
    if preamble:
        segs.append((PREAMBLE + "\n", CTX))
    segs.append((f"ENV:MG SIZE:{layout.size}\nMAP:\n", CTX))
    for row in layout.rows():
        segs.append((" ".join(row) + "\n", CTX))
    segs.append((f"OBJ:{obj_line(layout)}\nSTART:{fmt_pos(layout.start)}\n--\n", CTX))
    return segs


def step_prefix_segments(rtg, pos, mask: int, k: int) -> list:
    """Segments of one step up to (and including) 'A:'; the action token follows.
    rtg=None drops the RTG field (used for the no-RTG control / ablation)."""
    head = [] if rtg is None else [("R:", CTX), (fmt_rtg(rtg), RTG), ("|", CTX)]
    return head + [("P:", CTX), (fmt_pos(pos), POS), ("|M:", CTX), (fmt_mask(mask, k), MASK), ("|A:", CTX)]


def trajectory_segments(layout: Layout, traj, preamble: bool = True, rtg_of=None) -> list:
    """rtg_of(step_index, step) -> int|None overrides the RTG shown (control models); default = step.rtg."""
    k = len(layout.bonus)
    segs = header_segments(layout, preamble)
    for i, s in enumerate(traj.steps):
        segs += step_prefix_segments(s.rtg if rtg_of is None else rtg_of(i, s), s.pos, s.mask, k)
        segs.append((ACTION_TEXT[s.action], ACT if s.supervise else ACT_CTX))
        segs.append(("\n", CTX))
    return segs


def render_text(segs) -> str:
    return "".join(t for t, _ in segs)


@dataclass
class Encoded:
    ids: list
    readout: list = field(default_factory=list)  # token index whose hidden state predicts each supervised action
    target: list = field(default_factory=list)  # action class per supervised position
    rtg_spans: list = field(default_factory=list)  # (start, end) token span of each step's RTG
    step_readout: list = field(default_factory=list)  # readout index for every step (supervised or not)
    step_supervised: list = field(default_factory=list)


def encode_segments(tok, segs, actions=None) -> Encoded:
    """Tokenize segments piecewise. `actions` (class per ACT/ACT_CTX segment, in order) gives targets."""
    enc = Encoded(ids=[])
    a_iter = iter(actions or [])
    for text, kind in segs:
        ids = tok.encode(text, add_special_tokens=False)
        if kind in (ACT, ACT_CTX):
            assert len(ids) == 1, f"action {text!r} is not a single token: {ids}"
            enc.step_readout.append(len(enc.ids) - 1)
            enc.step_supervised.append(kind == ACT)
            a = next(a_iter, None)
            if kind == ACT:
                enc.readout.append(len(enc.ids) - 1)
                enc.target.append(a)
        elif kind == RTG:
            enc.rtg_spans.append((len(enc.ids), len(enc.ids) + len(ids)))
        enc.ids.extend(ids)
    return enc


def encode_trajectory(tok, layout: Layout, traj, preamble: bool = True, rtg_of=None) -> Encoded:
    segs = trajectory_segments(layout, traj, preamble, rtg_of)
    return encode_segments(tok, segs, [s.action for s in traj.steps])


def action_token_ids(tok) -> list:
    ids = []
    for t in ACTION_TEXT:
        x = tok.encode(t, add_special_tokens=False)
        assert len(x) == 1, (t, x)
        ids.append(x[0])
    assert len(set(ids)) == 4
    return ids


def parse_text(text: str):
    """Inverse of render_text for the step section: list of (rtg, (x,y), mask_str, action_idx)."""
    body = text.split("--\n", 1)[1]
    out = []
    for line in body.strip("\n").split("\n"):
        m = re.fullmatch(r"R:([+-]\d{2}\.\d)\|P:(\d{2}),(\d{2})\|M:([01]*)\|A: (UP|DOWN|LEFT|RIGHT)", line)
        assert m, line
        out.append((parse_rtg(m.group(1)), (int(m.group(2)), int(m.group(3))), m.group(4),
                    C.ACTION_NAMES.index(m.group(5))))
    return out
