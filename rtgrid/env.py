"""Cardinal-move gridworld simulator over a Layout. Rewards are integer tenths."""
from __future__ import annotations

from dataclasses import dataclass

from . import config as C
from .layout import COINC, GOALC, LAVAC, TREAC, WALL, Layout


@dataclass
class StepResult:
    reward: int
    terminated: bool  # goal or lava
    truncated: bool  # horizon
    bump: bool = False
    event: str = ""  # "", "coin", "treasure", "goal", "lava"


class CardinalEnv:
    def __init__(self, layout: Layout, horizon: int):
        self.layout = layout
        self.grid = layout.grid
        self.size = layout.size
        self.horizon = horizon
        self.bonus = layout.bonus
        self.bonus_idx = {p: i for i, p in enumerate(self.bonus)}
        self.reset()

    def reset(self):
        self.pos = tuple(self.layout.start)
        self.mask = 0
        self.t = 0
        self.cum = 0  # return so far (tenths)
        self.done = False
        return self.pos, self.mask

    def step(self, a: int) -> StepResult:
        assert not self.done, "step() after episode end"
        dx, dy = C.ACTION_DELTAS[a]
        nx, ny = self.pos[0] + dx, self.pos[1] + dy
        self.t += 1
        r = C.STEP
        bump = False
        event = ""
        terminated = False
        if not (0 <= nx < self.size and 0 <= ny < self.size) or self.grid[ny, nx] == WALL:
            r += C.BUMP
            bump = True
        else:
            self.pos = (nx, ny)
            ch = self.grid[ny, nx]
            if ch == GOALC:
                r += C.GOAL
                terminated, event = True, "goal"
            elif ch == LAVAC:
                r += C.LAVA
                terminated, event = True, "lava"
            elif ch in (COINC, TREAC) and not (self.mask >> self.bonus_idx[self.pos]) & 1:
                self.mask |= 1 << self.bonus_idx[self.pos]
                r += C.COIN if ch == COINC else C.TREASURE
                event = "coin" if ch == COINC else "treasure"
        truncated = (not terminated) and self.t >= self.horizon
        self.cum += r
        self.done = terminated or truncated
        return StepResult(r, terminated, truncated, bump, event)
