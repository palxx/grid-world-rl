"""Reward constants (integer tenths) and environment specs.

All rewards/returns are integers in units of 0.1 so RTG arithmetic is exact.
"""
from __future__ import annotations

from dataclasses import dataclass, field

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTION_NAMES = ("UP", "DOWN", "LEFT", "RIGHT")
ACTION_DELTAS = ((0, -1), (0, 1), (-1, 0), (1, 0))  # (dx, dy); x=col, y=row, origin top-left

# rewards in tenths
STEP = -1
BUMP = -5  # additional to STEP
COIN = 20
TREASURE = 50
GOAL = 100
LAVA = -50  # additional to STEP, terminal


@dataclass(frozen=True)
class EnvSpec:
    name: str
    size: int
    lines_x_options: tuple  # tuple of tuples of internal vertical wall x positions
    lines_y_options: tuple
    n_coins: tuple  # inclusive (lo, hi)
    n_treasures: tuple
    n_lava: tuple
    n_obstacles: tuple
    horizon: int
    n_extra_doors: tuple = (1, 3)
    min_goal_dist_frac: float = 0.6  # goal must be >= this fraction of max BFS distance from start
    # map-quality filters (tenths)
    min_levels: int = 4  # distinct, well separated feasible returns
    min_level_gap: int = 10
    min_net_gain: int = 5  # each bonus object must add >= this net of detour cost for some F


SPEC_8 = EnvSpec(
    name="mg8", size=8,
    lines_x_options=((3,), (4,)), lines_y_options=((), (3,), (4,)),
    n_coins=(1, 1), n_treasures=(1, 1), n_lava=(0, 1), n_obstacles=(0, 3),
    horizon=48, n_extra_doors=(0, 1), min_levels=4, min_level_gap=10,
)

SPEC_19 = EnvSpec(
    name="mg19", size=19,
    lines_x_options=((6, 12),), lines_y_options=((6, 12),),
    n_coins=(1, 4), n_treasures=(1, 2), n_lava=(0, 3), n_obstacles=(0, 6),
    horizon=150, n_extra_doors=(1, 3), min_levels=4, min_level_gap=10,
)

SPECS = {"mg8": SPEC_8, "mg19": SPEC_19}
