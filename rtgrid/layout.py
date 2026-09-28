"""Static map representation + dihedral-canonical hashing."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

WALL, FLOOR, GOALC, COINC, TREAC, LAVAC = "#", ".", "G", "c", "T", "L"


@dataclass
class Layout:
    grid: np.ndarray  # (H, W) array of single chars (dtype '<U1'), row=y, col=x
    start: tuple  # (x, y)

    @property
    def size(self) -> int:
        return self.grid.shape[0]

    @property
    def goal(self) -> tuple:
        ys, xs = np.where(self.grid == GOALC)
        return (int(xs[0]), int(ys[0]))

    def cells(self, ch: str) -> list:
        """Row-major sorted (y, then x) list of (x, y) cells holding ch."""
        ys, xs = np.where(self.grid == ch)
        return [(int(x), int(y)) for y, x in sorted(zip(ys.tolist(), xs.tolist()))]

    @property
    def coins(self):
        return self.cells(COINC)

    @property
    def treasures(self):
        return self.cells(TREAC)

    @property
    def lava(self):
        return self.cells(LAVAC)

    @property
    def bonus(self) -> list:
        """Bonus objects in mask-bit order: coins then treasures, each row-major."""
        return self.coins + self.treasures

    def rows(self) -> list:
        return ["".join(r) for r in self.grid]

    def to_dict(self) -> dict:
        return {"rows": self.rows(), "start": list(self.start)}

    @staticmethod
    def from_dict(d: dict) -> "Layout":
        grid = np.array([list(r) for r in d["rows"]], dtype="<U1")
        return Layout(grid, (int(d["start"][0]), int(d["start"][1])))


def _transforms(grid: np.ndarray, start: tuple):
    """All 8 dihedral transforms of (grid, start)."""
    n = grid.shape[0]
    marked = grid.copy()
    # encode start as an extra marker via separate tracking
    out = []
    g, (sx, sy) = grid, start
    for flip in (False, True):
        gg, sxx, syy = (g[:, ::-1], n - 1 - sx, sy) if flip else (g, sx, sy)
        cg, cx, cy = gg, sxx, syy
        for _ in range(4):
            out.append((cg, (cx, cy)))
            # rotate 90deg clockwise: new[y', x'] = old[n-1-x', y']  -> (x,y)->(n-1-y, x)
            cg = np.rot90(cg, k=-1)
            cx, cy = n - 1 - cy, cx
    return out


def canonical_hash(layout: Layout) -> str:
    """Hash invariant to the 8 dihedral symmetries (start included)."""
    best = None
    for g, s in _transforms(layout.grid, layout.start):
        blob = ("\n".join("".join(r) for r in g) + f"|{s[0]},{s[1]}").encode()
        if best is None or blob < best:
            best = blob
    return hashlib.sha1(best).hexdigest()[:16]
