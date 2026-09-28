import numpy as np
import pytest

from rtgrid.config import SPECS
from rtgrid.generator import make_layout
from rtgrid.planner import Planner


@pytest.fixture(scope="session", params=["mg8", "mg19"])
def spec(request):
    return SPECS[request.param]


@pytest.fixture(scope="session")
def maps(spec):
    out = []
    for s in range(12):
        lay, _ = make_layout(spec, s)
        out.append((s, lay, Planner(lay)))
    return out
