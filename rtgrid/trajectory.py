"""Trajectory records, rollouts (hindsight / DART / forked counterfactuals)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .env import CardinalEnv
from .generator import levels
from .layout import Layout
from .planner import Committed, Planner
from .policies import TargetPlanner, choose_target, make_policy, _rand_from


@dataclass
class Step:
    pos: tuple
    mask: int
    rtg: int  # tenths; RTG shown to the model *before* the action
    action: int  # action shown in context (executed)
    reward: int
    supervise: bool  # loss applies to this action token
    opt: int  # 4-bit mask of A(s, rtg)  (planner-defined RTG-conditioned optimal actions)


@dataclass
class Trajectory:
    map_seed: int
    source: str  # hindsight | dart | fork
    policy: str
    steps: list
    ret: int
    end: str  # goal | lava | timeout
    target: int | None = None  # commanded return (dart/fork) else None
    meta: dict = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.end == "goal"

    def to_dict(self):
        return {
            "map_seed": self.map_seed, "source": self.source, "policy": self.policy, "ret": self.ret,
            "end": self.end, "target": self.target, "meta": self.meta,
            "steps": [[s.pos[0], s.pos[1], s.mask, s.rtg, s.action, s.reward, int(s.supervise), s.opt]
                      for s in self.steps],
        }

    @staticmethod
    def from_dict(d):
        st = [Step((a[0], a[1]), a[2], a[3], a[4], a[5], bool(a[6]), a[7]) for a in d["steps"]]
        return Trajectory(d["map_seed"], d["source"], d["policy"], st, d["ret"], d["end"], d["target"], d["meta"])


def _optmask(planner, pos, mask, rtg):
    A, _ = planner.action_set(pos, mask, rtg)
    return sum(1 << a for a in A)


def _end(env, res):
    return "goal" if res.event == "goal" else "lava" if res.event == "lava" else "timeout"


def rollout_hindsight(layout: Layout, planner: Planner, horizon: int, policy, rng, map_seed=-1) -> Trajectory:
    """Behavior-policy rollout; RTG is the realized return-to-go (standard DT hindsight labeling)."""
    env = CardinalEnv(layout, horizon)
    policy.reset(layout, planner, rng)
    raw = []
    while not env.done:
        pos, mask, cum0 = env.pos, env.mask, env.cum
        a = policy.act(env)
        res = env.step(a)
        raw.append((pos, mask, cum0, a, res.reward))
    ret = env.cum
    steps = [Step(p, m, ret - c, a, r, True, _optmask(planner, p, m, ret - c)) for p, m, c, a, r in raw]
    return Trajectory(map_seed, "hindsight", policy.name, steps, ret, _end(env, res))


def _continue_dart(env, planner, rng, target, noise, raw, force_no_sup=0):
    """Continue an env rollout with planner labels toward commanded `target`; noisy executed actions
    are shown in context but their loss is masked (labels only supervised where exec == label)."""
    res = None
    router = Committed(planner, rng)
    while not env.done:
        pos, mask, cum0 = env.pos, env.mask, env.cum
        rtg = target - cum0
        label, ok = router.act(pos, mask, rtg)
        a = int(rng.integers(4)) if rng.random() < noise else label
        res = env.step(a)
        raw.append((pos, mask, rtg, a, res.reward, a == label and ok))
    return res


def rollout_dart(layout, planner, horizon, rng, noise, target=None, map_seed=-1) -> Trajectory:
    env = CardinalEnv(layout, horizon)
    target = target if target is not None else choose_target(planner, rng)
    raw = []
    res = _continue_dart(env, planner, rng, target, noise, raw)
    steps = [Step(p, m, rtg, a, r, sup, _optmask(planner, p, m, rtg)) for p, m, rtg, a, r, sup in raw]
    return Trajectory(map_seed, "dart", f"dart{noise:g}", steps, env.cum, _end(env, res), target=target)


def rollout_fork(layout, planner, horizon, rng, map_seed=-1, tries=8):
    """Counterfactual pair generator: a shared (state, action, reward) prefix labeled with two different
    commanded returns whose optimal action sets at the fork state are disjoint. Returns a list of
    Trajectories (one per commanded target); prefix steps are context-only (supervise=False)."""
    fr = planner.frontier()
    lv = levels(fr, 10)
    if len(lv) < 2:
        return []
    for _ in range(tries):
        Ra = lv[int(rng.integers(len(lv)))]
        base = rollout_dart(layout, planner, horizon, rng, 0.0, target=Ra, map_seed=map_seed)
        n = len(base.steps)
        if n < 3:
            continue
        cand = []
        for t in range(1, n - 1):
            s = base.steps[t]
            cum = Ra - s.rtg  # shared prefix return so far
            for Rb in lv:
                if Rb == Ra:
                    continue
                Aa = planner.action_set(s.pos, s.mask, Ra - cum)[0]
                Ab = planner.action_set(s.pos, s.mask, Rb - cum)[0]
                if Aa and Ab and not (Aa & Ab):
                    cand.append((t, Rb))
        if not cand:
            continue
        t, Rb = cand[int(rng.integers(len(cand)))]
        out = []
        for R in (Ra, Rb):
            env = CardinalEnv(layout, horizon)
            raw = []
            for s in base.steps[:t]:  # replay shared prefix
                rtg = R - env.cum
                res = env.step(s.action)
                raw.append((s.pos, s.mask, rtg, s.action, res.reward, False))
            res = _continue_dart(env, planner, rng, R, 0.0, raw)
            steps = [Step(p, m, rtg, a, r, sup, _optmask(planner, p, m, rtg)) for p, m, rtg, a, r, sup in raw]
            out.append(Trajectory(map_seed, "fork", "fork", steps, env.cum, _end(env, res), target=R,
                                  meta={"fork_t": t, "pair": [Ra, Rb]}))
        return out
    return []


# quality mixture (per trajectory), see experiment plan §10
MIXTURE = (
    ("clean", 0.35), ("dart", 0.15), ("fork", 0.10),
    ("eps", 0.05), ("boltz", 0.05), ("detour", 0.03), ("myopic", 0.02),
    ("partial", 0.10), ("random", 0.04), ("weak_greedy", 0.03), ("lava_blind", 0.03),
    ("dart_hi", 0.05),
)


def generate_for_layout(layout, planner, horizon, rng, n_traj, map_seed=-1, mixture=MIXTURE):
    kinds = [k for k, _ in mixture]
    w = np.array([p for _, p in mixture], dtype=float)
    w /= w.sum()
    out = []
    while len(out) < n_traj:
        kind = kinds[int(rng.choice(len(kinds), p=w))]
        if kind == "dart":
            out.append(rollout_dart(layout, planner, horizon, rng, float(rng.choice([0.05, 0.1])), map_seed=map_seed))
        elif kind == "dart_hi":
            out.append(rollout_dart(layout, planner, horizon, rng, 0.2, map_seed=map_seed))
        elif kind == "fork":
            out.extend(rollout_fork(layout, planner, horizon, rng, map_seed=map_seed))
        else:
            out.append(rollout_hindsight(layout, planner, horizon, make_policy(kind, rng), rng, map_seed))
    return out[:n_traj] if len(out) > n_traj else out
