"""Closed-loop metrics + cluster bootstrap over maps."""
from __future__ import annotations

import numpy as np


def _strategy_ok(res) -> bool | None:
    fr = res.job.planner.frontier()
    Fs = {F for F, r in fr.items() if r == res.job.target}
    if not Fs:
        return None  # target not on the efficient frontier
    return res.mask in Fs and res.success


def _opt_steps(res):
    fr = res.job.planner.frontier()
    Fs = [F for F, r in fr.items() if r == res.job.target]
    if not Fs:
        return None
    from .. import config as C
    return min(res.job.planner.steps_F(F, tuple(res.job.layout.start), 0) for F in Fs)


def episode_row(res) -> dict:
    steps = res.steps
    opt_steps = [s for s in steps if s["opt"]]
    probs = np.array([s["probs"] for s in steps])
    ent = float(-(probs * np.log(probs + 1e-12)).sum(1).mean())
    so = _strategy_ok(res)
    eff = None
    if so:
        o = _opt_steps(res)
        eff = o / len(steps) if o else None
    return {
        "map": res.job.tag.get("map", -1), "target": res.job.target, "ret": res.ret / 10, "success": float(res.success),
        "cond_err": abs(res.job.target - res.ret) / 10.0, "length": len(steps),
        "strategy_acc": None if so is None else float(so), "efficiency": eff,
        "opt_action_acc": float(np.mean([(s["opt"] >> s["action"]) & 1 for s in opt_steps])) if opt_steps else None,
        "fail_lava": float(res.end == "lava"), "fail_timeout": float(res.end == "timeout"), "entropy": ent,
    }


def summarize(results, n_boot=1000, seed=0) -> dict:
    """Mean and 95% cluster-bootstrap CI (resampling maps) of each metric."""
    rows = [episode_row(r) for r in results]
    maps = sorted({r["map"] for r in rows})
    by = {m: [r for r in rows if r["map"] == m] for m in maps}
    rng = np.random.default_rng(seed)
    out = {}
    keys = [k for k in rows[0] if k not in ("map", "target")]

    def agg(sel_maps, k):
        v = [r[k] for m in sel_maps for r in by[m] if r[k] is not None]
        return float(np.mean(v)) if v else float("nan")

    for k in keys:
        est = agg(maps, k)
        boots = [agg(rng.choice(maps, len(maps)), k) for _ in range(n_boot)]
        out[k] = {"mean": est, "lo": float(np.nanpercentile(boots, 2.5)), "hi": float(np.nanpercentile(boots, 97.5))}
    out["n_episodes"], out["n_maps"] = len(rows), len(maps)
    return out


def desired_vs_achieved(results):
    """Rows for the main plot: (desired, achieved, map) in reward units + regression slope/R^2."""
    d = np.array([r.job.target / 10 for r in results])
    a = np.array([r.ret / 10 for r in results])
    slope, icpt = np.polyfit(d, a, 1) if len(d) > 1 and d.std() > 0 else (float("nan"), float("nan"))
    r2 = float(np.corrcoef(d, a)[0, 1] ** 2) if len(d) > 1 and d.std() > 0 and a.std() > 0 else float("nan")
    shown = np.array([(r.job.target if r.job.shown_target is None else r.job.shown_target) / 10 for r in results])
    slope_shown = float(np.polyfit(shown, a, 1)[0]) if shown.std() > 0 else float("nan")
    return {"desired": d.tolist(), "shown": shown.tolist(), "achieved": a.tolist(), "slope": float(slope),
            "slope_vs_shown": slope_shown, "intercept": float(icpt), "r2": r2}
