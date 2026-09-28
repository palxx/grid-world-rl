"""End-to-end LLM plumbing on a tiny random Qwen3 (CPU): loss mask, readout, KV-cache rollout, SFT, sweeps."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import Qwen3Config, Qwen3ForCausalLM

from rtgrid.config import SPECS
from rtgrid.dataset import write_dataset
from rtgrid.generator import make_layout
from rtgrid.llm.data import TrajDataset, collate
from rtgrid.llm.metrics import summarize, desired_vs_achieved
from rtgrid.llm.readout import action_logits
from rtgrid.llm.rollout import Job, run_episodes
from rtgrid.llm.sensitivity import jsd, rtg_sweep, tv
from rtgrid.llm.sft import loss_and_stats, train
from rtgrid.planner import Planner
from rtgrid.serialize import action_token_ids, encode_trajectory, header_segments, step_prefix_segments
from rtgrid.stubtok import StubTokenizer
from rtgrid.trajectory import Trajectory, generate_for_layout


def tiny_model(seed=0):
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=4096, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=16384,
                      tie_word_embeddings=False)
    return Qwen3ForCausalLM(cfg).eval()


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("d"))
    write_dataset("mg8", d, 8, 3, 3, traj_per_map=6, workers=2, test_traj_per_map=0)
    tok = StubTokenizer()
    return d, tok


def test_readout_matches_full_vocab_logits(setup):
    d, tok = setup
    m = tiny_model()
    ids = action_token_ids(tok)
    ds = TrajDataset(d, "train", tok)
    b = collate([ds[0], ds[1]])
    with torch.no_grad():
        got = action_logits(m, b["input_ids"], b["attention_mask"], b["positions"], ids)
        full = m(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
    for i in range(2):
        for j in range(int((b["positions"][i] >= 0).sum())):
            want = full[i, b["positions"][i, j]][ids]
            assert torch.allclose(got[i, j], want.float(), atol=1e-4)


def test_loss_only_on_action_tokens(setup):
    d, tok = setup
    ds = TrajDataset(d, "train", tok)
    ex = ds[0]
    ids = action_token_ids(tok)
    assert len(ex["readout"]) == len(ex["target"]) > 0
    assert all(ex["ids"][r + 1] == ids[t] for r, t in zip(ex["readout"], ex["target"]))
    b = collate([ex])
    assert int((b["targets"] != -100).sum()) == len(ex["target"])


def test_cached_rollout_matches_full_recompute(setup):
    d, tok = setup
    m = tiny_model(1)
    ids = action_token_ids(tok)
    spec = SPECS["mg8"]
    jobs = []
    for s in range(6):
        lay, _ = make_layout(spec, s)
        pl = Planner(lay)
        R = sorted(pl.frontier().values())[s % 2]
        jobs.append(Job(lay, pl, R, spec.horizon, {"map": s}))
    res = run_episodes(m, tok, jobs, ids, "cpu", batch_size=4, greedy=True)
    for r in res[:4]:
        # replay the episode with a full (uncached) forward and compare every recorded distribution
        lay, pl = r.job.layout, r.job.planner
        from rtgrid.llm.sensitivity import rtg_sweep
        hist, cum = [], 0
        for st in r.steps:
            rtg = r.job.target
            p = rtg_sweep(m, tok, lay, hist, (st["pos"], st["mask"]), [rtg], ids, "cpu")[0]
            assert np.allclose(p, st["probs"], atol=2e-4), (p, st["probs"])
            hist.append((st["pos"], st["mask"], st["action"], st["reward"]))
    # metrics run on results
    s = summarize(res, n_boot=20)
    assert s["n_episodes"] == 6 and 0 <= s["success"]["mean"] <= 1
    dv = desired_vs_achieved(res)
    assert len(dv["desired"]) == 6


def test_sft_reduces_loss_and_learns(setup, tmp_path):
    d, tok = setup
    m = tiny_model(2).train()
    ids = action_token_ids(tok)
    ds = TrajDataset(d, "train", tok, max_trajs=16)
    b = collate([ds[i] for i in range(4)])
    l0, _ = loss_and_stats(m, b, ids, "cpu")
    train(m, ds, None, ids, str(tmp_path), steps=200, lr=3e-3, bs=4, device="cpu", log_every=30, save_adapters=False)
    l1, _ = loss_and_stats(m, b, ids, "cpu")
    assert l1.item() < l0.item() * 0.7, (l0.item(), l1.item())


def test_rtg_sweep_and_divergence(setup):
    d, tok = setup
    m = tiny_model(3)
    ids = action_token_ids(tok)
    lay, _ = make_layout(SPECS["mg8"], 1)
    pl = Planner(lay)
    levels = sorted(set(pl.frontier().values()))[:3]
    p = rtg_sweep(m, tok, lay, [], (tuple(lay.start), 0), levels, ids, "cpu")
    assert p.shape == (len(levels), 4) and np.allclose(p.sum(1), 1, atol=1e-5)
    assert 0 <= jsd(p[0], p[1]) <= np.log(2) + 1e-9 and 0 <= tv(p[0], p[1]) <= 1
    assert jsd(p[0], p[0]) == pytest.approx(0, abs=1e-9)


def test_interp_bridge_patching_steering_ablation(setup):
    pytest.importorskip("transformer_lens")
    import tempfile
    from rtgrid.interp import tl
    from rtgrid.llm.prompts import build_prompt
    d, tok = setup
    m = tiny_model(4)
    path = tempfile.mkdtemp()
    m.save_pretrained(path)
    bridge = tl.load_bridge(m, path, device="cpu", dtype=torch.float32)
    ids = action_token_ids(tok)
    lay, _ = make_layout(SPECS["mg8"], 2)
    pl = Planner(lay)
    lv = sorted(set(pl.frontier().values()))
    st = (tuple(lay.start), 0)
    lo, hi = build_prompt(tok, lay, [], st, lv[0]), build_prompt(tok, lay, [], st, lv[-1])
    assert len(lo.ids) == len(hi.ids) and len(lo.rtg_spans) == 1 and lo.rtg_spans[0][1] - lo.rtg_spans[0][0] == 5
    # bridge readout == restricted HF readout
    p_hf = rtg_sweep(m, tok, lay, [], st, [lv[0]], ids, "cpu")[0]
    p_br = tl.run_probs(bridge, lo.ids, ids)[0].numpy()
    assert np.allclose(p_hf, p_br, atol=1e-4)
    # patching ALL positions at layer 0 reproduces the clean run exactly (effect == 1 when they differ)
    T = len(lo.ids)
    eff, p_low, p_high = tl.patching_sweep(bridge, lo.ids, hi.ids, [0, 1], {"all": range(T)}, ids, 0b1111)
    clean = tl.cache_resid(bridge, hi.ids, [0])
    p_pat = tl.run_probs(bridge, lo.ids, ids, [tl.patch_hook(0, clean[0], range(T))])
    assert torch.allclose(p_pat, p_high, atol=1e-4)
    # patching only the RTG token span changes the output less than or equal to the full patch; zero steering is a no-op
    v = torch.randn(64)
    p0 = tl.run_probs(bridge, lo.ids, ids)
    assert torch.allclose(p0, tl.run_probs(bridge, lo.ids, ids, [tl.steer_hook(1, v, 0.0, [lo.readout])]), atol=1e-5)
    assert not torch.allclose(p0, tl.run_probs(bridge, lo.ids, ids, [tl.steer_hook(1, v, 50.0, [lo.readout])]), atol=1e-3)
    # ablating a direction leaves zero component along it (hooks on one point run in order)
    seen = {}

    def grab(act, hook):
        seen["x"] = act.clone()
        return act

    tl_h = tl.ablate_hook(1, v)
    bridge.run_with_hooks(torch.tensor([lo.ids]), fwd_hooks=[tl_h, (tl.resid(1), grab)])
    comp = (seen["x"][0] @ (v / v.norm())).abs().max().item()
    assert comp < 1e-4, comp
