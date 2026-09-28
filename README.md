# grid-world-rl — return-conditioned LLM policies on MiniGrid-generated gridworlds

Fine-tune a pretrained causal LM (Qwen3-8B-Base + LoRA) as a **return-conditioned policy** π(a | s, RTG, history) on
text-serialized gridworld trajectories, then study *where and how the desired return is represented and causally used*.
No new architecture: the LM reads a fixed-width text format and the action is read out as a restricted 4-way softmax.

See [`docs/DESIGN.md`](docs/DESIGN.md) for the experimental design and the empirical findings that shaped it.

## Layout

| Module | Purpose |
|---|---|
| `rtgrid/generator.py` | Layouts built with **MiniGrid** (`MiniGridEnv`, `Grid`, `Wall/Goal/Lava/Ball/Box`): 19×19 = 3×3 rooms of 5×5, doorways (spanning tree + loops), obstacles, lava, coins (Ball), treasure (Box). Map-quality filters via the exact planner |
| `rtgrid/env.py` | Cardinal-move simulator (UP/DOWN/LEFT/RIGHT), integer-tenth rewards, terminal goal/lava, horizon truncation |
| `rtgrid/planner.py` | **Exact** planner: efficient-return frontier, RTG-conditioned optimal action sets A(s,R), strategy-committed router |
| `rtgrid/policies.py`, `trajectory.py` | Behavior policies; hindsight-relabeled rollouts; **DART** (noise + RTG-aware labels); **fork** counterfactual pairs |
| `rtgrid/serialize.py` | Fixed-width text format, piecewise tokenization, loss layout (action tokens only) |
| `rtgrid/dataset.py` | Split-by-map with disjoint seeds **and** dihedral-canonical layout hashes; RTG-band holdout |
| `rtgrid/llm/` | Restricted action readout, SFT (LoRA), batched KV-cached closed-loop eval, sensitivity (JSD/TV/margin), metrics with cluster bootstrap |
| `rtgrid/interp/tl.py` | TransformerLens (`TransformerBridge`) patching / steering / directional ablation / group-CV probes |

## Quick start

```bash
pip install -e ".[llm,interp]"
pytest -q                                            # 30 tests (planner is validated against brute-force BFS)

python scripts/horizon_stats.py --spec mg19 --n 500  # empirical horizon selection
python scripts/check_tokenizer.py --tokenizer Qwen/Qwen3-8B-Base   # MUST pass on the real tokenizer

# 8x8 smoke / pilot data
python scripts/gen_dataset.py --spec mg8 --out data/mg8 --n-train 3000 --n-val 200 --n-test 300 --traj-per-map 8
python scripts/dataset_stats.py --data data/mg8

# SFT (one process per GPU; independent seeds / controls in parallel)
CUDA_VISIBLE_DEVICES=0 python -m rtgrid.llm.sft --data data/mg8 --out runs/mg8_rtg   --rtg-mode correct --steps 1000
CUDA_VISIBLE_DEVICES=1 python -m rtgrid.llm.sft --data data/mg8 --out runs/mg8_nortg --rtg-mode none    --steps 1000
CUDA_VISIBLE_DEVICES=3 python -m rtgrid.llm.sft --data data/mg8 --out runs/mg8_shuf  --rtg-mode shuffled --steps 1000

# closed-loop eval: same maps / RTG levels / prompt / readout for every checkpoint
python -m rtgrid.llm.eval_cli --data data/mg8 --spec mg8 --adapter runs/mg8_rtg/ckpt_1000 --rtg-mode correct --out results/sft.json
python -m rtgrid.llm.eval_cli --data data/mg8 --spec mg8 --rtg-mode correct --out results/base.json      # untouched base
python -m rtgrid.llm.eval_cli --data data/mg8 --spec mg8 --adapter runs/mg8_rtg/ckpt_1000 --rtg-mode shuffled --out results/sft_shuf.json
```

Everything except real-Qwen/GPU execution is tested on CPU with a tiny random Qwen3 (`tests/test_llm_tiny.py`).
