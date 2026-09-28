# Design notes and empirical findings

## Environment (MiniGrid-generated, cardinal actions, fully observable)
* Rewards (tenths): step −0.1, wall bump −0.5 (extra), coin +2, treasure +5, goal +10 (terminal), lava −5 (terminal).
  All four actions are always legal (wall bumps are penalized, not masked, so the action mask leaks no map information).
* Map filters (exact planner): collect-all route exists; ≥4 return levels separated by ≥1.0; every bonus object gives
  ≥ +0.5 net for some strategy (otherwise it is not a real strategy, just noise).
* Object values are deliberately *not* generic: a coin (+2.0) costs exactly 20 detour steps, so **different object sets
  can tie exactly on return**. The RTG-conditioned optimal set A(s,R) is therefore the union over tied strategies (used
  for scoring), while data generation uses a *strategy-committed* router (`planner.Committed`); a memoryless
  random tie-break oscillated between tied strategies and timed out on ~2% of "clean" 19×19 trajectories.

## Horizons (empirical, `scripts/horizon_stats.py`)
| spec | goal-only shortest p50 / p99 | collect-all route p50 / p99 | horizon H |
|---|---|---|---|
| mg8  | 7 / 15 | 13 / 22 | **48** (≈2.2× p99) |
| mg19 | 23 / 42 | 48 / 84 | **150** (≈1.8× p99; 1.5× p99 = 126) |

## Serialization
`R:+15.1|P:07,12|M:101|A: UP` per step (RTG is 5 fixed-width tokens; digits are single tokens in Qwen).
Map = one space-separated character per cell (1 token/cell). ~525-token header at 19×19, ~24 tokens/step.
Tokenization is piecewise per field so rollout prompts and training sequences are identical by construction.
Loss: only the action token, restricted 4-way softmax over the action-token logits; `lm_head` is evaluated only at
readout positions (no full-vocab logits).

## Data sources
1. **hindsight** — behavior policies (clean planner, ε-greedy, Boltzmann, detour, myopic, partial, random, weak-greedy,
   lava-blind); RTG = realized return-to-go.
2. **dart** — commanded target R*, noisy execution, actions labeled by the committed RTG-aware planner. Noisy steps stay in
   context but are loss-masked; RTG = R* − return so far (exactly the test-time bookkeeping).
3. **fork** — counterfactual pairs: identical (state, action, reward) prefix, two commanded returns whose optimal action
   sets at the fork state are *disjoint*; prefix is context-only.

Measured 19×19 mixture (3600 trajectories, 300 maps): 85% goal / 12% timeout / 3% lava; mean 61 steps (≈2.0k tokens);
8.3 distinct initial RTGs per map; **68% of supervised decisions are RTG-sensitive** (two feasible levels demand disjoint
actions), so the model cannot ignore RTG and still fit the data. 100k trajectories ≈ 200M tokens.

## Known caveats
* Return distribution has a heavy negative tail (failure trajectories bump walls for up to H steps); RTG width allows ≥ −99.9.
* Return error alone is gameable (dithering); `metrics.py` also reports strategy accuracy and efficiency.
* History format repeats the RTG at every step, so single-position patching can be diluted by backup copies — use
  first-decision prompts (one RTG token) or patch all RTG spans jointly.
* Real-Qwen tokenizer behavior is unverified in the dev sandbox (no HF access): run `scripts/check_tokenizer.py`.
