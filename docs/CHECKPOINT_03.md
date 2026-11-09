# Checkpoint 03: length scaling and hard-rule optimization

Plan and budget: `docs/CAMPAIGN_03.md`. All jobs ran from frozen snapshot `runs/frozen/b1f106e851cb`. Tables come from `experiments/summarize03.py`; rerun it to refresh once the pending jobs finish.

## Report fixes (CHECKPOINT_02)
- **A–E table:** each condition's k and S are now listed explicitly.
- **Scopes:** every impossibility certificate now states its scope (family, init, boundary, S).
- **Statuses:** SAT, UNSAT and UNKNOWN are defined separately.
- **Sizes:** min-exception results are labelled "upper bound only (no proof)".

## A. Length scaling
Setting: exact only, ba32_4500 start frame 4500, F_priv k=1, S=2, 30 min CPU each.

Size accounting:
- Rule and recording bits both come from decode-verified bitstreams in `ca/costs.py`.
- The shared header and the first frame (768 raw bits) are excluded from both sides.
- The light cone passes at S=2 for all four lengths. L=40 also has a memoryless state repeat (frames 28→29).

| frames | changed px | status | first solution (s, E) | best E (s) | rule bits | changelist | xor_rle | frame_rle | rule / best recording |
|---|---|---|---|---|---|---|---|---|---|
| 5 | 74 | upper bound only | 14.6 s, 561 | 217 (1096 s) | 1159 | 678 | 878 | 1274 | 1.71 |
| 10 | 129 | upper bound only | 367 s, 573 | 572 (624 s) | 1940 | 1260 | 1637 | 2791 | 1.54 |
| 20 | 261 | **UNKNOWN_timeout_no_solution** (encode 54 s) | – | – | – | 2692 | 3471 | 6107 | – |
| 40 | 405 | **UNKNOWN_timeout_no_solution** (encode 112 s) | – | – | – | 4476 | 5789 | 13261 | – |

- L20 and L40 found no solution within 30 min. **This is not evidence that a large rule is needed.**
- The fall from 1.71 to 1.54 compares two upper bounds that got very different amounts of search (13 vs 2 improvements). **It is not a trend.**
- Exact rules are still larger than every recording baseline wherever one was found.

## B. Hard-rule optimization, ba32_3000_5 E (F_priv k=1, S=2, boundary 0, hidden init 0)

**Setup**
- 4 fixed wiring seeds (1000–1003) for every GPU method. No seed scaling.
- 600 s training wall per method. GPU methods ran via gpuq; D1 and D2 ran on 1 CPU thread each.
- Checkpoint = best **hard autonomous rollout**. Soft metrics are at that same checkpoint (§B2); the loss trajectories are in each `log.json`.
- Hold-frame baseline error 0.0070. Recording the target costs 195 bits (changelist) / 260 (xor_rle).
- `hard err` averages over all 5 frames, as in CHECKPOINT_02. §B2 errors are over frames 1..4.

**Video-only discovery** (end-to-end, no SAT information):

| id | method | exact | best hard err | change-IoU | live gates | circuit bits | LUT-exc bits | steps / evals |
|---|---|---|---|---|---|---|---|---|
| R0 | relaxation (soft gates, soft states) | 0/4 | **0.0057** | 0.07 | 363 | 7131 | 1626 | 16854 |
| HG | ST gates | 0/4 | 0.1021 | 0.01 | 498 | 10101 | 1590 | 16946 |
| HS | ST states (all) | 0/4 | 0.0070 (= hold) | 0.00 | 37 | 567 | 1948 | 16908 |
| HH | ST hidden only | 0/4 | 0.0070 (= hold) | 0.00 | 6 | 97 | 682 | 16915 |
| HB | ST gates + states | 0/4 | 0.0336 | 0.03 | 46 | 711 | 1556 | 16876 |
| D1 | gate-op hill-climb from R0's circuit (CPU) | 0/1 | 0.0055 (22→21 wrong px) | 0.15 | 363 | 7131 | 1324 | 89,113 evals |
| D2 | SAT repair: exact rule nearest R0's LUT (CPU) | **exact** | 0 | 1.00 | – | – | 1726 (keep-exc 664) | Hamming 282→**78** (upper bound, 491 s) |

**SAT-teacher diagnostics** (not discovery; counted separately):

| id | method | exact | best hard err | live gates | circuit bits | note |
|---|---|---|---|---|---|---|
| T1 | R0 + hidden-trajectory supervision | 0/4 | 0.0070 (= hold) | 48 | 743 | supervising the hidden channel did **not** help at this budget |
| T2 | supervised distillation of the SAT LUT | **1/4** (seed 1003, first 20 s eval) | 0 | 576 | 11967 | the architecture can represent an exact rule, re-confirmed under frozen code |

All exported rules were replayed by Node, and the hash matched the Python rollout.

### B2. Where the soft/hard gap comes from
Selected checkpoints; errors over frames 1..4.

| id | soft g + soft s | soft g + round s | soft g + round hidden | hard g + soft s | hard (hard g + round s) | one-step disagreement (vis, hid) | per-frame hard |
|---|---|---|---|---|---|---|---|
| R0 | **0.0000** | 0.0072 | 0.0039 | 0.0072 | 0.0072 | 0.000, 0.000 | 0, .005, .009, .014 |
| HG | 0.251 | 0.251 | 0.251 | 0.128 | 0.128 | 0.504, 1.000 | .068 → .188 |
| HS | 0.264 | 0.029 | 0.265 | 0.009 | 0.009 | 0.058, 0.516 | .001 → .017 |
| HH | 0.007 | 0.009 | 0.007 | 0.009 | 0.009 | 0, 0 | .001 → .017 |
| HB | 0.251 | 0.251 | 0.251 | 0.042 | 0.042 | 0.461, 0.060 | .022 → .059 |
| T1 | 0.009 | 0.009 | 0.009 | 0.009 | 0.009 | 0, 0.018 | .001 → .017 |
| T2 | 0.041 | 0.000 | 0.013 | 0.000 | 0.000 | 0, 0 | 0, 0, 0, 0 |

**Interpretation**
- **R0: the gap is in the states, not in gate selection.**
  - Soft gates on soft states are exact.
  - Hard and soft gates disagree nowhere, one step at a time, on the hard trajectory.
  - Binarizing the states alone (soft gates kept) reproduces the full hard error. Binarizing only the hidden channel gives half of it (0.0039).
  - The hard error then grows frame by frame (0 → 0.014): the error accumulates.
  - The relaxation solves the clip by passing **graded values through both channels**. Binarizing only the hidden channel is not enough, as you anticipated.
- **Straight-through did not fix it at this budget.**
  - HS and HH collapsed to the identity ("hold") rule. Their soft path (0.26 for HS) and hard path diverge.
  - HG and HB break the forward/backward agreement outright (one-step disagreement 0.46–1.0).
  - These are optimization failures of ST at 600 s and 4 seeds. This is not a demonstration that ST cannot work.
- **T2 shows the reverse gap.** A rule that is exact when hard has a **soft** rollout error of 0.041. The soft model is a different dynamical system, so the loss on it is not a faithful proxy for the hard objective.
- **D1 (local op mutations) is nearly stuck.** 89k evaluations improved one pixel. The hard landscape around R0's circuit is plateau-dominated under single-gate moves.
- **D2 shows R0's hardened rule is at most 78 LUT entries from an exact rule** (out of 2048 LUT entries). The relaxation gets "close" in rollout error, but the exact rule still needs dozens of table changes. For now the solver is what makes the result exact.

## C. Total campaign cost
- **GPU:** 5,007 GPU-s = **1.39 GPU-h** from gpuq records (pilots 903 s, main 4,104 s). The estimate was 1.35 GPU-h; pilot job overhead was 2.1 min, not 1.
- **CPU:**
  - length scaling 4 × 1,800 s;
  - D2 600 s;
  - D1 600 s, plus a crashed first attempt;
  - encoding/verification ≈ 3 min;
  - total ≈ **2.4 CPU-h** (estimate 2.5).
- **Implementation and configuration errors** (kept separate from the findings):
  - **D1's first run crashed at startup:** XLA `pthread_create` failed under a 16 GB address-space cap (`runs/exp07/D1_main_crash_as16G.log`). It was rerun with 64 GB from the same frozen code.
  - **L20/L40 status labelling:** verify said "timeout before encoding" when encoding had finished. Analysis-side fix in `exp03c verify`; the search was unchanged.

## D. Conclusions and the next experiment
1. **Exactness:** only SAT-based methods (direct SAT, D2 repair) produced exact rules from the video alone. Gradient methods reached near-misses (R0 0.0057 against hold 0.0070) but no exact rule.
2. **Size:** every exact rule found is larger than every recording baseline of the same clip (1.5–1.7× at 5–10 frames, upper bounds). Longer clips are still unresolved at a 30-min budget, so compression by a rule is **not shown**, and it is **not refuted** either.
3. **Soft/hard gap:** for ba32_3000 it comes from graded states in both channels and accumulates over the rollout. Gate selection is not the source, and straight-through at this budget makes training collapse or diverge.

**Next (proposed, not run):**
- **(a) Make long clips tractable** instead of only adding time. Build the SAT problem incrementally across lengths: solve L=10, then add frames using the previous rule as phases or assumptions (CEGIS-style). Same 30 min per length, so it is comparable with this run.
- **(b) State-binarization curriculum.** Add a penalty/annealing schedule pushing every state toward {0,1} during training (not ST), selected by hard rollout, against R0 at equal budget. Then D2 repair as the exactness finisher. Measure the Hamming-to-exact distance as the progress metric. (Superseded in CHECKPOINT_04: distance to one known SAT solution is not a valid metric; only a found upper bound to *some* exact rule is reported, and it is not a primary metric.)
