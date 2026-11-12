# Path D4 experiment log

Entries are written as results arrive; nothing below is edited after the fact (corrections are appended).

## Pre-registration (before any D4 training)
- **Benchmark windows:** frozen in `experiments/pd_d4.py::BENCH` after checking scene continuity on contact sheets
  (`runs/path_d/d4/renders/scene*_every4.png`).
  - Scene 3998 is one continuous scene (3996–4182) with changing motion types: growth, near-static, a second figure
    rising, arm motion, slow motion, zoom.
  - Train pool 3998 / 4014 / 4066 / 4150; val 4038 / 4078 / 4162; test 4022 / 4090 / 4170 + test16 4106–4122.
  - Single-window clips 1221 (translation), 4500 (contour), 1499 (wipe).
- **Selection uses val only** (`experiments/pd_d4_select.py` reads no test field).
- **Metrics:** `experiments/pd_vis.py`.
  - Shape: foreground IoU (foreground = not the original frame's border colour), boundary F1 (tolerance 1 px), thin
    recall, component counts, area and centroid drift.
  - Motion: lag-1/4 change precision, recall, F1, mass, on/off balance.
  - Degeneration: constant frames, terminal period, 2×2 spatial entropy vs the original, first IoU < 0.5.
- **Strong-progress bar:** on the untouched 16-frame test (4106–4122), error below hold and the prior readable rule,
  lag-1 change F1 ≥ 0.30, meaningful IoU and boundary F1, in more than one run.

## Diagnosis of the round-2 64-gate circuit (question 1)
Continuous from 3996, 64 free steps:
- **It freezes into a fixed wedge-shaped silhouette.** Terminal period 1; IoU < 0.5 from step 12. IoU 0.17, boundary
  F1 0.33, foreground area 0.69 of the true area (the true figure keeps growing), 2×2 entropy 0.55 vs 0.76.
- **The motion it makes is sparse:** lag-1 change precision 0.76 but recall 0.25.
- **The silhouette is the circuit's own attractor**, not a degraded copy of the target.
- **Contributing defect found now:** the round-2 8-frame clip runs trained the last curriculum stage with horizon 16
  on an 11-frame sequence. JAX clamped the out-of-range indices, so that stage taught "repeat the last frame".
  - This affects the D3 L8 results in README §11, which stay as recorded.
  - The 16-frame runs were unaffected.
  - Fixed in `pd25.train`: only horizons whose targets exist, with an assertion.
  - Render: `runs/path_d/d4/renders/old64_continuous_3996.png`.

## D4-A: multi-window training (hypothesis: one short window is too little trajectory diversity)
- **Configs**, equal updates (96 k step-evaluations), 2 seeds, k = 2, 2-frame prefix, original loss:
  A one window (3998), B two (3998, 4066), C four (3998, 4014, 4066, 4150), D contiguous 3998–4022 (24 frames).
- **Controls:** at equal updates, A is the repeated-exposure-of-one-window control. Aq is A at ¼ of the updates
  (equal per-window exposure to C).
- **Result (val only; validation windows move little, hold error 0.022):**

  | config | train change F1 | val change F1 | val error |
  |---|---|---|---|
  | A one window | 0.49–0.75 | 0.03 | 0.052–0.075 |
  | B two windows | 0.24–0.25 | 0.05–0.07 | 0.035–0.068 |
  | C four windows | 0.12–0.23 | 0.03 | 0.087–0.097 |
  | D contiguous | 0.13–0.17 | 0.02–0.05 | 0.069–0.160 |
  | Aq ¼ budget | 0.35–0.41 | 0.03–0.05 | 0.031–0.061 |

  - **No configuration beats hold on val error.**
  - The prior readable rule is far worse (0.187); the memoryless table is at 0.030–0.049.
  - Validation IoU is 0.42–0.86: the silhouettes stay roughly in place because the windows barely move.
- **Interpretation:** more windows make the circuit a compromise (training F1 falls from 0.49–0.75 to 0.12–0.25)
  without transferring motion. The poor transfer is not a data-quantity problem at this circuit size and objective.
- **Decision:** reject multi-window as the fix. Keep B (best val F1, marginally) as the D4-B starting configuration.
- **Next:** D4-B motion-aware loss.
- **GPU:** 42.2 min.

## D4-B: motion-aware objective (hypothesis: pixel loss rewards under-changing)
Config B, 96 k, 2 seeds. Loss (a contour band weight, b change weight, c temporal-change term):

| variant | a | b | c |
|---|---|---|---|
| base | 0 | 0 | 0 (D4-A runs) |
| contour | 2 | 0 | 0 |
| change | 0 | 4 | 1 |
| both | 2 | 4 | 1 |

## D4-C diagnosis (run early, on the D4-A circuits; CPU only, `experiments/pd_d4_harden.py`)
- **Gates:** the middle update layers are undecided. 40–72 % of gates have max softmax p < 0.9, and the per-layer
  soft/hard disagreement on the same visited inputs reaches 12–22 %. The final layer agrees anyway (≤ 0.15 %).
- **Rollout:** soft and hard visible outputs diverge gradually: 0.001–0.007 at step 1, 0.07–0.12 at step 16. On the
  training window the hard error is 1.7–3× the soft error by step 7 (A s1: 0.021 vs 0.007; C s1: 0.070 vs 0.042).
- **The decisive test:** the soft circuit itself on the val windows. Soft val change F1 is 0.02–0.10 (A s1, B s1, C s1),
  the same failure as the hard circuits (0.01–0.08).
- **Conclusion:** the soft-to-hard gap costs fitted-window accuracy, but it is **not** why motion fails to transfer.
- **Decision:** no GPU spent on hardening interventions (gate temperature, gradual hardening): the measured bottleneck
  is elsewhere.
- **D4-B result (val only):** change F1 is 0.03–0.07 in every variant, the same as base (0.05–0.07).
  - **change** (b 4, c 1): recall up a little (0.14–0.18) but error and IoU worse (val error 0.079–0.143, IoU
    0.45–0.65). Flicker: **rejected**.
  - **contour:** no consistent effect.
  - **both:** best of s0 (error 0.047, boundary F1 0.92) but not better than base on s1.
- **Decision:** no objective fixes transfer. Keep the original loss as default; "both" is neutral.
- **GPU:** 27.3 min.

## Can any local rule transfer in this scene? (CPU, `runs/path_d/d4/table_transfer_check.json`)
Exact lookup tables fitted on all four training windows, scored one step ahead on the val windows (hold in brackets):
- 3×3: 0.009 / 0.017 / 0.012 (hold 0.006 / 0.011 / 0.002);
- 3×3 + one previous frame (= what hidden memory could give): 0.008 / 0.015 / 0.004;
- 5×5: 0.009 / 0.017 / 0.010.

None beats hold. Within the 3998 scene the local motion law differs between windows (cf. Readable Rules FINDINGS
§10a for the whole song).
- **Decision:** a bigger neighbourhood, more memory or more depth cannot fix *transfer*, so no D4-E for transfer.

## D4-E (one controlled question): what limits *fitting* a longer contiguous stretch?
- Evidence: D (24 contiguous frames) fits badly (train change F1 0.13–0.17 vs 0.49–0.75 for 8 frames).
- Controlled test, 2 seeds each:
  - D with k = 4 hidden channels (96 k);
  - D with 3× budget (288 k, k = 2), the optimization control.
- Also: the single-window secondary clips 1221, 4500, 1499 (base loss, seed 0).
- **D4-E result** (train metrics are over the 24 fitted frames):

  | run | train error | train change F1 |
  |---|---|---|
  | D k = 2, 96 k | 0.082 / 0.055 | 0.17 / 0.13 |
  | D k = 4, 96 k | 0.046 | 0.16 (s1 collapsed to a 1-gate circuit, error 0.95) |
  | D k = 2, 288 k | 0.056 / 0.047 | 0.15 / 0.19 |

  - Val change F1 ≤ 0.05 everywhere.
  - **Neither more latent channels nor 3× optimization makes a 24-frame contiguous fit work. Rejected.**
- **Secondary clips** (single window, val = the next 8 frames):
  - 1221: val change F1 0.17 (readable 0.10) but error 0.069 vs hold 0.036;
  - 4500: change F1 0.01;
  - 1499: error = hold.
- **GPU so far:** 113 min = 1.88 GPU-h (one run stopped after ~10 s: it started before a channel-count fix; rerun as
  job 31).

## Amendment, registered before running: within-phase transfer (config G)
- **Evidence:** all D4 val windows lie in a *different motion phase* from the training windows (near-static 4038,
  rising 4078, zoom 4162). The round-2 3998 L16 circuit transferred to the *same-phase* continuation 4014–4030
  (F1 0.62, mass 0.70).
- **Hypothesis:** motion transfers within a phase but not across phases.
- **Config G:** train the contiguous 3998–4014 (16 frames, growth phase), val 4014–4022 (same phase).
- **Test:** the already-frozen untouched 4022–4030 and test16 4106–4122, plus a continuous run 3998→4030 (reported at
  the end only).
- **Runs:** loss base and "both" (2, 4, 1), seeds 0 / 1, budget 144 k.
- **Config G result** (val 4014–4022: hold 0.054, readable 0.058, table 0.043; readable change F1 0.06):

  | run | val error | change F1 | boundary F1 |
  |---|---|---|---|
  | base s0 | 0.041 | 0.16 | |
  | base s1 | 0.033 | 0.13 | 0.92 |
  | both s0 | 0.049 | 0.08 | |
  | both s1 | 0.041 | 0.14 | |

  - All four beat hold and readable on error; 3/4 beat the table.
  - Change F1 is 2–3× the cross-phase runs (0.03–0.07), but far below the 0.30 milestone.
  - **Interpretation:** within-phase transfer exists but is weak. The "both" loss does not help.
- **Correction to the amendment rationale:** the round-2 "F1 0.62" was the old evaluator's lag-4 change F1 *with spatial
  tolerance* (`rr_run.evaluate`). D4 reports exact-pixel lag-1/lag-4 change F1. The numbers are not comparable, so the
  amendment's evidence was weaker than stated. The amendment itself was registered before running and stands.
- **Selection (val change F1, tie → val error), fixed now:**
  - within-phase: **G_s0_b144k** (second seed G_s1_b144k reported alongside);
  - cross-phase benchmark: **B_s1_b96k**.
- **GPU so far:** 137 min = 2.29 GPU-h.

## D4-D: simplification of the selected circuits
Exact (simplify + ABC, SAT-checked) is already applied to every run. Behaviour-aware unit tying is scored on the run's
**training windows only** (fresh-start error may rise ≤ 0.003); val is re-scored, test only in the final report.
- **Pruning on pixel error only fails:**
  - G_s0 503 → **1 gate** at the same train error, foreground IoU 0.00: a degenerate output. The pixel metric is
    blind to shape.
  - G_s1 → 16 gates (val IoU 0.28); B_s1 → 34 gates.
- **Rejected**; replaced by a shape-aware criterion: train error +0.003, foreground IoU −0.02, change F1 −0.03.
- **Shape-aware result:**

  | circuit | gates | val error | val change F1 | val IoU |
  |---|---|---|---|---|
  | G_s0 | 503 → **30** | 0.051 (still < hold 0.054) | 0.16 → 0.10 | 0.49 |
  | G_s1 | 476 → 84 | 0.037 | 0.13 → 0.05 | |
  | B_s1 | 432 → 106 | val unchanged | | |

- **Decision:** the unpruned G_s0 stays the primary candidate (better motion). The 30-gate version is reported as
  the compact variant.
- **Selection final** → test windows and renders are run now, once, for G_s0, G_s1, G_s0__pruned_shape and B_s1.
- **Test, run once after selection** (full table in README §13):
  - same-phase 4022–4030: G_s0 error 0.032, change F1 0.18, IoU 0.72; G_s1 0.024 / 0.12 / 0.80. The round-2 circuit:
    0.036 / 0.07 / 0.69.
  - other-phase 4106–4122: every circuit is worse than hold.
  - continuous: IoU < 0.5 by step 5–9; the figure turns into a growing blob.
- **Stop:** five controlled attempts gave no cross-phase transfer. Budget left unused on purpose (2.28 / 8 GPU-h).
- **Verdict: PARTIAL.**
