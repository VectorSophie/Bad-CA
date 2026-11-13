# Context Stage 0: are Bad Apple's local laws a small shared codebook?

**Question.** FINDINGS §10a showed that each 16-frame window has a local law (its own optimal 3×3 table beats hold) but
the law of one window does not transfer to another. Stage 0 asks whether those laws fall into a **small shared set of
modes**, so that a fixed rule plus a few context bits could select between them. That is the premise of a context-nudged
logic CA. CPU only, exact lookup tables, no learned circuits. Branch `context-s0`.

**Verdict (2026-11-13):**

| question | result |
|---|---|
| global shared codebook (one code per window) | **NO-GO** on held-out scenes: K = 8 recovers 18 % of the per-window gap (34 % at K = 16) and only equals a 49-way global shift |
| observation-only code selection (2-frame prefix) | **NO-GO**: worse than K = 1 on held-out scenes (gap −0.17) |
| autonomous multi-step behaviour | **NO-GO**: rollout change F1 ≤ 0.05; even the per-window *in-sample* table reaches only 0.05–0.11 |
| regional context (coarse grid, oracle) | **PARTIAL**: one-step gap 0.55–0.90 at 12–36 bits per window, rollout F1 still ≤ 0.07 |
| synthetic Boolean controllability | executor sanity only (hand-written, CPU): **passes**. No learning and no GPU spent (see §6) |

## 1. Setup (frozen before any fit: `runs/context_s0/manifest.json`, commit 5b88946)
- **Windows:** the 24 scenes and the 16-frame windows of `runs/readable_rules/gen/prereg_windows.json` (FINDINGS §10).
- **Scene split:** seed 20261113, 12 fit / 6 val / 6 test scenes.
- **Fitting pool:** 175 tiles of 16 frames at stride 16 over the whole video, after excluding:
  - every val and test scene (±2 frames);
  - every evaluation window together with its 3-frame prefix (±2).
- **Evaluation windows:**
  - `same_scene`: adjacent and distant windows of the fit scenes, 24 windows (their train windows are pool-eligible);
  - `cross_scene`: train, adjacent and distant windows of the val and test scenes;
  - `cut`: windows that straddle a wipe or cut.
- **Counts:** val 24 windows, test 60. Moving windows (true change ≥ 0.005 per transition): val cross-scene 18, test
  cross-scene 18, same-scene 20, cut 17.
- **Rules:**
  - `m0`: memoryless 3×3, 9 bits; this is the §10a R = 1 table;
  - `m1`: 3×3 of frame t plus 3×3 of frame t−1, 18 bits; this is the D4 locality check.
  - Majority tables, with zero padding as in §10a and D4. An unseen pattern or a tie holds the cell.
- **Codebook:** hard-EM, i.e. k-means over windows with exact majority tables and a k-means++-style init by loss.
  K ∈ {1, 2, 4, 8, 16, 32}, 8 restarts each; the restart with the lowest *training* loss is kept.
- **Assignments on held-out windows (codebook frozen):**
  - `oracle1`: argmin one-step error over the whole window (uses the ground truth);
  - `oracle_roll`: argmin rollout error;
  - `obs`: argmin one-step error on the 2 transitions *before* the window, ties to the most frequent code;
  - `knn`: 5-NN on prefix motion features against pool windows;
  - `prior`: the most frequent code.
- **Baselines:**
  - hold;
  - K = 1;
  - per-window table, fitted *in-sample* on the evaluated window itself: a much less restricted oracle;
  - shift oracle and shift obs: the best constant (dy, dx) in [−3, 3]², 49 codes;
  - Readable Rules G1 on the same windows.
- **K selection on val:** the smallest K whose val oracle gap recovery is ≥ 0.9 × the maximum over K. This gave
  **K = 8** for both m0 and m1. Test was then read once.
- **Gap recovery:** g = (E[K=1] − E[K]) / (E[K=1] − E[per-window]), where E is the mean one-step error over the
  window set (unweighted).

## 2. Results (m0, moving windows, unweighted means)

### Val (selection), cross-scene, n = 18
Hold 0.0306, K = 1 0.0309, per-window 0.0230.

| K | oracle one-step | gap | beats hold | obs one-step | rollout (oracle code) / hold | rollout change F1 |
|---|---|---|---|---|---|---|
| 2 | 0.0284 | 0.32 | 0.39 | 0.0301 | 0.127 / 0.137 | 0.03 |
| 4 | 0.0282 | 0.35 | 0.39 | 0.0301 | 0.124 | 0.04 |
| **8** | **0.0271** | **0.48** | **0.61** | 0.0302 | **0.120** | 0.05 |
| 16 | 0.0282 | 0.35 | 0.50 | 0.0341 | 0.129 | 0.03 |
| 32 | 0.0275 | 0.43 | 0.61 | 0.0316 | 0.120 | 0.04 |

### Test, scored once after selection

| group (n) | hold | K = 1 | per-window | K = 8 oracle (gap, beats hold) | K = 16 oracle (gap) | K = 8 obs (gap) | shift oracle / obs | rollout K = 8 / hold, change F1 |
|---|---|---|---|---|---|---|---|---|
| cross-scene (18) | 0.0257 | 0.0259 | 0.0205 | 0.0249 (0.18, 0.44) | 0.0241 (0.34) | 0.0268 (−0.17) | **0.0250** / 0.0294 | 0.116 / 0.120, 0.02 |
| same-scene (20) | 0.0257 | 0.0256 | 0.0190 | 0.0226 (0.45, 0.50) | 0.0229 (0.40) | 0.0250 (0.09) | 0.0228 / 0.0261 | 0.112 / 0.129, 0.03 |
| cut (17) | 0.0735 | 0.0734 | 0.0605 | 0.0688 (0.36, 0.94) | 0.0677 (0.44) | 0.0727 (0.06) | 0.0706 / 0.0742 | 0.212 / 0.215, 0.04 |

**m1** (with the previous frame), test cross-scene: K = 1 0.0253, per-window (in-sample) 0.0147, K = 8 oracle 0.0232
(gap 0.20), obs 0.0252, rollout change F1 0.04. The per-window m1 table has 18 input bits and only 15 × 1,728 samples,
so it largely memorises; its gap denominator is inflated.

**Other details:**
- **Rollout horizon** (K = 8 oracle, cross-scene), error at step 1/4/8/15: 0.027 / 0.076 / 0.125 / 0.177. The rule
  freezes (terminal period 1) in 72 % of the windows.
- **Strata** (test, K = 8, by true change rate 0.005–0.015 / 0.015–0.04 / ≥ 0.04): n = 12/18/25; oracle gap
  0.09 / 0.34 / 0.36; obs gap −0.39 / −0.12 / 0.07.
- **Restart spread** (val oracle error over all val windows including cuts, 8 restarts): K = 8 0.0383–0.0403,
  K = 16 0.0383–0.0389.
- **Readable Rules G1** on the 12 fit scenes' adjacent windows (all, not only moving), rollout error: staged 0.086,
  staged k = 1 0.080, rollout-only 0.100, shift 0.074, hold 0.074. The K = 8 codebook gives 0.066 (oracle) and 0.069
  (obs). It is better than G1, but mostly by staying close to hold.
- **Near-duplicates:** 2 of 84 evaluation windows have ≥ 8 frames within 9 px of a pool frame. Nothing was dropped.

## 3. Regional context (m0, oracle per region, moving windows, test cross+same scene, n = 38)
Reference on the same windows: global K = 1 0.0257, per-window 0.0197.

| grid | K | bits / window | one-step (frozen global codebook / refit on regions) | gap | rollout change F1 |
|---|---|---|---|---|---|
| 1×2 | 8 | 6 | 0.0232 / 0.0228 | 0.42 / 0.49 | 0.03 |
| 2×2 | 8 | 12 | 0.0224 / 0.0225 | 0.55 / 0.54 | 0.03–0.04 |
| 3×3 | 8 | 27 | 0.0217 / 0.0219 | 0.66 / 0.64 | 0.05–0.06 |
| 3×3 | 16 | 36 | 0.0210 / 0.0203 | 0.79 / 0.90 | 0.05–0.07 |

On val, 3×3 K = 16 gives a gap of 0.68 / 0.72. Regional selection explains far more of the one-step law than one global
code, so different parts of a frame do follow different local laws at the same time. It still produces no motion in
rollout.

## 4. What the modes are (m0, K = 8; `runs/context_s0/modes.json`, `experiments/s0_interp.py`)
Labels come from rules declared in the script, measured on 15-step rollouts from every pool window start.

| code | pool windows | true speed of those windows (median px/frame) | rule's own rollout | flip patterns (SOP terms) |
|---|---|---|---|---|
| 5 | 92 | 0.13 | **static** (hold), frozen 100 % | 8 (7) |
| 3 | 16 | 0.70 | **translation**, 0.68 px/step, births from the W / NW side (eastward) | 168 (52) |
| 6 | 7 | 0.50 | deformation, 0.77 px/step, births from the E / NE side (westward) | 133 (51) |
| 7 | 14 | 0.38 | deformation, 0.32 px/step; 86 % frozen by step 64 | 100 (47) |
| 0, 1, 2 | 15 / 16 / 14 | 0.92 / 0.31 / 0.26 | near-static, ≥ 99 % frozen by step 64 | 71–93 (39–47) |
| 4 | 1 | 1.11 | deformation, growth 1.20× | 39 (25) |

- **The largest mode is "do nothing".** Code 5 explains 92 of 175 pool windows.
- **The only clearly lawful motion is translation.** Codes 3 and 6 are east- and west-moving birth rules. Code 3
  moves at the speed of its windows (0.68 vs 0.70).
- **Rules fitted to slower windows freeze.** True speeds are 0.26–0.38 px/frame, but the rules settle into period 1.
  Code 0 (0.92) freezes too, so this is not simply a sub-pixel effect.
- **Modes are not small edits of each other.** Pairs differ on 47–199 of the 512 patterns seen in the pool. Every
  moving mode needs 39–52 SOP terms.
- **A hand-written global shift matches the learned codebook.** The shift codebook (49 codes, 5.6 bits, no table) is
  equal to the learned K = 8 codebook on cross-scene test (0.0250 vs 0.0249 one-step). What the shared codebook
  captures is essentially global translation.

## 5. Flip vs direct, and information cost
- **Predictions are identical by construction.** `tests/test_s0_codebook.py` checks this on every key, for m0 and m1
  sizes. Exact tables gain nothing from the flip form.
- **Description length differs** (ideal enumerative code of the 1-set):

  | | flip form | direct form | change |
  |---|---|---|---|
  | K = 8 m0 codebook | 2,536 bits | 4,123 bits | −38 % |
  | per-window m0 table (mean) | 154 bits | 514 bits | |
  | K = 8 m1 codebook | 89,920 bits | 2,097,220 bits | |

  The flip form is the more compact encoding of sparse motion; that is an inductive-bias argument for later
  differentiable training.
- **Cost on test cross-scene (18 moving windows):**

  | method | cost | one-step error |
  |---|---|---|
  | codebook (K = 8) | 2,536 + 18 × 3 bits | 0.0249 |
  | per-window tables (in-sample oracle) | 18 × 154 bits | 0.0205 |
  | shift | 18 × 5.6 bits | 0.0250 |
  | hold | 0 bits | 0.0257 |

  Initial state (2 × 1,728 bits) is common to all methods. Timing costs 0 bits on the fixed 16-frame grid. The
  codebook is not a meaningful compression of what the shift already gives.

## 6. Stage 1 (conditional): only the CPU executor sanity was run
- **Why not the learned stage:** the shared-mode evidence is negative, and the one positive signal (translation) is
  reproduced by a hand-written shift. A learned synthetic experiment would not resolve the open question.
- **What was run:** `tests/test_s0_context_exec.py`, a **hand-written** 17-gate circuit (not learned).
  - Context is 2 channels that the circuit copies through.
  - Modes: hold, move east, grow, shrink.
  - Equal to the reference rule on all 2^11 relevant inputs, and on 20 unseen blob starts × 4 modes over 24 steps.
  - A switch east → grow → shrink carries the visible state, and Python == Node (SHA-256).
- **Format finding:** `badca/logic-ca-v1` can hold an *initial* context as state, but has no event mechanism. Its
  prefix only overwrites channel 0, so scheduled switching was done by re-exporting the carried state. Scheduled
  context needs a format extension.

## 7. Interpretation
1. **Small global codebooks do not explain held-out scenes** beyond what one global translation code does. The local
   laws of different windows are not few reusable modes at radius 1 with ≤ 1 frame of memory.
2. **The ceiling was low to begin with.** Even the in-sample per-window table beats hold by only 20 % one-step (m0)
   and reproduces almost no motion in rollout (change F1 0.05–0.11). "Each window has its own local law" (§10a) is
   true but weak. Most of the visible change at 48×36 is not a function of a 3×3 patch, even inside one window.
3. **Context bits address the wrong bottleneck for tables.** Choosing the right table does not create motion; the
   table form freezes. Whatever generates Bad Apple's motion locally needs state (direction, phase or velocity) that
   persists across steps, and possibly spatially varying control.
4. **What is not ruled out:**
   - circuits with persistent hidden channels plus context (a table has no hidden state);
   - velocity-carrying state;
   - larger reach.

## 8. Reproduce

```sh
.venv/bin/python experiments/s0_codebook.py run        # ~50 s wall, ~220 CPU-s (manifest already frozen)
.venv/bin/python experiments/s0_report.py              # val only (selection)
.venv/bin/python experiments/s0_report.py --test       # test (read once in this record)
.venv/bin/python experiments/s0_interp.py              # modes
.venv/bin/python tests/test_s0_codebook.py; .venv/bin/python tests/test_s0_context_exec.py
```

- **Repeatability:** a second full run reproduced `results.json` and every restart loss exactly.
- **Cost:** 0.14 core-h metered (`runs/context_s0/cpu_ledger.jsonl`, including two failed launches: a pickling error
  and a parse error) plus about 0.05 core-h unmetered for tests and reports (estimate). GPU 0.
