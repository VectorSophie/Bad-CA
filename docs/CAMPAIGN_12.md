# Campaign 12: phases, explicit rules, rule-inspection UI, documentation (plan; fixed before the pilot results were read)

Written 2026-11-10 while the pilot jobs were running, before any pilot result was read. Later decisions are appended
at the end and never rewrite this part. Results go to `docs/CHECKPOINT_12.md`; machine-readable results to
`runs/c12/results12.json`.

## 0. Starting point (C9–C11; see their CHECKPOINTs)
- **Short intervals:** a 10,320-param continuous NCA (C10) replays them well. 3,884 and 1,304 failed the C11 gates.
  Distillation was worse than direct training.
- **L120 (1760–1879, no cut):** the same size passes the sustained-motion gate (error ≈ 0.031).
- **L240 (1760–1999):** fails. The interval adds a wipe (1880) and a shot change (~1911), so length and transitions
  are confounded.
- There is no `AGENTS.md`. The project instructions are `claude_ca_*.md`.

## 1. Replay semantics (fixed)
Every main result uses **type 1** (fixed rule) or **type 2** (time-varying CA with an external schedule):
- Frame t is produced from frame t−1 by S = 4 synchronous substeps of the rule of the phase that owns t.
- Visible and hidden state are **carried across phase boundaries**.
- Nothing from the target is read after the first frame.
- There are no resets, corrections, coordinates or frame-number inputs.

The schedule and every rule/delta are stored in the export (`ncab-v2`, `rule-v1`).

**Other types:**
- **Type 3** (internal state decides the phase): a possible pilot only.
- **Type 4** (reset per clip): diagnostic only. It is never counted as autonomous success.

**Stored initial information:**
- The first frame.
- Optionally a learned low-resolution initial hidden array `init_g`: (C−1) × ⌈H/f⌉ × ⌈W/f⌉, nearest-upsampled by
  factor f.
- It is counted in the file size and reported separately.
- A uniform (black) first frame with zero hidden cannot break symmetry under a shift-invariant rule with a zero
  boundary: all interior cells stay identical. This is why the opening needs stored initial information. It will be
  reported as such, not as a rule achievement.

## 2. Short survey → design choices
Sources are cited by name and year; no external code is reused.

**Schedules and segmentation**
- **Time-varying CA / non-uniform CA** (Sipper, *Evolution of Parallel Cellular Machines*, 1997) and **switching
  dynamical systems** (Fox et al. 2009; Linderman et al. 2017 rSLDS).
  - Piecewise dynamics with discrete switches. Here the switch times are an external schedule found from the video,
    not inferred online. This is type 2.
  - rSLDS-style state-dependent switching would be type 3 and is not pursued (§1).
- **Change-point detection / MDL segmentation:** PELT (Killick et al. 2012), Bayesian online CPD (Adams & MacKay
  2007), MDL (Rissanen 1978).
  - → `ca/segment.py`, stage 1: an exact DP for K segments minimising the code length of
    "next centre bit | 3×3 pattern" under one table per segment (+ BIC table cost), plus cut/uniform-frame and
    sustained-change (wipe) detection for reasons.
  - Stage 2 (trained residuals, limited boundary moves) decides whether the boundaries help a CA.
- Video-analysis boundaries are not assumed to be rule boundaries (stage 2).

**Neural CA and adaptation**
- **NCA:** growing / texture / dynamic-texture NCAs (Mordvintsev et al. 2020; Niklasson et al. 2021; Pajouheshgar et
  al. 2023 DyNCA) for the update and training practice already in use since C9.
- **Low-rank adaptation:** LoRA (Hu et al. 2021).
  - → "shared base + per-phase rank-r deltas" (`mode=delta`). Encoding = all factor and bias-delta floats; the
    count, magnitude and bytes are reported separately.

**Explicit rules**
- **Rule extraction and symbolic models:** decision-tree extraction from networks (Craven & Shavlik 1996 TREPAN);
  symbolic regression and program synthesis (Cranmer 2023 PySR; Ellis et al. 2021 DreamCoder).
  - → explicit path B = **decision lists over integer neighbourhood features with k ≤ 3 hidden bits** (`ca/rules.py`).
    They are learned by a small CART (own code; scikit-learn is not in the env and is not installed).
  - Either learned directly from frames, or extracted from a trained NCA via quantised NCA states.
  - Always evaluated by **executing** the rules autonomously.
  - Symbolic regression and program synthesis are not used: no extra dependency, and the pilot first tests whether
    a readable decision list can replay anything at all.
- **Earlier explicit attempts** in this project: SAT/CGP/DiffLogic Boolean circuits, C1–C8. Exact on very short clips;
  they failed on real motion.

**Spatial information at a cut**
- Under a shift-invariant rule, cells with the same local state update identically. A rule change alone cannot
  "place" a new shot; the information must already be in the carried state (hidden), or come from stored initial
  information.
- Joint training lets earlier phases prepare hidden state for later ones. Greedy training cannot.
- We measure which helps.

## 3. Pilot (≤ 4 GPU-h, ≤ 8 CPU core-h)
**Engine and format controls (done before the pilot):**
- `ca/segment.py`: an inversion cut is found exactly.
- `ca/phase_nca.py`: v2 round trip for fixed/indep/delta with learned init. A v2 file holding the C10 rule replays
  identical to v1, Node sha `f2401adb…`.
- Node v2 random models: binary output identical; relative continuous difference ~1e-4.
- `ca/rules.py`: Conway's Life recovered **exactly** as 3 rules from teacher-forced pairs; a known hidden-bit rule is
  recovered exactly.
- `player/rule_core.mjs` matches Python **bit-exactly** on random rule programs with hidden bits and phases.

**GPU pilot on L240 = 1760–1999:**
- Common settings: 2 seeds (greedy: seed 0), 12,000 steps, cosine lr 5e-4 → 5e-6, L0 + 2·LB, hidden init 0.
- Joint arms use full BPTT from frame 0.

| arm | rule params / seed |
|---|---|
| `fixed64` (C11 size) | 10,320 |
| `fixed256` (capacity-matched) | 41,232 |
| `uniform4` (K = 4 equal phases, independent rules) | 41,280 |
| `auto4` (stage-1 MDL phases: 1761 / 1843 / 1878 / 1917, independent) | 41,280 |
| `autodelta4` (same phases, shared base + rank-4 deltas) | 10,320 + 4 × 1,232 |
| `greedy4` (auto phases, sequential, warm start per phase, 12,000 steps per phase) | 41,280 |

- greedy4 is matched in frame-steps to joint training, not in optimizer steps.

**CPU pilot:** explicit rules on the short main task (4500–4555) and on L120:
- (a) learned directly from frames (k = 0, S = 1);
- (b) extracted from a trained NCA (quantised visible + k hidden bits from NCA channels, NCA substeps, S = 4).
Both are evaluated by autonomous execution.

## 4. Gates and comparison rules (fixed now)
- **Error:** full-interval pixel error of the autonomous replay from the export, against the preprocessed binary
  original. All seeds are reported.
- **Representative seed:** seed 0; with 2 seeds both are always shown.
- **Interval success** (any length), all of:
  1. **Sustained motion** (C11 gate): in each third, lag-4 change F1 ≥ 0.5 and change mass 0.5–1.5; and late-third
     error ≤ 3 × early-third error or ≤ 0.03.
  2. **Full error** ≤ 0.035 (the L120 large-model level: 0.0304 / 0.0323).
  3. **Worst 16-frame window error** ≤ 0.09.
  4. **Transitions:** mean error over frames within ±3 of every detected cut or wipe onset ≤ 0.10, and no
     uniform-frame (fade) is missed. A missed fade means the replay is not blank when the original is blank.
  5. **Visual veto** only.
- **"Phases help":** an arm counts as better than another when its full error is lower in **both** seeds and it is
  not worse by more than 10 % on the worst window. With 2 seeds this is a statement about this interval and
  settings only.
- **Capacity control:** phase benefit is claimed only if `auto4` (or `uniform4`) beats `fixed256` (same parameter
  count) by the rule above.
- **Auto vs uniform:** compared at the same K, same steps and same parameter count by the rule above.
- **Explicit rules:**
  - Reported with complexity (rules, conditions, constants, hidden bits, bytes) and the C11 metrics.
  - The one-step teacher-forced match and the autonomous replay are reported separately.
  - "Usable" is not claimed unless the full-span error ≤ C9-level for that interval (main 0.0129).
- **Cost:**
  - GPU = assigned GPUs × job runtime, summed per job (the user's definition; a shared GPU is not discounted).
    Queue waits are recorded separately.
  - Ledgers: `runs/c12/gpu_ledger.jsonl` and `runs/c12/cpu_ledger.jsonl`.

## 5. After the pilot (filled in later, appended)
Allocation of the remaining budget (24 GPU-h, 48 CPU core-h total) and stop rules are fixed after reading the pilot.

## Decisions appended during the pilot
- **Pilot budget:** joint arms on L240 run at ~5.4 steps/s for 2 seeds, ~37 min each. Projected pilot total
  ≈ 4.3 GPU-h with `greedy4` at 12,000 steps per phase, i.e. over the 4 GPU-h pilot cap.
  - An attempt to cancel the queued `greedy4` job and resubmit it with fewer steps was **refused by the session's
    permission policy**.
  - The job runs as originally submitted, and the overrun is reported as such. It counts toward the 24 GPU-h total;
    no other pilot job was added.

## 5. Pilot results and the fixed allocation (appended after the pilot, before the main runs)

**Pilot, L240 = 1760–1999** (`runs/c12/eval/pilot.json`; full-interval error, seeds s0/s1):

| arm | full error | gates passed |
|---|---|---|
| fixed64 | 0.072 / 0.065 | none |
| auto4 (joint) | 0.028 / 0.033 | s0 all; s1 all but sustained (early-third change mass 0.48) |
| uniform4 (joint) | 0.027 / 0.028 | both seeds, all |
| autodelta4 (rank 4) | 0.049 / 0.054 | – |
| greedy4, warm start | – | phase 4 exploded (the previous rule run past its trained horizon) |
| greedy4, cold start (hold init per phase; main budget) | 0.062 | – |
| fixed256 | 0.283 / 0.283 | **invalid control:** non-finite from the first steps (stuck guard); no evidence about capacity |

- **auto4 vs uniform4:** auto is **not** better (rule: lower in both seeds).
- **greedy (cold) vs joint:** phases 1–2 are similar. The wipe and shot-change phases are much worse in greedy
  (0.099 / 0.085), consistent with joint BPTT letting earlier phases prepare hidden state for a new shot.
- **Explicit rules** (CPU, main 4500–4555; `runs/c12/eval/rules_pilot.json`):
  - Executed autonomously, they collapse to hold (unweighted) or to flicker (change-weighted).
  - Short phases help: K = 14 programs, 117 rules, error 0.0995, vs hold 0.183 and C10 0.004.
  - Extraction from the C10 NCA (k = 1–3 quantised hidden bits) gives at best 0.153.
  - Both positive controls, Life and the hidden-bit rule, were recovered exactly. So the engine and learner work;
    the limit is the representation and learning on real motion.
- **Pilot GPU:** jobs 10–15 = 232 min = 3.87 GPU-h; with the UI agent's check job (0.2 min) and the cold-greedy
  test (job 17, 10.8 min), ≈ 4.05 GPU-h.

**Main allocation** (≈ 19.9 GPU-h left, two GPUs). Each line is a gpuq job unless noted:

| GPU | job | steps / settings | cost |
|---|---|---|---|
| A | **Whole song 0–6571, chunked-joint**, 1 seed | 8,000 steps per chunk, lr 5e-4 cosine | est. ~9 GPU-h |
| B | fixed256 repair | lr 1.25e-4, 2 seeds, 12k steps | ~1 GPU-h |
| B | L120 (cut-free) uniform2 / auto2 vs the C11 fixed result | 2 seeds | ~0.7 GPU-h |
| B | stage-2 segmentation check on L240: uniform6 vs auto6 (more, shorter phases) | 2 seeds each | ~1.3 GPU-h |
| – | evaluation, videos, UI and docs | CPU | – |

**Song run details:**
- Phases: `runs/c12/song_phases_auto110.json` (stage-1 MDL, K = 110, min length 16, max 120, grid 4, cut penalty 1e5).
- Chunks: groups of consecutive phases totalling ~200–280 frames, trained jointly inside each chunk.
- State is carried from chunk to chunk; no reset.
- Chunk 0 starts from the black first frame with a **learned low-res initial hidden** (factor 4: 15 × 9 × 12 = 1,620
  floats, counted in the file).
- Evaluation is a full rollout from frame 0 from the export.
- Single seed: exploratory.

**Stop rules:**
- The song run stops by itself if the carried state becomes non-finite.
- I stop extending if 3 consecutive chunks end with training error > 0.20. The longest prefix that passes the
  gates is then reported.
- No budget extension. If the projected total exceeds 24 GPU-h, B-side jobs are dropped first.
- **Loophole check (`runs/c12/eval/baselines.json`):** the pre-registered C12 gates were applied to known loophole
  baselines built from the TRUE original.

  | baseline | L240 full error | gates |
  |---|---|---|
  | hold | 0.401 | fails |
  | 1-px shrink | 0.109 | fails |
  | edge flicker | 0.055 | fails |
  | **3×3 majority blur** | **0.023** | **passes every gate** |

  - The majority blur erases thin parts: thin-part recall 0.59, gap recall 0.62.
  - So the gates as fixed **do not reject blur-type smoothing**. They are kept unchanged (no post-hoc edits), but
    every result is reported with thin-part and gap recall, specks and holes next to the gates.
  - No success is claimed from the gates alone.
- **Correction to §1 (black opening):** the claim that a uniform first frame with hidden 0 "cannot break symmetry"
  under a zero boundary is **wrong**.
  - After the first update the bias makes the interior state non-zero, while the zero padding stays 0, so the
    border differs from the interior and the difference spreads inward.
  - Measured on the song's chunk-0 rules (`runs/c12/eval/opening_symmetry.json`): with hidden 0 the replay is not
    uniform (1,728 distinct visible values in the last frame).
  - With the learned initial hidden (1,620 values) chunk 0 reaches error 0.0079. The same rules with hidden 0 give
    0.451, but they were trained with `init_g`, so this does not test whether training without stored initial
    information could work. That was not run (budget).
  - The learned initial hidden remains part of the song run and is counted in its file.
- **Main results so far** (`runs/c12/eval/main.json`; both seeds):

  | run | full error | gates |
  |---|---|---|
  | fixed256, lr 1.25e-4 (valid capacity control) | 0.047 / 0.045 | fail |
  | 4 phases at the same parameter count | 0.027–0.033 | – |
  | L120 (cut-free), uniform2 | 0.031 / 0.030 | – |
  | L120 (cut-free), auto2 | 0.029 / 0.029 | – |
  | C11 fixed rule on L120, for reference (C11 trainer, 30k steps) | 0.030 / 0.032 | – |
  | L240, uniform6 | 0.019 / 0.021 | all pass |
  | L240, auto6 | 0.021 / 0.022 | all pass |

  - **Capacity control:** 4 phases beat the wide fixed rule in both seeds, with better worst windows (0.07 vs 0.12) and
    thin parts (0.71–0.75 vs 0.61–0.64).
  - **Cut-free L120:** phases bring **no clear gain**. auto2 is lower than uniform2 in both seeds, but its s1 worst
    window is 16 % worse, so it is not better by the rule.
  - **L240:** more phases help. **Auto is not better than uniform at K = 6 either.**
- **Added song arm (budget allows; decided before any song result):** the same chunked-joint song run with
  **uniform** phases (`runs/c12/song_phases_uniform110.json`, K = 110), the same chunking, steps, seed and init.
  - This compares automatic vs uniform segmentation at song scale.
  - Estimate ~7 GPU-h, for a projected total ≈ 21 of 24 GPU-h.

## 6. Song result and the robustness repair (appended after the song run)
- **Training numbers are not replays:** the automatic song run trained all 28 chunks; per-chunk training errors were
  0.004–0.075, and 0.161 for one chunk. But the **full autonomous replay from the export** (numpy, and JAX on CPU,
  which agree) matches training only through chunk 3 (frames ≤ 971).
  - From chunk 4 on it diverges: 0.064 vs 0.026, then 0.12–0.51.
  - Full-song error is 0.339; 113 of 191 fades are missed; hidden values reach ~5e18.
- **Cause (measured):** each chunk was trained from a carried state computed on the GPU. The CPU replay recomputes
  that state with tiny float-order differences, and the chained rules amplify them. The rules fit one exact
  trajectory; they are not robust to perturbation.
- **Uniform song arm (job 24):** it could not be cancelled (permission policy), so it runs to completion and is
  reported, but it is expected to show the same failure.
- **The one bounded repair:** chunked training with **training-only Gaussian state noise σ = 0.02 per frame**.
  - Same automatic phases, steps and seed, on the song prefix 0–1467 (6 chunks), where the divergence began.
  - Success = the CPU replay from the export tracks the training errors past frame 972.
  - If it works, the noise recipe is the recommended next step for the whole song. No second song run is affordable
    in this budget.
- **Repair result** (gpuq 25, 1.56 h): with training-only state noise σ = 0.02, the CPU replay from the export
  **equals the training error in every chunk** (0.007 / 0.007 / 0.010 / 0.006 / 0.018 / 0.015). The no-noise run
  replays at 0.008 / 0.008 / 0.015 / 0.012 / 0.064 / 0.122.
  - Prefix 0–1467: error 0.0104, all C12 gates pass, 0/43 fades missed.
  - Node vs numpy max |Δv| is 1.8e-4 (no-noise song: 4.5e19).
- **Extension within the remaining budget:** the uniform song run should end at ~7.5 GPU-h, which leaves ~1.3 GPU-h.
  - The noise run is resumed for 3 more chunks, to frame 2115 (`song_phases_auto110_prefix2116.json`).
  - The chunk grouping of the first 6 chunks is identical, so they are reused.
  - Projected total ≈ 23.5 GPU-h. Nothing further is started.
