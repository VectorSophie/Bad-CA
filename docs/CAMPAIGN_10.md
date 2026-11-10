# Campaign 10: cleaner autonomous replay at the same model size (plan, diagnosis, frozen settings)

Written 2026-11-10. Sections 1–4 were written **before the pilot results were read**. Section 5 was
appended after the pilot and **before the main runs**. Later sections only append. Results go to
`docs/CHECKPOINT_10.md`.

**Goal:** improve the outline, local shape and timing of the Campaign 9 C-trained Neural CA (10,320
params) at the same size.
- Model reduction and longer replays are out of scope.
- **Budget:** GPU 2 h, CPU 8 core-h, everything included.
  - CPU ledger: `runs/c10/cpu_ledger.jsonl`.
  - GPU: gpuq job times.

## 1. What was actually trained and replayed in Campaign 9 (checked in code and manifests)
- **Regime C** (`experiments/exp33_nca.py`): target = clip frames `0..t0+15`, replay from clip frame 0 with hidden 0.
  - Main task: `ba48_4500_60`, src **4500–4555**, 56 frames.
  - Confirm task: `ba48_1800_60`, src **1800–1859**, 60 frames.
  - The export's `n_frames` (56 / 60) is the whole training span.
- **What Campaign 9 videos showed:** only the window (4540–4555 / 1844–1859), cut from this rollout.
- **Check:** `experiments/exp36_full10.py` replays each export from its first frame alone (no clip access),
  across 6 models (2 tasks × 3 seeds). All of these hold:
  - the export's first frame equals the original;
  - the full binary rollout equals C9's stored `C_full`;
  - the window equals C9's stage-C replay;
  - max |Δ continuous| vs C9 = 0.0.
- **Pre-training videos:** `runs/c10/media/pre10_*`:
  - `FULL_*`: whole span, original / C9 binary / continuous / diff;
  - `WINDOW_*`: the window, cut from the full rollout;
  - `ZOOM_FULL_*`: rows 4–31 × cols 0–27 (main arm/shoulder), rows 0–27 × cols 20–47 (confirm staff + hand), ×12;
  - `FULL_*_s0_s1_s2`: all seeds.

## 2. Diagnosis (arrays only; `runs/c10/pre_diag.json`; numbers are min–max over the 3 seeds)
**What the continuous output is:**
- It is the raw visible channel of the residual state. It was regressed toward 0/1 with L2, and the output is `v > 0.5`.
- It is **neither a probability nor a logit**. Values leave [0, 1]:
  - 15–28 % of pixels are < 0;
  - 5–7 % are > 1;
  - range −0.33 to 1.62.
- Margins are therefore read as distance |v − 0.5| in output units.

| question | main | confirm |
|---|---|---|
| full-span pixel error | 0.0086–0.0197 | 0.0255–0.0286 |
| share of wrong pixels inside the 2-px band around the TRUE contour | **91–97 %** | **98 %** |
| interior / background error (fraction of all pixels) | ≤ 0.0008 / ≤ 0.0010 | ≤ 0.0004 / ≤ 0.0001 |
| wrong pixels with \|v−0.5\| < 0.1 / 0.1–0.25 / ≥ 0.25 | 37–38 / 32–35 / 27–31 % | 32–34 / 27–28 / 33–39 % |
| global threshold sweep 0.3–0.7 (diagnostic only) | 0.5 is best or within 1e-4 | 0.5 best |
| error by thirds (early / mid / late) | 0.005–0.011 / 0.004–0.010 / **0.017–0.039** | 0.019–0.024 / 0.028–0.030 / 0.030–0.032 |
| window error pixel-frames whose error run began before the window | 0.6–6 % | 3–4 % |
| window error if the visible is reset to the true frame at the window start, hidden kept (diagnostic) | same or higher (0.019–0.055 vs 0.019–0.043) | higher (0.042–0.047 vs 0.031–0.034) |
| error run length (frames): mean / max / share of error pixel-frames in runs ≥ 4 | 1.3–1.9 / 8–14 / 8–32 % | 1.5 / 10–13 / 14–15 % |
| spurious one-frame blips (pred vs original's own blips) | 2–15 vs 231 | 27–48 vs 665 |
| change mass vs original, lag 1 / lag 4 | **0.46–0.73 / 0.65–0.91** | **0.47–0.51 / 0.69–0.72** |
| post-hoc time shift: lowest err(P_t, F_{t+d}) | d = 0 for every seed | d = 0 for every seed |
| small (≤ 3 px) holes, pred vs original (per span) | **2–6 vs 19** | 33–39 vs 30 |
| small specks, pred vs original | 0–5 vs 0 | 47–88 vs 104 |

**Observed facts:**
- **Contour errors:** the "noise" is almost entirely one- or two-pixel outline errors.
  - Bites into the arm are FN.
  - Bumps and gap-filling are FP; at 4552 the true gap between arm and body is narrowed.
  - Interior and background errors are negligible.
- **Errors are new during the action:** they arise there, not accumulated from the start.
  - Under 6 % of window errors started before the window.
  - Resetting the visible state to the truth at the window start does not reduce them.
  - The error rate rises 2–4× in the last third (main), where the arm moves.
- **Under-motion:** the replay changes 30–54 % fewer pixels per frame than the original (lag 1).
  - At lag 4 the shortfall is 9–35 %.
  - The best time alignment is still d = 0. The impression of "delay" is under-movement and contour
    error, not a whole-frame time offset.
- **Margins:** about a third of wrong pixels are near the threshold and a third are confidently wrong.
  - No global threshold helps, so this is not a thresholding problem.
- **Lost small holes (main):** the model fills small true holes (2–6 small holes vs 19 in the original).
- **Rare flicker:** spurious one-frame blips are far fewer than the original's own blips.

**Hypotheses (inference, to be tested):**
- **H1, more training (arm A):** the plain loss is under-trained. The C9 seed-0 best checkpoint was the last evaluation.
- **H2, contour term (arm B):** a loss that targets the contour band corrects the 1–2 px outline errors.
  - The mean L2 spends most of its weight on easy interior/background pixels; the band is 14 % / 28 % of pixels.
- **H3, change term (arm C):** a term that matches the original's frame changes reduces under-motion
  without adding flicker.

## 3. Arms (model and replay fixed: C=16, Fh=64, S=4, zero pad, hidden 0, threshold 0.5, same preprocessing)
**Code:** `experiments/exp37_ft10.py`, with all formulas in its docstring.
- Start from the C9 `best_seed{0,1,2}` checkpoints.
- Fresh Adam moments for every arm, because C9 saved no optimizer state.
- Adam lr 2e-3 with per-tensor gradient normalisation.
- No randomness.
- All models of one task are vmapped in **one** gpuq job, so the step count and wall time are identical across arms.

**Loss terms** (v = continuous visible, frames 1..T−1; y = original):
- **A:** `L0 = mean (v − y)²` (the C9 loss).
- **B:** `L0 + λB · LB`, with `LB = Σ band·relu(0.25 − s(v − 0.5))² / Σ band`, where `s = 2y − 1`.
  - `band` = pixels within 1 px of the original's contour, on both sides.
  - This is a margin hinge, only on contour-band pixels. It pushes them to the correct side.
  - It neither smooths nor removes holes: true thin parts and gaps are inside the band and are targets.
- **C:** `L0 + λC · LC`, with `LC = ½ Σ_{k∈{1,4}} Σ (D_k clip(v,0,1) − D_k Y)² / Σ dil(D_k Y ≠ 0)`.
  - Y includes the true frame 0.
  - The target is the original's change, not zero.
  - Stalling or delaying a change is penalised.
  - Normalised by the number of (dilated) true-change pixels.
- **Scale at the C9 checkpoints:**

  | task | LB/L0 | LC/L0 |
  |---|---|---|
  | main | 1.8–2.1 | 11–14 |
  | confirm | 1.0–1.1 | 7 |

**Checkpoint selection (same rule as C9):**
- Every 30 s, by **binary pixel error** over the whole regime.
- The unchanged C9 checkpoint is scored first, so "best" may remain C9.
- `last_*` is also saved.

## 4. Pilot, weight choice, and evaluation rules (fixed before the pilot was read)
**Pilot** (main only, one gpuq job, 300 s):
- Models: A, B λB ∈ {0.5, 2}, C λC ∈ {0.05, 0.2}.
- Seeds 0–2 each, so 15 models.

**Weight choice rule:** per arm, using pilot `best_*` checkpoints, full-span metrics (`ca/diag10.py`), medians over 3 seeds.
- **B:** among λB whose median full-span lag-1 and lag-4 change F1 are ≥ pilot-A − 0.03, take the one with the lowest
  median full-span **contour-band error**.
- **C:** among λC whose median full-span pixel error is ≤ 1.05 × pilot-A, take the one with the highest median
  full-span **lag-1 change F1**.
- If no candidate is allowed, use the smaller λ.
- The same λ is then used on confirm, with no confirm-specific tuning.

**Main runs:** one job per task, with A, B(λB*) and C(λC*) × seeds 0–2 (9 models).
- 900 s per job, the same as C9's training time.
- A 3-seed comparison fits the budget, so the scope is not reduced.
- **Optional B+C job:** runs only if B or C qualifies as an improvement on main and at least 35 min of GPU budget remain.
  - 3 seeds per task, the same 900 s, λ = the chosen λB*, λC*.

**Baselines:** the unchanged C9 checkpoint of each seed.

**Metrics** (`ca/diag10.py`, binary replay from the **export**, original time alignment):
- full-span and early/mid/late pixel error;
- error in the contour band / interior / background;
- FP / FN;
- contour F1;
- lag-1 and lag-4 change F1 and mass ratio;
- spurious blips;
- error run lengths;
- near-threshold distribution;
- small holes/specks (auxiliary only);
- the window metrics and the C9 stage-C pass rule (err/hold ≤ 0.70, lag-4 F1 ≥ 0.50 main / 0.65 confirm,
  lag-4 precision ≥ 0.40);
- the post-hoc shift curve (diagnostic only).

## Improvement criteria (fixed before the pilot)
An arm counts as an **improved candidate on a task** if all of the following hold, comparing each seed
with its own C9 checkpoint:
1. **Shape:** median full-span contour-band error reduced by ≥ 10 % relative, and lower than C9 in ≥ 2 of 3 seeds.
2. **Motion:** median full-span lag-1 and lag-4 change F1 each ≥ C9 − 0.03.
   - The number of seeds passing the C9 stage-C rule is not lower.
3. **No local trade:** for no third (early / mid / late) does the median full-span pixel error rise by > 10 % relative.
4. **No new flicker:** median spurious blips ≤ max(1.5 × C9, C9 + 10).

**Representatives:**
- **Improved candidate:** among arms that qualify, the one with the lowest median full-span contour-band error.
- **Seeds:** the median seed by full-span pixel error is the representative; best and worst are also shown.
- **Trade-offs:** reported per metric, never folded into one score.

## 5a. Pilot 1 (lr 2e-3, as Campaign 9): failed setting, appended after reading it
- **Run:** gpuq job 72, 5.1 min, 3,175 steps, 15 models.
- **Result:** restarting Adam (fresh moments) at lr 2e-3 knocked **every** model off its C9 optimum. Binary
  error after 30 s was 1.5–3.5× the C9 value.
  - 300 s was not enough to recover. Only 2 of 15 models went below their start: B0.5_s1 0.0178 and
    C0.05_s1 0.0179, against 0.0197.
  - B0.5_s2 collapsed to a constant state (err 0.243).
  - So the selected "best" was mostly the unchanged C9 checkpoint (it = 0).
- **Why the start is hard to beat:** C9's checkpoint is the lowest of a strongly fluctuating lr-2e-3 trajectory,
  which ranges 0.013–0.078 within the last minutes of a C9 run. Beating it at the same lr is mostly a lottery.
  - This is an **optimization** effect of the restart. It is not evidence about the loss terms.
- **Change for all arms equally:** continuation lr **2e-4**. Everything else is unchanged.
  - The pilot is repeated once (job 73, the same 15 models and 300 s).
  - The weight rule of §4 is applied to pilot 2.
  - If pilot 2 also fails to move below C9, the main runs still go ahead at 2e-4 and the result is reported as is.

## 5. Pilot 2 and the frozen main settings (appended before the main runs)
- **Run:** pilot 2, gpuq job 73: lr 2e-4, 300 s, 3,195 steps, 15 models.
  - Every model moved below its C9 start smoothly.
  - Medians: `runs/c10/eval/c10_pilot2_main_best.json`.
- **§4 rule:**
  - **B:** both λB pass the motion check; the lowest median band error is **λB = 2** (0.0056 vs 0.0062 for 0.5, 0.0072 for A).
  - **C:** both λC are within 1.05 × A pixel error; the highest median lag-1 F1 is **λC = 0.05** (0.846 vs 0.844).
  - Pilot medians already show A < C9, i.e. extra training alone helps. This is recorded as a pilot observation only.
- **Frozen main settings** (`runs/c10/cfg/main.json`):

  | setting | value |
  |---|---|
  | per task | A (λ = 0), B (λB = 2), C (λC = 0.05) × seeds 0–2 = 9 models in one job |
  | lr | 2e-4 |
  | budget | 900 s per job |
  | start | the C9 best checkpoints |
  | Adam | fresh moments |
  | checkpoint selection | binary-error selection every 30 s |

- **Code:** frozen in `runs/frozen/<hash>` (printed in the job command).
- **Confirm:** the identical config.
- **Evaluation:** `best_*` is the primary result. `last_*` is also evaluated as a diagnostic.

## 6. After the main runs (appended; does not change §4–5)
**Main jobs (74, 75):**
- A, B and C qualify on both tasks; B has the lowest band error.

**B+C follow-up (jobs 76, 77):** triggered by the §4 rule (B qualifies, more than 35 min of GPU left).
- Same 900 s, but only 3 models per job, so it ran **~2× the optimizer steps** (21,192 / 20,182 vs 10,688 / 10,297).
- The plan matched wall time, not steps. This is a flaw in the follow-up design, recorded here.
- Matched-step check from the training logs (binary error, best at ≤ the main-job step count): B+C ≈ B on all
  6 seed–task pairs, within ±0.0003.
- By the §4 selection rule, B+C is still the representative candidate.
- Its advantage over B is attributed to the extra steps, **not** to the change term.
