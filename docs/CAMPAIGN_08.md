# Campaign 08: can a small CA circuit follow Bad Apple's motion into the late part of a clip? (pre-run plan)

Written 2026-11-09, **before the λ pilot and every main run**. Already done before this was written, all
on the ledger `runs/c8/cpu_ledger.jsonl`:
- the C7 comparison videos (`runs/c8/media/c7_best_vs_hold_*`);
- the loophole check of fixed baselines (`runs/c8/loopholes.json`);
- the control generation;
- a 40 s smoke test of CGP v2.

Results go to `docs/CHECKPOINT_08.md`. Pilot outcomes and the fixed λ are appended below; this part is
not edited afterwards.

## Fixed setting
- **Main clip:** `ba48_4500_60`, frames 4500–4559, 48×36.
  - Preprocessing: box resize to 48 px wide, threshold 128 (`ca/video.py`, PREPROC_VERSION 1).
  - SHA-256 of all frames is recorded in every result.
- **Second clip:** `ba48_1800_60`, frames 1800–1859. The **same procedure is re-applied** to it: λ, arms
  and budgets are fixed on the main clip and not re-tuned. A rule found on one clip is never claimed to
  replay the other.
- **Model:**
  - boundary 0, hidden init all 0;
  - autonomous rollout of all 60 frames from the first frame;
  - no target injection, hidden reset or rule switching.
  - Only **observed** frames (every S generations) are compared. Intermediate generations are never
    called "motion".

## Structures and objectives (2×2)

| arm | structure | objective |
|---|---|---|
| A | base = C7: F_priv k=1 (own hidden), S=2 (`k1_S2`) | pix |
| B | base | mot |
| C | ext: F_shared k=1 (neighbours' hidden readable), S=4 (`k1s_S4`): same hidden bit count | pix |
| D | ext | mot |

**Objectives:**
- **pix** (the C7 objective): mean pixel error over frames 1..59; ties broken by fewer active gates.
- **mot:** J_λ = mean over seg ∈ {early 1–19, mid 20–39, late 40–59} of [err_seg / max(hold_seg, 1e-3) + λ (1 − F1tol_seg)];
  ties broken by fewer active gates.
  - **F1tol (implemented in `ca/motion.py`):**
    - directional changes: on = X_t ∧ ¬X_{t−1}, off = ¬X_t ∧ X_{t−1};
    - precision side: a predicted change is a hit if a true change of the **same direction** lies in its 3×3 window;
    - recall side: a true change is recovered if a predicted same-direction change lies in its window.
    - Micro-summed per segment.
    - **Empty cases:** no predicted change → precision 1; no true change → recall 1; both empty → F1 1.
  - **Segment balance:** gives the late segment the same weight as the early one. The err/hold term
    keeps shape accuracy.
  - **Loophole check (λ=1, done):** every cheat baseline scores worse than hold.

    | baseline | J |
    |---|---|
    | hold | 2.00 |
    | black | 2.37 |
    | shift-right | 2.63 |
    | 5% random flips | 3.38 |
    | flicker | 3.94 |
    | white | 5.59 |

    The 1-frame-lag oracle scores 0.555 and C7's best 1.42.

**Search:** CGP v2 (`experiments/exp27_cgp2.py`).
- The C7 algorithm: (1+4)-ES, single-active-gene mutation, neutral drift; G = 32 nodes, the C7 upper
  budget.
- Direct gate-level grid simulation.
- Only the target frames are given.
- Saved on every improvement: genome, fitness, RNG state.

## Positive controls (run first)
Known circuits, simulated by the same evaluator; the generator itself replays exactly. The circuits are
in `experiments/c8_targets.py`.

| control | how it is made | what it tests |
|---|---|---|
| `pcmove_<struct>` | hidden′ = NOT h (clock), visible′ = h ? W : C | sustained rightward motion (S/2 px per frame) for all 60 frames, on a wide field |
| `pccycle_<struct>` | a 3–6-gate random circuit from fixed screening (rng 2026): period 2–8 in the late segment, 2–30% of pixels changing per frame, not a global inversion | localised periodic change |

- **Neither** can be passed by hold, black/white or flicker.
- **Runs:**
  - each control × {base, ext} × **mot** (λ=1): 3 seeds × 180 s;
  - **pix**: 2 seeds × 180 s.
- **Control pass:** exact replay in ≥ 1 seed, for each control × structure under the mot objective.
- **If a control fails:**
  - record which of optimisation, representation or budget is unknown;
  - check that the oracle circuit replays (it does);
  - run one seed with 3× the budget;
  - the main runs on that structure are reported as "**control not passed**" and are not
    interpreted as Bad Apple impossibility.

## λ pilot (fixed before the main run)
- **Runs:** B (base, mot) on the main clip, λ ∈ {0.5, 1, 2}, seeds 100–101 (not the main seeds), 300 s each.
- **Choice:** the λ with the highest mean over seeds of **min over segments of F1tol**. Ties within
  0.01 go to the lower mean err/hold.
- **Then fixed for every mot run** on both clips.

## Main runs, seeds, budgets, order
1. Controls (above).
2. λ pilot.
3. **2×2 on `ba48_4500_60`:** arms A–D, seeds **0–4**, **600 CPU-s** each, G=32.
4. **2×2 on `ba48_1800_60`:** arms A–D, seeds **0–2**, 600 CPU-s each.
5. **Conditional ablation:** only if C or D beats A or B by the effect rule below. Then `k1s_S2`
   (neighbour hidden only) and `k1_S4` (S=4 only), with the better objective, seeds 0–4, 600 s.
6. **Conditional curriculum follow-up:** only if some arm reaches "partial" or better.
   - The prefix length grows over 20 → 40 → 60 frames (200 s each), within one CGP run (the genome is
     carried over).
   - **Evaluation is always the full 60-frame autonomous rollout from frame 0.**
   - 3 seeds.
7. **Re-evaluation of the C7 best**, kept separate from new search (already in `loopholes.json`).
8. **Optional post-hoc reduction** for candidates that meet the success rule: delete a gate (rewire it
   to one of its inputs) when the 60-frame report still meets the rule. Bounded by 120 s each.

**Budget ceiling:** 12 CPU core-h; **no GPU**. CGP here is numpy grid simulation of tens of gates, and
porting it would cost more than it saves.

**Planned:**

| part | core-h |
|---|---|
| controls | ≤ 1.0 (early stop on exact) |
| λ pilot | 0.5 |
| main | 3.3 |
| second clip | 2.0 |
| ablation | ≤ 1.7 |
| curriculum | ≤ 0.5 |
| videos and verification | 0.3 |
| **total** | **≤ 9.3** |

**Execution:** 1 thread per process, `prlimit --as` 16 GB, a wall cap of 3× the budget. Every arm gets the
same seeds and the same CPU budget. Evaluations and throughput are recorded, since S=4 halves
throughput per evaluation.

## Success criteria (fixed now, from the baselines in `loopholes.json`)
**Segment followed** iff **all** of:
1. F1tol ≥ **0.25**. The best cheat baseline reaches 0.134 in the late segment of the main clip
   (flicker) and 0.204 on the second; 0.25 is about 42% of the 1-frame-lag oracle's late 0.595.
2. Tolerant precision ≥ **0.20**. Flicker gets 0.03–0.12, random flips 0.02–0.13.
3. Predicted/true change-mass ratio in **[0.25, 3]**, so neither frozen nor flickering.
4. Segment pixel error < segment hold error.

**Result levels:**
- **Followed until late (success)** = early, mid and late all followed, in the full 60-frame autonomous
  replay.
- **Partial** = early followed and the late segment not followed.
- **Fail** = otherwise.
- C7's best scores early F1 0.253, mid 0.034, late 0.000, so it is **partial at most** (and fails the
  mid segment).

**Arm effect** (objective or structure), claimed only if:
- the better arm has **more** success+partial seeds, or
- its median late F1tol is ≥ 0.05 higher **and** ≥ 4/5 of its seeds exceed the other arm's median.
- Otherwise: "no detectable effect at this budget".

**Stop rules:**
- a structure's control fails → stop interpreting it (see above);
- the ledger reaches 10.5 core-h → no new jobs.

## Video deliverables and representative selection (fixed now)
- **Videos:** MP4 (libx264, verified by decoding back) + GIF; 1× (30 fps) and 0.25× (7.5 fps).
  - Nearest-neighbour scaling; labels for clip, source frame and time, seed, active gates, k and S.
  - Diff panel: red = false negative (target white, replay black), blue = false positive.
  - Replays come from **decoding the exported file**. Original frames are used only for display
    and error computation.
- **Required videos:**
  - ① original / C7 best / C8 representative;
  - ② original / C8 representative / diff;
  - ③ each control: original / recovered;
  - plus a failure case.
- **Representative selection:**
  - **arm:** the most success+partial seeds, then the highest median late F1tol;
  - **representative:** the seed with the **median** late F1tol in that arm;
  - also shown: the best seed and the **worst** seed (failure) of that arm;
  - the same for the second clip.
- **All seeds' metrics** are in the results table. The success rate is always reported with the best.

---
## Pilot results and fixed λ (appended before the main runs; snapshot df872b43bf6d)
**Controls: all pass.** Every control × structure × objective is exact in ≥ 1 seed.

| control | structure | mot seeds exact | pix seeds exact | fewest active gates in an exact solution |
|---|---|---|---|---|
| pcmove | base | 3/3 | 2/2 | 1 |
| pcmove | ext | 3/3 | 2/2 | 0 (pure wiring delay line) |
| pccycle | base | 2/3 | 2/2 | 2 |
| pccycle | ext | 2/3 | 2/2 | 1 |

- The non-exact mot seeds still met the success rule: F1tol 0.96–0.99.
- The generators were never given to the search.

**λ pilot** (arm B, seeds 100–101, 300 s each):

| λ | min-segment F1tol per seed | mean |
|---|---|---|
| 0.5 | 0.05 / 0.00 | 0.025 |
| 1 | 0.04 / 0.00 | 0.02 |
| 2 | 0.15 / 0.13 | **0.14** |

**→ λ = 2 is fixed** for every mot run on both clips.

## Decisions after the main 2×2 (appended before the follow-up; per the rules above)
- **Objective effect (B vs A, main clip): detected** on late F1tol.
  - Median 0.173 vs 0.000, so the gap is ≥ 0.05.
  - 4/5 of B's seeds are above A's median.
  - The success+partial counts are tied (3 vs 3).
- **Structure effect: not detected.** C/D score below A/B on every measure, so **the ablation is not run.**
- **Curriculum follow-up: triggered,** since partial seeds exist.
  - Arm B config (k1_S2, mot λ=2) with prefixes 20 → 40 → 60 in equal CPU thirds, the genome carried over.
  - Seeds 0–2, 600 CPU-s.
  - Evaluated on the full 60-frame autonomous replay.
  - Implemented as OBJ `motcur` in `exp27_cgp2.py`.
- **Teacher-forced per-segment diagnostics:** not run.
