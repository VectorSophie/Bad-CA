# Campaign 09: is there a representation + search that reproduces a short real Bad Apple action? (pre-run plan)

Written 2026-11-09, **before any model is trained**. Already done before this was written, all on the ledger
`runs/c9/cpu_ledger.jsonl`:
- the task-candidate scan;
- the pre-training videos;
- the window baselines and certificates;
- an NCA evaluator check.

Results go to `docs/CHECKPOINT_09.md`. Later decisions are appended below and do not edit this part.

## 1. Tasks (chosen before training; `runs/c9/task_candidates.json`, contact sheets `runs/c9/media/cand_*`)
- **Candidates:** every 16-frame window (stride 4) of `ba48_4500_60` and `ba48_1800_60` was scored:
  - uniform frames;
  - max per-frame change (a scene cut if > 0.25; none found);
  - mean change;
  - revert rate;
  - 4-frame net change.
- **Visual check:** the four highest net-change windows were inspected by eye.
- **Main task:** `ba48_4500_60` frames **4540–4555** (t0 = 40, 16 frames).
  - An arm rises from the body and bends at the elbow.
  - Net 4-frame change 0.109 (the highest in its clip), revert 0.16.
- **Confirm task:** `ba48_1800_60` frames **1844–1859** (t0 = 44).
  - The staff and arm swing and the body widens.
  - Net 4-frame change 0.133 (the highest in its clip), revert 0.07.
- **Rejected:**
  - 4536–4551: it overlaps the main task.
  - 1800–1815: thin smoke and staff details are hard to read at 48 px.
  - All low-motion windows of 4500, e.g. 4524–4539: net 4-frame change 0.017.
- **Preprocessing:** unchanged; 48 px wide, box resize, threshold 128.
  - Threshold 96 / 128 / 160 gives nearly the same window statistics (main mean change
    0.040 / 0.041 / 0.041, net 4-frame change 0.111 / 0.109 / 0.109), so the motion is not a
    thresholding artefact.
- **Pre-training video:** `runs/c9/media/pre_task_*_c8_*`, the original window vs the C8 representative.
- **Certificates** (`runs/c9/window_baselines.json`): any radius-1 rule, constant hidden init.
  - **Light cone from the window start:** fails at S=1 and S=2 (S=2: 3 conflicting pixels at frame 1
    on main, 1 on confirm), **passes at S=4**.
  - **From the clip start (stage C):** S=4 passes for main and **fails for confirm (1 pixel at
    frame 3)**. On the confirm task, an exact stage-C replay is impossible for every radius-1 model with
    a constant init. Approximate replay is still possible.

## 2. Diagnostic stages (never conflated)
| stage | what is given | protocol |
|---|---|---|
| **A** (teacher-forced) | at every frame the visible state is set to the TRUE frame t−1; hidden is carried from the model's own previous step, starting at 0 from the window's first frame | one-frame prediction of frame t, every t in the window |
| **A-reset** (diagnostic) | as A, hidden := 0 at every frame | memoryless one-step |
| **B** (short autonomous) | the true window frame 0 once, **hidden = 0** | autonomous for 15 frames |
| **B-warm** (separate diagnostic) | hidden from a teacher-forced warm-up over the true clip frames 0..t0−1 | then autonomous |
| **C** (original goal) | the clip's first frame (4500 / 1800) and hidden 0, once | autonomous through the window's end; metrics on the window frames |

Only C counts as reproducing the action. A and B are diagnostics.

## 3. Models
- **① CGP circuits:** binary.
  - Base structure = C7/C8 `k1_S2` (F_priv k=1, S=2).
  - Search: CGP v2 (`exp27_cgp2.py`, (1+4)-ES, **pixel objective**). The C8 motion objective is not used,
    because it produced flicker.
  - **Capacity:** gate budget G = **32** (C7/C8) vs **128**. Active gates are reported separately.
  - Same 600 CPU-s per run. A failure of G=128 at an equal budget is **not** evidence that it cannot
    represent the action.
  - **Training regimes:**
    - **B-train:** target = the 16-frame window, start = the true window frame, hidden 0;
    - **C-train:** target = clip frames 0..t0+15, from the clip start.
  - Seeds 0–2 for each (task, G, regime). 2 tasks × 2 G × 2 regimes × 3 seeds = 24 runs = **4 CPU core-h**.
  - **Controls:** the C8 controls (sustained motion, periodic change, both structures, exact recovery) are
    reused, not re-run.
  - S=2 cannot be exact on the windows (light cone). The 1–3 conflicting pixels bound the error only
    marginally.
- **② Neural CA** (`ca/nca.py`): continuous state, deterministic.
  - **State:** C = 16 channels (1 visible + 15 hidden), hidden init 0, zero padding.
  - **Update:** 3×3 learned perception with Fh = 64 features, ReLU, a 1×1 residual update.
  - **Steps:** S = 4 per frame, so the receptive field is 4 px per frame (light cone passes on the windows).
  - **Size:** 10,320 float32 parameters. No coordinates, time input or per-position parameters.
  - **Output:** binary visible = s₀ > 0.5. The continuous state is carried unchanged at deployment
    (it is **not** binarised inside the rollout).
  - **It is not a Boolean CA.** It tests whether the action is reproducible at all with a local, shared,
    recurrent rule.
  - **Training:** JAX on GPU via gpuq.
    - Loss L2 on the continuous visible at every observed frame of the regime's target, with BPTT over
      the whole regime; per-step gradient checkpointing.
    - Adam lr 2e-3, per-tensor gradient normalisation (as in Richardson et al.'s trainer).
    - Last layer zero-initialised, so the model starts as hold.
    - Same B-train / C-train regimes.
    - 3 seeds vmapped per job, 900 s training per job.
    - **Checkpoint chosen by binary pixel error** of the regime's own autonomous rollout, evaluated
      every 30 s. Never chosen by loss or by motion metrics.
  - **Positive control:** a hand-set NCA in this exact architecture (`known_mover`).
    - A global hidden clock plus "visible′ = h ? W : C" moves objects right by 2 px per frame
      (3 objects, a 24×80 field, 30 frames).
    - Evaluator check: replays the symbolic shift exactly (done).
    - Training from random init on its frames only must reach an exact binary replay in ≥ 1 of 3 seeds.

## 4. Metrics (`ca/motion9.py`, binary frames of the window)
- pixel error and err/hold;
- **lag-1 and lag-4 tolerant directional change** precision / recall / F1. Lag 4 compares X_t with X_{t−4},
  so a period-2 flicker cancels and earns nothing.
- revert rate of the replay vs of the original. This is a diagnostic only: the original also reverts
  (0.16 / 0.07), so revert is never penalised.
- **Stage A metric:** err/hold of the one-step predictions, and **TF-change F1**: the tolerant F1 of the
  predicted change (P_t vs the true F_{t−1}) against the true change (F_t vs F_{t−1}). Copying the last
  true frame gives F1 0 and err/hold 0.30.

**Baselines on each window** (`window_baselines.json`):
- **Cheats:** hold, black, white, invert, edge checkerboard flicker, local on-then-revert blips, shift-right.
- **Reference:** true clip delayed by 1 frame.
- **Highest cheat-baseline lag-4 F1:**
  - main 0.24 (the C8 representative itself);
  - confirm **0.50** (shift-right).
  - Reference lag-4 F1: 0.89 / 0.97.

## 5. Pass criteria (fixed now)
**A pass:** TF-change F1tol ≥ 0.5 **and** err/hold ≤ 0.30 (at least as good as copying the last true frame).

**B pass / C pass** (window frames of the autonomous rollout), all of:
1. err/hold ≤ 0.70;
2. lag-4 F1tol ≥ **0.50 on main / 0.65 on confirm**: max(0.5, best cheat baseline + 0.15);
3. lag-4 tolerant precision ≥ 0.40;
4. the video shows the arm/staff motion. This can veto a numeric pass, never create one.

**Also reported:**
- per seed: pass/fail of A, B, B-warm and C, plus all raw metrics;
- per (model, capacity, regime): the number of passing seeds.

**Representative selection,** per task and model family:
- the (capacity, regime) with the most C passes, then the highest median lag-4 F1 in C;
- shown: its median seed, plus its best and worst seed.

**Reduction:** one method, **only if some C pass exists**.
- CGP: active-gate pruning.
- NCA: a smaller NCA (C=8, Fh=16) trained on the passing model's own replays (distillation).
- Otherwise the A/B/C diagnosis is finalised instead.

## 6. Budget and operation
- **Ceiling:** CPU 12 core-h, GPU 2 h.
- **Plan:**

  | part | budget |
  |---|---|
  | CGP | 4 core-h |
  | evaluation, export checks, videos | ≤ 1.5 core-h |
  | NCA control | 1 gpuq job, ≤ 10 min |
  | NCA main | 4 jobs (2 tasks × 2 regimes) × ≤ 20 min |
  | **GPU total** | ≤ 1.5 h |

- **Execution:** gpuq for every GPU job; frozen code; checkpoints, RNG state and configs saved
  immediately.
- **Export/verify:**
  - **CGP:** container files, Python vs Node byte match, as before.
  - **NCA:** an `nca-v1` JSON (float32 weights, config, first frame), replayed by numpy (`rollout_np`)
    **and** by an independent Node implementation (`player/nca_replay.mjs`). Reported:
    - binary-frame mismatches between implementations;
    - the maximum |Δvisible|;
    - the number of visible values within 1e-3 of the 0.5 threshold.

---
## Decisions after the main runs (appended before the reduction runs)
- **Stage C passes exist:** NCA C-trained, 3/3 seeds on both tasks.
- **Visual check:** the videos show the arm and staff motion, so the visual criterion is met and nothing is vetoed.
- **Reduction (the single pre-registered method):** distil the representative into a smaller NCA,
  C=8, Fh=16, 1,304 params.
  - Representatives, by the selection rule: main `nca_main_C_s2`, confirm `nca_confirm_C_s1`.
  - The targets are the teacher's own binary replay frames over the C regime.
  - 3 seeds, 600 s, one gpuq job per task.
  - Evaluated on stages A/B/C against the **original** frames, and compared with the teacher.
