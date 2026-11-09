# Checkpoint 09: is there a representation + search that reproduces a short real Bad Apple action?

**Sources**
- **Plan:** `docs/CAMPAIGN_09.md`. The tasks, stages, thresholds and selection rule were fixed before
  training. Decisions taken after the main runs are appended there, following those rules.
- **Machine-readable:**

  | file | contents |
  |---|---|
  | `runs/c9/results.json` | 42 models × stages A, A-reset, B, B-warm, C, with pass flags and verification |
  | `runs/c9/summary09.md` | readable table |
  | `runs/c9/window_baselines.json` | baselines and certificates |
  | `runs/c9/task_candidates.json` | task scan |

- **Exports:**
  - CGP: `runs/c9/cgp/*.best.cab` (container files);
  - NCA: `runs/c9/exports/*.nca.json` (float32 weights, config, first frame);
  - per-stage replays: `runs/c9/replays/*.npz`.

**Tasks** (chosen before training; candidates and reasons in CAMPAIGN_09 §1):
- **main:** `ba48_4500_60` frames **4540–4555**, an arm rises and bends.
- **confirm:** `ba48_1800_60` frames **1844–1859**, the staff and arm swing.
- **Preprocessing:** unchanged (48 px, threshold 128). Changing the threshold barely changes the motion.

## 1. Videos to open first
All files are in `runs/c9/media/`.
- **Formats:** MP4 1× (30 fps) and `_0.25x` (7.5 fps), plus GIFs. All were decoded back to check frame
  count and fps.
- **Opening:** from the repository root, `xdg-open runs/c9/media/<name>_1x.mp4`, or mpv/VLC/a browser.

| file (`…_1x.mp4`) | shows |
|---|---|
| `main_1_orig_c8_cgp_nca_autonomousC` | original / C8 CGP / C9 CGP / **C9 NCA**. All four run autonomously from frame 4500 (stage C), with no injection and no reset. |
| `main_2_NCA_orig_tf_autonomous` | original / NCA teacher-forced (stage A) / NCA autonomous (C) / NCA continuous visible (a separate grey panel) |
| `main_3_NCA_{rep,best,worst}_orig_C_diff` | original / NCA stage-C replay / diff (red = missed white, blue = extra white) for the median, best and worst seed |
| `main_4_reduction_teacher_vs_distilled` | original / NCA 10,320-param teacher / **distilled NCA, 1,304 params** / diff |
| `main_2_CGP_*`, `main_3_CGP_*` | the same for the CGP representative |
| `confirm_*` | the same set for the confirm task |
| `control_nca_mover_orig_recovered` | NCA positive control (the CGP controls are reused from `runs/c8/media/control_*`) |
| `pre_task_*_c8` | made before training: the task window vs the C8 result |

**Panel labels:** model, capacity, training regime, seed, parameters or gates, hidden channels, S, and
whether teacher forcing or a reset is used.

## 2. Is there a candidate that autonomously reproduces the real action?
**Yes, with a continuous-state Neural CA. No, with Boolean CGP circuits.**
- **NCA:** C-trained Neural CA, 16 channels (1 visible + 15 continuous hidden), a 3×3 learned
  perception with 64 features and 1×1 update, S=4, **10,320 float32 parameters**.
  - Running autonomously from the clip's first frame, with hidden 0 and no injection, it passes the
    pre-registered stage-C criteria in **3/3 seeds on both tasks**.

    | task | lag-4 tolerant F1 | err/hold |
    |---|---|---|
    | main | 0.97 / 0.75 / 0.91 | 0.14–0.32 |
    | confirm | 0.95 / 0.95 / 0.92 | 0.22–0.24 |

    The best cheat baselines reach 0.24 (main) and 0.50 (confirm).
  - In the videos, the arm rises and bends (main), and the staff and arm swing (confirm).
  - **It is not a Boolean CA,** and with about 330 kbit of weights it is **not a small rule** and not compression.
- **CGP:** circuits of 8–25 active gates, G=32 or 128, pass **no stage** in any of the 24 runs.
- **NCA verification:**
  - Every NCA model replays from its export alone, standalone, in numpy and in the independent Node
    player (`player/nca_replay.mjs`).
  - **0 binary-pixel mismatches** in all 18 NCA/dNCA models.
  - Maximum |Δ continuous visible| 2.6e-6 to 8.5e-5.
  - The representatives have 11 (main) and 17 (confirm) continuous visible values within 1e-3 of the
    0.5 threshold, yet still no binary mismatch between numpy and Node.
  - "Implementations agree" and "matches the original" are reported separately: the original is
    matched approximately, not exactly.

## 3. Stage A / B / C: number of seeds passing each stage

| model (capacity, train regime) | A main | B main | B-warm main | **C main** | A conf | B conf | **C conf** |
|---|---|---|---|---|---|---|---|
| CGP G=32, B-trained | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| CGP G=32, C-trained | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| CGP G=128, B-trained | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| CGP G=128, C-trained | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| NCA 16×64, B-trained | **3/3** | **3/3 (exact)** | 0/3 | 0/3 | 3/3 | 3/3 | 0/3 |
| NCA 16×64, C-trained | 0/3 | 0/3 | 2/3 | **3/3** | 0/3 | 0/3 | **3/3** |
| distilled NCA 8×16, C (reduction) | 0/3 | 0/3 | 0/3 | **3/3** | 0/3 | 0/3 | **2/3** |

- **A (teacher-forced one-step):**
  - B-trained NCAs predict the next frame almost perfectly (TF-change F1 0.97–1.00).
  - C-trained NCAs **fail A** (0.04–0.10), as do all CGPs (best 0.56, err/hold ≥ 0.34).
- **B (true window frame once, hidden 0):**
  - B-trained NCAs reproduce the 16-frame window **exactly**.
  - CGP comes closest at lag-4 F1 0.42–0.51 with err/hold 0.65–0.80. No run meets all criteria.
- **C (from the clip start):**
  - only C-trained NCAs pass, and their distilled versions;
  - B-trained NCAs fail C (lag-4 F1 0.18–0.37, err/hold 2.3–3.6).
- **The crossed pattern is the key evidence:**
  - each NCA passes only the stage it was trained for;
  - the C-trained model cannot predict one step from the true frame;
  - the B-trained model collapses when started 40 frames earlier.
  - This is **consistent with trajectory-specific solutions**: the continuous hidden state carries
    timing and phase built up from its own start, not a general local motion rule. It does not prove it.
  - B-warm (hidden from a teacher-forced warm-up) works for only 2/3 main C-models and no B-models, which
    fits the same picture.

## 4. Effect of capacity and representation
- **CGP capacity (G=32 vs G=128 at equal 600 CPU-s):** no detectable difference.
  - B-regime lag-4 F1 median 0.47 vs 0.44 (main) and 0.39 vs 0.32 (confirm).
  - Active gates are 8–18 in both.
  - With equal time the larger search space is not explored more, so this is **not** evidence that
    128 gates cannot represent the action.
- **Representation change (CGP → NCA):** decisive for passing B and C.
  - It changes several things at once: continuous state, 15 hidden channels instead of 1 bit, S=4 vs 2,
    gradient training vs evolution, about 10⁴ parameters vs about 10 gates.
  - So which factor matters is **not separated**.
  - The NCA learned its sustained-motion positive control from scratch in 60 s, exactly; the CGP
    controls from C8 also pass. Both searchers can find sustained motion **when a model of their own
    family generated it**.
- **Reduction** (the one pre-registered method, distillation into C=8, Fh=16, 1,304 params, trained on
  the teacher's own replay):
  - **main:** 3/3 still pass C (lag-4 F1 0.72–0.86, err/hold 0.28–0.37, vs the teacher's 0.91 / 0.19);
  - **confirm:** 2/3 pass (0.57–0.72, err/hold 0.47–0.56).
  - Quality drops but the action stays recognisable (`*_4_reduction_*` videos).
  - 8× fewer parameters, still continuous: **not a Boolean rule.**

## 5. What can be concluded about representation limits, search failure and rollout accumulation
- **Information and feasibility:**
  - the light cone passes at S=4 on both windows;
  - exact stage C at S=4 on confirm is impossible for any radius-1 constant-init model (1 pixel at
    frame 3). The NCA's approximate pass is consistent with that.
  - CGP at S=2 cannot be exact on the windows (1–3 conflicting pixels), but that bounds only a few
    pixels and does not explain its failures.
- **Rollout accumulation:** for the NCA it is real but **avoidable by training on the full rollout**.
  - B-trained models are exact for 15 frames from the true window frame, yet fail when the rollout
    starts 40 frames earlier. C-trained models pass.
  - It is the training horizon that decides it, not capacity.
- **CGP: search failure vs representation limit — not separable here.**
  - Even in B, with a true start frame and a 16-frame horizon, CGP reaches only lag-4 F1 ≤ 0.51.
  - Its teacher-forced one-step change F1 is ≤ 0.56, so even **one-step** prediction from true
    frames is weak at 8–25 gates.
  - This fits "small circuits with 1 hidden bit cannot express this local motion" **or** "(1+4)-ES
    does not find them in 600 s". The G=128 result does not separate the two.
  - The NCA's ~0.98 one-step F1 shows that the information needed for one-step prediction is present
    in the local 3×3 context plus hidden state, given enough continuous capacity.
- **NCA generality:** the crossed A/B/C pattern means the passing NCAs are **specific to their start
  state and horizon**. Training a model that passes A, B and C at once was not tried.

## 6. Rule size, storage and compute
- **Storage:**

  | model | export | raw weights | vs best recording of the frames it replays |
  |---|---|---|---|
  | NCA | 56.2 kB JSON | 41.3 kB float32 | lzma 1,559 B (main 56 frames), 3,031 B (confirm) |
  | distilled NCA | 8.2 kB JSON | 5.2 kB float32 | same |
  | CGP | 92–162 B container | – | – (CGP does not reproduce the motion) |

  **The NCA models are 3–40× larger than a recording of the frames. Not compression.**
- **Replay cost:** NCA (T−1)·S = 220 steps × 1,728 cells × (9·16·64 + 64·16) MACs, about 3.9 GMAC; the
  Node replay takes seconds.
- **Compute:**
  - **CPU:** 4.09 core-h of the 12 ceiling (CGP 14,410 s; evaluation, exports and Node checks 216 s;
    videos 60 s; baselines 39 s).
  - **GPU:** 1.51 h of 2, 7 gpuq jobs.

    | jobs | what | wall |
    |---|---|---|
    | 59 | NCA control | 10.1 min |
    | 60–63 | NCA main/confirm × B/C | 4 × 15.1 min |
    | 64–65 | distillation | 2 × 10.1 min |

    There were no failures. Compile time was 1.5 s per job, inside the job wall time.

## 7. One path to focus on next, and why
**Train a recurrent NCA on all three stages at once and test whether it learns a general local motion
rule rather than a trajectory.**
- Concretely: rollouts from the clip start, from mid-clip true frames with hidden 0, and
  teacher-forced steps, in one objective. Then measure A, B and C together.
- **Why this one:**
  - it is the only family that reached the original goal (C), with a matching video;
  - but its passes are crossed and stage-specific, which suggests memorisation of a trajectory;
  - a model that passes all three is the precondition for any meaningful small-rule extraction
    (distillation to discrete gates).
- Further CGP capacity or seeds are not recommended until a model that passes A exists to distil from:
  CGP failed even the teacher-forced one-step stage.

## Reproduction
- **Snapshots:**

  | snapshot | used for |
  |---|---|
  | `runs/frozen/4ede93988577` | CGP runs, NCA control |
  | `runs/frozen/60c677c5cc77` | NCA main/confirm |
  | `runs/frozen/0c626a76f65c` | distillation |
  | `runs/frozen/14fabab48c98` | final evaluation |
  | `runs/frozen/4ffc2ea1bafd` and later | videos |

- **Search and training:**
  - `python experiments/exp27_cgp2.py "<clip>@a-b" k1_S2 {32|128} SEED 600 pix 0 c9/cgp`
  - `gpuq -G 1 python experiments/exp33_nca.py {main|confirm|control} {B|C} 900 c9/nca [--small 8 16 --teacher EXPORT]`
- **Analysis:**
  - `python experiments/exp34_eval9.py` (stages, exports, Node checks)
  - `python experiments/exp35_videos9.py` (videos)
  - `python experiments/exp30_tasks.py`, `exp31_prevideo.py`, `exp32_window_baselines.py` (pre-training)
- **Standalone NCA replay:** `node player/nca_replay.mjs runs/c9/exports/<id>.nca.json`.
