# Campaign 11: a smaller model, longer playback, video→CA pipeline and local UI (plan, fixed before the main runs)

Written 2026-11-10 after the pilot and **before any main run**. Later decisions are appended at the end. Results go
to `docs/CHECKPOINT_11.md`.

**Terminology:**
- The models are continuous-state Neural CAs. A parameter count is not a rule count, and these are not Boolean CAs.
- "Autonomous playback" means:
  - start from the encoded first frame with hidden 0;
  - apply the exported local update S = 4 times per frame;
  - no target frames, resets, corrections, time or position inputs after the first frame.

## 0. State checked
- **Repository:** HEAD `be53ddf`, clean. There is no `AGENTS.md`; the project instructions are `claude_ca_*.md`.
- **Frame indexing:** 0-based decoded frame index of `refs/Bad-Apple-Starter/BadApple!!.mp4`.
  - sha256 `d8f21232…cd0`: 6,572 frames, 30 fps, 480×360.
  - Grids are 48×36, box resize, threshold `gray >= 128` (`ca/video.py`, `experiments/exp03_sat_ablation.py`).
  - The `main` / `confirm` targets here equal the C9/C10 targets bit for bit (checked).
- **Provenance:** the file comes from the cloned `C-mykk/Bad-Apple-Starter` repo, whose README says it was fetched
  with yt-dlp from a YouTube URL. That upstream identity is **not verified**; only the sha256 above is recorded.
- **Teachers** (frozen; `runs/c11/teacher_sha256.txt`):

  | task | export | sha256 | full-span error (export replay) |
  |---|---|---|---|
  | main | `runs/c10/exports/c10_combo10_main_best_BC_s0.nca.json` | `d969a5be…8b47` | 0.0016 |
  | confirm | `runs/c10/exports/c10_combo10_confirm_best_BC_s1.nca.json` | `abc81a83…69a3` | 0.0130 |

  - The continuous visible ranges −0.73…1.93 (main) and −0.50…1.73 (confirm).
  - It is a raw state value, not a probability. It is clipped to [0, 1] before use as a soft target.
  - The teacher's binary output agrees with the target on 99.84 % / 98.70 % of pixels.
- **Parameter counts** (from `nca.n_params`): 8×16 → **1,304**; 6×10 → 616; 12×32 → 3,884; 16×64 → 10,320.

## 1. Fixed settings (all parts)
- **Fixed for every model:**
  - 48×36 grids, S = 4, 3×3 zero-padded perception, ReLU, 1×1 residual update;
  - float32 state and compute;
  - hidden 0, threshold 0.5.
- **Trainer:** `ca/nca_train.py`, shared with the pipeline.
  - Fixed **30,000** optimizer steps.
  - Adam with per-tensor gradient normalisation.
  - **Cosine lr 2e-3 → 2e-5.**
  - Loss: L0 + **2·LB** (the C10 contour hinge).
  - Every 500 steps the binary error of the full autonomous rollout is computed. The best checkpoint is kept, and a
    resumable state is saved.
- **Pilot** (gpuq jobs 78–81; `runs/c11/pilot`, `runs/c11/probe`). Main task, 1,304-param model, 3 seeds, 6,000 steps:

  | run | best binary error |
  |---|---|
  | lr const 2e-3 | 0.0255 / 0.0285 / 0.0266 |
  | cosine 2e-3 → 2e-5 | 0.0261 / 0.0278 / 0.0273 |

  - Both are tied within seed noise and still improving.
  - **Cosine** was chosen because C10 showed most of the gain comes in a low-lr phase, and with a constant
    2e-3 the best checkpoint is a lottery (C10 pilot 1). This is a judgement, not a measured win.
  - **30k steps** (≈ 17.5 min per 3-model job) was chosen from the curves and the budget.
- **Distillation:** λT = 1, fixed and **not tuned**.
  - `LT = Σ A·(v − clip(t,0,1))² / Σ A`, where A = pixels where the teacher's binary agrees with the target.
  - So teacher specks and holes are never taught, and the target loss is always kept.
  - Hidden channels are never matched, and no teacher hidden state is imported.
- **Direct vs distilled:**
  - Identical initial weights per seed (`default_rng(seed)`).
  - Identical steps, target frames every step, and selection.
  - Separate jobs; actual GPU time is reported separately.
  - Teacher generation is one numpy replay (< 1 s CPU).

## 2. Part 1: smaller model on main (4500–4555) and confirm (1800–1859)
**Arms:**
- A: C9 1,304-param distilled models (historical reference, exports from C9).
- B: 1,304 direct.
- C: 1,304 distilled.
- B and C use 3 paired seeds per task.

**Gates.** Median over seeds, full span, measured with `experiments/exp41_c11_eval.py`. "C10" = the C10 B+C medians in
`runs/c11/eval/ref_*.json` (main / confirm):

| gate | C10-level threshold, main | C10-level threshold, confirm |
|---|---|---|
| pixel error ≤ 1.5 × C10 | 0.0060 | 0.0201 |
| late-third error ≤ 1.5 × C10 | 0.0090 | 0.0152 |
| lag-1 change F1 ≥ C10 − 0.05 | 0.861 | 0.721 |
| change mass (lag 1) ≥ C10 − 0.10 | 0.71 | 0.53 |
| thin-part recall ≥ C10 − 0.05 | 0.878 | 0.829 |
| thin-gap recall ≥ C10 − 0.05 | 0.766 | 0.869 |
| spurious blips ≤ max(2 × C10, C10 + 10) | 26 | 58 |
| small specks: \|pred − true\| ≤ \|C10 − true\| + 10 | ≤ 17 (true 0) | within 12 of 104 |
| C9 window rule | ≥ 2/3 seeds | ≥ 2/3 seeds |

- Plus a visual veto (it can only veto).
- **C9-level** (usable but noisier): error ≤ C9-large median (0.0129 / 0.0266), late third ≤ C9 late (0.0226 / 0.0297),
  and the C9 rule ≥ 2/3.

**Size decision:**
- If B or C at 1,304 reaches C10-level on both tasks, the optional 616-param model is tried, with the same recipe and
  the better of B/C.
- Otherwise **one** intermediate size, 12×32 = 3,884, is tried with the same recipe and the better of B/C.
- The "selected small architecture" is the smallest that reaches C10-level on both tasks.
  - Failing that, the smallest at C9-level, stated as reduced quality.
  - Failing that, none, and the 10,320 model is used everywhere.
- **Representative seed:** the median full-span error. Best and worst are also reported. All seeds are listed.

## 3. Part 2: longer autonomous playback (intervals fixed before any long training)
**Source inspection** (`runs/c11/media/interval_sheet_*.png`; per-frame change and uniform frames):
- 4500 onward has fades to black at 4568–4573 and 4594–4599. Rejected for long runs.
- 1686 is a full-frame inversion cut (per-frame change 0.966).
- 1686–1759 is an inverted scene that morphs into a horizontal line.
- The dancer emerges from the line at 1760 and dances with the cane until 1879.
- 1880–1882 is a large wipe, then a different shot; another shot change follows near 1911.
- Song-wide scan: no 240-frame window with sustained motion is cut-free. The best scores (5630, 2780) also contain
  morphs and zooms.

**Intervals:**
- **L120 = 1760–1879:** one continuous scene, no cut. The figure emerges, then the cane swings. It contains the
  confirm interval.
- **L240 = 1760–1999:** nested, same first frame. It **contains the 1880 wipe and the ~1911 shot change**, so it
  doubles as the transition stress test.

**Runs:**
- L120: the 10,320 model and the selected small model, 2 seeds each, 30k steps, trained directly on L120.
  - There is no long teacher, so no short-clip teacher is used.
- L240: only if L120 shows sustained motion.
  - One seed (exploratory), the large model, plus the small model if it succeeded at 120.
  - Initialised from the L120 best checkpoint of the same architecture and seed, then trained on the full 240 target.
    This is a warm start, recorded as such. The evaluation is always a full rollout from the first frame.
- If both fail at 120: one bounded diagnostic, then stop extending.

**"Sustained motion" at a length** (per seed, original time alignment), all of:
1. In each third (early / middle / late), lag-4 change F1 ≥ 0.5 and lag-4 change mass 0.5–1.5.
2. Late-third error ≤ 3 × early-third error, or ≤ 0.03.
3. Visual check (veto only).

## 4. Part 3: storage
- **Final candidates:** export to `ncab-v1` (`ca/ncab.py`), with float32 and float16 weight storage.
  - Reload, then do a full autonomous replay with float32 state and compute.
  - Float16 is kept only if the binary output and gates are unchanged within the same thresholds; otherwise float32
    stays the default.
- **Byte breakdown:** weights / first frame / header, the deflated file, and the JSON presentation overhead.
- **Recording baseline, same convention:** the binary frames packed 1 bit per pixel, then the same zlib-9, plus lzma.
  - This is a reference only. An approximate replay is not lossless compression.

## 5. Part 4–5: pipeline and UI
- **Pipeline:** `ca/pipeline.py` (`python -m ca.pipeline ...`), documented in its docstring and in CHECKPOINT_11.
- **UI:** local server, bound to 127.0.0.1, in `ui/`.
- **Development costs** are reported separately from experiment costs.

## Budget
- **Ceiling:** GPU 6 h aggregate, CPU 12 core-h, pilots and failures included.
- **Ledger:** CPU in `runs/c11/cpu_ledger.jsonl`; GPU from gpuq job times.

## Decisions appended during the runs
- **Long-run recipe:** L120 large-model run 1 (gpuq 82, lr 2e-3) went **NaN** in both seeds within 500–1,500 steps. It
  was cancelled at 10.9 min.
  - The short probe (fast decay) had trained normally, so the diagnosis is numerical: a 480-step rollout explodes
    at a high lr.
  - Before any long result existed, the long-run recipe was changed to **cosine lr 5e-4 → 5e-6**, with a trainer
    guard that skips a non-finite step per model (sampled counts in `hist.json`).
  - This applies to all L120/L240 runs of both sizes. It is **not** the "one repair after both fail" of §3.
- **Queue incident:** around 14:58 the task-spooler queue was reset, by another session or a server restart.
  - Running jobs 83 (main-D) and 84 (main-T) continued as orphan processes and are timed from their logs.
  - Queued jobs 85 and 86 (confirm-D and confirm-T) were lost before they started and were resubmitted unchanged
    with the same frozen code.
  - Job ids restarted at 0.
- **Queue incident 2:** around 15:25 the queue was reset again.
  - It now lists another session's job (`p3-pipeline`, id 0, GPU 1).
  - My jobs 0 (L120 r2) and 1 (confirm-D) continue as orphan processes.
  - Queued job 2 (confirm-T) was lost again and will be resubmitted.
  - To avoid stacking on GPUs the queue believes are free, new submissions wait until my orphan processes finish.
- **Size decision (Part 1):**
  - The 1,304-param direct model on main finished at **0.0174 / 0.0178 / 0.0175**. It fails both the C10-level gate
    (≤ 0.0060) and the C9-level gate (≤ 0.0129).
  - A model must pass on both tasks, so 1,304 cannot be the selected architecture via direct training.
  - Distillation at matched steps (2k / 5k / 8k / 10k) was not better than direct training (e.g. 10k: D 0.025–0.027,
    T 0.027–0.029). It is still running to 30k and will be reported in full.
  - Per §2, the **one intermediate size, 12×32 = 3,884 params**, is trained now with the direct recipe on both tasks
    (3 seeds, 30k steps, same settings).
  - If distilled 1,304 ends clearly better than direct, that is reported. A distilled 3,884 run is added only if
    budget remains.
- **Matrix reduction (budget):** the GPUs are shared (another session's job, plus my orphaned runs), and steps run
  2–3× slower than in the pilot.
  - To keep budget for length extension, **distillation on the confirm task was dropped**: confirm-T was cancelled
    after ~3 min, at step ~1,000; its partial run is not used.
  - The distillation comparison is reported on **main only** (3 paired seeds, complete). Confirm is trained directly.
- **L120 small model:** the 1,304 size fails even C9-level on both tasks.
  - The 3,884 model is the only small candidate left, so its L120 run (2 seeds, lr 5e-4, 30k steps) is submitted
    **before** its Part 1 verdict, to fit the budget.
  - If 3,884 fails the Part 1 gates, its L120 result is reported only as a size-limited reference.
- **GPU budget accounting:** with shared GPUs, per-job wall time overstates use. The budget is counted as **device
  occupancy**: for each GPU, the time during which at least one of my jobs is resident; the upper bound is
  2 GPUs × the time since the first submission (~14:55). Per-job wall times are also listed.
- **Part 1 verdict** (gates of §2, 3-seed medians, full span; `runs/c11/eval/p1_*.json`):
  - **1,304 direct:** main 0.0175, confirm 0.0470. Fails C10-level and C9-level on both tasks.
  - **1,304 distilled** (main): 0.0218. Worse than direct at identical steps.
  - **3,884 direct:**
    - Main: 0.0062, late third 0.0129. It misses C10-level on error (gate 0.0060) and on the late third (gate 0.0090).
      Every other C10-level gate passes. It **passes C9-level**.
    - Confirm: 0.0287, above the C9-level gate of 0.0266. **Fails** C9-level.
  - **Result by the pre-registered rule:** no smaller architecture qualifies on both tasks. The 10,320 model remains
    the reference. The 3,884 model is reported as **usable on main with reduced quality** (C9-level), not as a
    general replacement.
  - The L120 comparison keeps 3,884 as the small model; its run was launched before this verdict.
- **L120 large (10,320):** full-span error 0.0304 / 0.0323. Both seeds pass the sustained-motion gate: lag-4 F1 per
  third 0.85 / 0.81 / 0.95 and 0.85 / 0.76 / 0.94; change mass 0.56–0.87.
  - So L240 runs for the large model: **1 seed (exploratory)**, initialised from `L120/big16x64_r2/best_s0.npz`.
  - 12,000 steps (budget-limited), cosine lr 5e-4 → 5e-6, trained on the full 240-frame target, evaluated by a full
    rollout from frame 1760.
- **L120 small (3,884):** full-span error 0.0611 / 0.0537. It **fails** the sustained-motion gate in both seeds.
  - Middle-third change mass is 0.37 / 0.45, i.e. under-motion.
  - So L240 runs only for the large model. This is recorded as the size limitation.
  - With only 2 seeds per L120 group, the videos show **both** seeds; seed 0 is used where a single panel is needed.
    This rule was fixed here, before the videos.
- **L240 warm start failed** (gpuq 7, cancelled at step ~5,200, ~11 min).
  - The L120 model's state blows up beyond its trained 120 frames: at step 0 the binary error over 240 frames is
    0.243 and L0 is 3.3e4.
  - Within 400 steps the loss became non-finite. From then on the non-finite guard skipped every step, so the run
    was stuck. Limitation of the guard: it cannot recover once the weights sit in a non-finite region.
  - **The one bounded repair (§3):** L240 from scratch with the L120 recipe (seed 0, lr 5e-4 cosine, 15,000 steps;
    budget-limited, fewer than L120's 30k). If it also fails, the extension stops at 120.
- **L240 repair** (gpuq 8, from scratch, 15,000 steps, 27.2 min):
  - Full-span error 0.0656; thirds 0.069 / **0.100** / 0.028. The middle third contains the 1880 wipe and the 1911
    shot change.
  - Lag-4 change mass 0.45 / 0.41 / 0.68. It **fails** the sustained-motion gate.
  - The extension **stops at 120 frames**, as pre-registered. The 240 export is kept only as a labelled failure example.
- **Pipeline bug found while making the L240 example:** `prep` updated every output slot for every decoded frame, so a late interval needed O(frames × decoded) image conversions. The slow import was killed after ~10 min. Fixed to update only the nearest slot; results are unchanged (`tests/test_pipeline.py` passes, Bad Apple prep is still bit-identical).
