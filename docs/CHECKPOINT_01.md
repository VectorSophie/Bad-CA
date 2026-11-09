# Checkpoint 01 — exact-LUT baselines, independent replay, measured limit

Date 2026-11-08. All numbers below come from `experiments/exp01_lut_baselines.py` (results in
`runs/exp01/results.json`) and `experiments/exp02_resolution_and_minimize.py`. CPU only; whole run takes about 3 s, plus about 15 s to decode the video once.

## Environment
- 2× RTX PRO 6000 Blackwell (98 GB), 128 cores, 251 GB RAM. No system ffmpeg. PyAV 19 (bundled FFmpeg) used for decode.
- `.venv` (uv, Python 3.12): numpy 2.5.4, sympy 1.14, av 19.0.2, pillow 13. JAX CUDA was still downloading at the time of writing, so the JAX/GPU smoke test is still pending.

## Sources inspected (shallow clones in `refs/`)
| repo | commit | license file |
|---|---|---|
| C-mykk/Bad-Apple-Starter | 7b04ba8e | **none** (local research use only; don't redistribute frames) |
| maxencefaldor/cax | 5c8f14e9 | yes |
| rplsmn/logicars | 1f316a43 | **none** |
| google-research/self-organising-systems (notebooks/ sparse) | 3d5547ca | yes (Apache-2.0 repo) |
Code from these is not reused yet. They are cloned for reading only.

## Data validation (Bad Apple, starter repo's mp4)
- h264 480×360, 6572 frames, constant 1/30 s PTS, 0 → 219.033 s. Frame count matches the starter's 6572.
- The starter's own JSON/TXT has two issues: `glob()` is not sorted, though consecutive-diff checks show the order is temporally coherent on that disk, and `resize((36,48))` squashes 4:3 landscape into portrait. **Not used.** We decode the mp4 ourselves with PyAV, box downsample at preserved 4:3, cache grayscale, then threshold at 128.
- 182 frames are spatially uniform, including **frames 0–42 (the black opening)**. A homogeneous deterministic CA with a constant boundary cannot leave a uniform state, so every clip here starts at a non-uniform frame. Song-start playback needs an explicit anchor, boundary or seed decision (open item).

## Results

**Correctness.** The LUT stepper matches an independent naive per-cell Life over 30 steps of a 32×32 soup, and the glider translates (1,1) after 4 steps. The Node replayer (`player/replay.mjs`) shares no code with the Python side and reads only the declarative export. **Its SHA-256 matched the Python rollout on all 17 exported runs** (1 Conway, 3 synthetic, 12 Bad Apple, 1 minimized rule).

**Known-rule recovery.** A 64×64 Conway trajectory (20 steps) visits all 512 Moore contexts with zero conflicts. Majority fit equals B3/S23 exactly, and the outer-totalistic fit returns {B3, S2, S3}. Rollout is exact.

**Synthetic motion (24×24, 1 generation/frame, no hidden state).**
| clip | hold-frame err | Moore-LUT one-step floor | Moore-LUT rollout err |
|---|---|---|---|
| translate | 0.089 | 0 | **0 (exact)** |
| grow→shrink disc | 0.095 | 0.045 | 0.095 (= hold) |
| rotating bar | 0.136 | 0.069 | 0.136 (= hold) |
Translation is solved exactly. Growth followed by shrinkage, and rotation, are each 100% conflicting: identical local contexts need opposite successors.

**Bad Apple, 64×48, 12 clips (6 start times × 30/90 frames × 3 boundaries).**
- The best memoryless Moore LUT's one-step error floor is about equal to the clip's change rate. At 64 px, **only ~3% of changed pixels are predictable from the radius-1 context.** Radius-2 explains 14–30% (more at lower resolution), but it sees thousands of distinct contexts in a 90-frame clip, so that part is mostly memorized and is not a compact rule.
- Autonomous rollout therefore stays near the freeze-frame baseline. It beats hold only on a few slow 30-frame clips (e.g. t=30 s: 0.091 vs 0.219; t=100 s: 0.038 vs 0.079), and change-IoU is ≤0.08 everywhere. Conway rollout is far worse than hold (0.24–0.82).
- Boundary choice (0/1/wrap) moves results by <0.02.
- The resolution sweep (16/32/64/128 px) leaves radius-1 explained change at 3.1–3.6%. **This is not a resolution artifact.**

**Compactness via compiler-style simplification.** The fitted LUT for t=30 s/30 f sees 130 contexts; the other 382 are don't-cares. Quine–McCluskey (sympy SOPform) reduces it to **11 product terms / 21 ops** in 1.4 s. Rollout error goes 0.09098 → 0.09104, and the JS replay of the minimized export matches. Treating unseen contexts as don't-cares works, but the minimized rule must be re-rolled because it changes behavior off the observed trajectory.

## What this means for the plan
1. **Search was never the bottleneck for memoryless rules.** For 1 generation/frame and no hidden state, majority vote gives the best teacher-forced LUT in milliseconds. DiffLogic over the same inputs cannot beat that floor, because it represents a subset of the same functions. The suggested learner only adds value once the model includes **unobserved variables**: hidden bits per cell, several generations per displayed frame, or both. Gradient search, SAT and constraint projection (Elser) belong in that regime.
2. Bad Apple motion is not locally determined from one binary frame. A cell needs memory, such as which way an edge is travelling, which is exactly the hidden state that has to be found. The identical-state argument from the story shows up here at the scale of local contexts.
3. Pixel error misleads, as predicted: hold-frame already scores 0.02–0.33. Change-IoU is the metric that exposes failure, so keep it as a primary metric.

## Next experiment (proposed, not run)
**Exp03: hidden-state synthesis on the synthetic failures, then a slow Bad Apple clip.**
- Hypothesis: 1–2 hidden bits per cell (constant-0 init) plus 1–2 generations/frame make grow→shrink and rotate solvable exactly at 24×24, and raise Bad Apple change-IoU above the 0.08 memoryless ceiling.
- Methods, at equal wall budget: (a) exact SAT/CEGIS synthesis (unknowns = LUT entries + hidden trajectory, as a discrete counterpart to Elser's projection); (b) DiffLogic circuit with hidden channels and autonomous-rollout loss, then hardening, on GPU through `gpuq`. Baselines are this checkpoint's memoryless LUT and hold-frame.
- Budget: 5-min pilot per method per clip to measure scaling. Stop if neither beats the memoryless floor on rotate_bar within 30 min. Report hidden bits, rule size after minimization, and the soft→hard gap.
- Open issue to settle alongside: the song's uniform black opening (anchor vs. boundary vs. seeded hidden state, with its information cost).
