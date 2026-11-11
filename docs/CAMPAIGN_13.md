# Campaign 13: whole-song demo (work log, criteria fixed before the whole-song result)

Started 2026-11-11 from commit `827ac5c`. Results go to `docs/CHECKPOINT_13.md`. This file is append-only after §3;
§0 is the live run state (kept current so the work can be resumed after a context change).

## 0. Live run state (update on every change)
| item | value |
|---|---|
| main run | gpuq **30** `c13-song-noise-s0`: whole song, resumed from the 9 noise-trained C12 chunks; `runs/c13/song_noise_s0`; internal limit 6.5 h |
| seed check | gpuq **34** `c13-song-noise-s1`: same recipe, seed 1, from scratch; `runs/c13/song_noise_s1`; internal limit 8.5 h |
| s0-tw2 | gpuq 36 (queued behind another session's 2-GPU job 35): `runs/c13/pilot_wipe_tw2` (chunks 0-6 from the pilot) resumed to the whole song with `--trans-weight 2`; internal limit 7.5 h |
| done | gpuq 29 (diagnosis), 31–33 (wipe pilots) |
| published | 2026-11-11: display name **Bad CA!!**; remote `origin` = https://github.com/VectorSophie/Bad-CA (`master`); first push `0c891d0` (merge of the remote LICENSE commit `c100dd2`, GPL-3.0; no force push). In-progress `runs/c13` not pushed. |
| last checkpoint | per chunk: `runs/c13/song_noise_s0/best_c{i}_s0.npz` + `chunks.json` (replay error per chunk) |
| resume | re-submit the same command; finished chunks are skipped |
| budget used | see `runs/c13/gpu_ledger.jsonl` (GPU, assigned GPUs × runtime) and `runs/c13/cpu_ledger.jsonl` |
| next | evaluate pilots → if a variant fixes the wipe without hurting, plan a 2nd whole-song run (seed 1, improved recipe) within budget |

## 1. Checked facts (from files, not from earlier reports)
- **Source:** 6,572 decoded frames at 30 fps → frames 0–6571.
- **Noise (C12):** `ca/phase_nca.jax_rollout(noise=...)` adds σ·N(0,1) to the **whole state** (visible + hidden)
  **after every frame** (after the S = 4 substeps), in training only. Replay never adds noise.
- **Carried state (C12):** chunk k started from the state the GPU computed with chunks < k (best checkpoints, no
  noise). Within a chunk, BPTT crosses phase boundaries; it does not cross chunk boundaries. No target is injected.
- **"Replay equals training" (C12)** meant per-chunk mean binary error equal to 3 decimals, not bit identity.
- **Stability diagnosis** (gpuq 29, `runs/c13/eval/diag_stability.json`): GPU (JAX) vs CPU (numpy) replay of the same
  export, per frame.
  - **Noise model 0–2115:** max |Δvisible| ≤ 3e-5. The first binary pixel difference is at frame 1866, and no frame
    has > 1 % of pixels different. Max |state| ≤ 31. **Stable.**
  - **C12 whole-song model (no noise):** first binary difference at frame 340, > 1 % at 980; the state grows to ~1e17.
  - **The fast wipe (bad 16-frame window 1499–1530, peak 0.159)** shows no GPU/CPU difference. It is a
    **quality** limit, separate from the numerical one.

## 2. Changes, each tied to an observed problem
1. **`--cpu-carry`:** the carried state between chunks is computed by the numpy reference, the same arithmetic as the
   export replay.
   - Problem it removes: training chunk k from a GPU state the replay never reaches.
   - It also records each chunk's **true replay error** in `chunks.json`.
   - Smoke test: the recorded value equals the export replay (0.123711 = 0.123711).
2. **`--max-hours`:** stops cleanly between chunks (resumable, partial export flagged).
   - Problem it removes: C12 could not cancel its own over-long jobs (permission policy).
3. **Wipe pilot** (problem: the 1499–1530 wipe; the phase boundary sits at 1510 while the wipe starts at 1505):
   - `--trans-weight 2`: per-frame loss weight 1 + 2·change/mean change;
   - the boundary moved 1510 → 1504;
   - both.
   - Each retrains chunk 6 (1468–1685) from the same 6 earlier chunks.
- Not changed: architecture, S, lr schedule, steps per chunk (8,000), σ = 0.02, the 110 automatic phases (uniform was
  not better in C12), learned low-res initial hidden.
- **Main run:** the C12 noise recipe + `--cpu-carry`, resumed from the existing 9 chunks.

## 3. Acceptance criteria for the whole song (fixed before any whole-song result; `experiments/exp52_accept13.py`)
Computed on the numpy replay of the export, plus the Node replay. The C12 gates are recorded next to them, unchanged.

| id | criterion |
|---|---|
| W1 | 240-frame window error ≤ 0.06 in ≥ 90 % of windows, and ≤ 0.15 in every window |
| W2 | frames inside failing 16-frame windows (error > 0.09) ≤ 5 % |
| M | where the original changes (lag-4 mass > 1 %): lag-4 change F1 ≥ 0.5 and change mass 0.5–1.5 in ≥ 90 % of windows |
| T | thin-part and thin-gap recall ≥ 0.70 |
| K | small specks and small holes ≤ 6 × original |
| B | spurious one-frame flips ≤ 1.0 per frame |
| P | mean error within ±3 frames of phase boundaries ≤ 2 × global error |
| F | fade frames reproduced (frame error ≤ 0.05) for ≥ 90 % of them |
| S | max \|state\| ≤ 1e3 at every frame; Node vs numpy binary mismatch ≤ 0.1 % overall and ≤ 1 % in every frame |

**Calibration** (done before the whole-song result; `runs/c13/eval/accept.json`):
- **The 0–1467 result the user accepted** passes all criteria except K at 3×: it has 3.6× the original's specks and
  5.0× its holes. K was therefore set to 6×. This is a recorded choice: the criteria encode "the accepted C12 look", not
  a stricter standard.
- **Majority blur of the true original (full song):** fails T (gap recall 0.63) and B (1.38 flips per frame). The C12
  loophole is closed.
- **0–2115:** passes all criteria with K at 6×. W2 = 1.5 % of frames fail (the wipe). It fails the C12 worst-window gate.

**"Whole song complete"** = all 6,572 frames replayed from the export by numpy, Node and the browser, and every
criterion above passes. Anything less is reported as not complete, with the failing criteria and windows listed.

## Budget
- **Caps:** GPU 48 GPU-h, CPU 64 core-h; diagnosis and pilots ≤ 4 GPU-h.
- **Whole-song estimate per seed (C12 measured):** ~15 min per chunk × 28 chunks ≈ 7 GPU-h. Resuming from 9 chunks
  ≈ 4.8 GPU-h.

## 4. Wipe pilot result and the run plan (appended before any whole-song result)
**Pilot** (chunk 6 only, single seed, `runs/c13/eval/wipe_pilot.json`):

| variant | wipe 1499–1530 error | worst 16-frame window | frames in failing windows |
|---|---|---|---|
| C12 | 0.100 | 0.159 | 17 |
| trans-weight 2 | 0.078 | 0.120 | 13 |
| boundary 1510 → 1504 | 0.094 | 0.158 | 16 |
| both | 0.074 | 0.121 | 12 |

- Transition weighting helps the fast wipe. The boundary shift alone does not. No variant reaches the 0.09 C12 window gate.

**Runs:**
1. **Main s0** (gpuq 30, running): the C12 recipe + numpy carry.
2. **s1** (new, now): the same recipe from scratch, seed 1. This is the reproducibility check of the procedure.
3. **s0-tw2** (after the main run frees a GPU): the pilot directory `runs/c13/pilot_wipe_tw2` (chunks 0–5 as C12, chunk 6
   with trans-weight 2) resumed to the whole song with `--trans-weight 2`.
- Projected total ≈ 18 GPU-h of 48.

**Representative selection (fixed now):** among whole-song exports that pass every §3 criterion on the numpy replay,
plus the Node check, the lowest full-song error becomes the featured model. If none passes, the interim 0–1467 model
stays featured and the whole song is reported as not complete.
