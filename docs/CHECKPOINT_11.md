# Checkpoint 11: smaller model, longer playback, video→CA pipeline and local UI

**Plan and every decision made during the runs:** `docs/CAMPAIGN_11.md`.
**Results table (every seed):** `runs/c11/summary11.md`. Machine-readable versions:
- `runs/c11/eval/*.json`;
- the storage results: `runs/c11/eval/storage_*.json`;
- the implementation drift: `runs/c11/eval/node_drift.json`.

**Terms used below:**
- The models are **continuous-state Neural CAs** (learned float32 weights). A parameter count is not a rule count.
- Every number comes from an **autonomous replay of the export**: the first frame plus hidden 0, nothing injected
  afterwards, scored against the original over the **full** interval.
- "Representative" = the median-error seed of three. Groups with two seeds show both.

## Open first
1. **UI:** `.venv/bin/python ui/server.py`, then open http://127.0.0.1:8765/ (local only; stop with Ctrl-C).
   - It has seven ready examples, e.g. `example_c10_main` (C10) and `example_c11_L120_10320` (C11, 120 frames).
2. `runs/c11/media/c11_main_ZOOM_arm_rows4-31_cols0-27_0.25x.mp4`: arm/shoulder ×12, original / C10 / 3,884 / 1,304 / diffs.
3. `runs/c11/media/c11_L120_FULL_1760-1879_orig_large_s0s1_small_s0s1_1x.mp4`: 120-frame autonomous replay, large vs small.
4. `runs/c11/media/c11_main_FULL_4500-4555_orig_C10_1304direct_1304distilled_1x.mp4`: direct vs distilled 1,304.
5. `runs/c11/media/c11_L240_FULL_1760-1999_orig_large_diff_EXPLORATORY_1x.mp4`: the 240-frame attempt, a failure example.

**Video formats and checks:**
- All exist at 1x and 0.25x, as MP4 and GIF.
- Each was decoded back: frame count 56 / 60 / 120 / 240, 30 fps (7.5 fps for 0.25x).
- There are also `c11_main_FULL_*_3884_best_worst`, `c11_confirm_FULL_*`, and the source contact sheets `interval_sheet_*.png`.

## 1. Smaller model, same task
All runs: fixed 30,000 steps, cosine lr 2e-3 → 2e-5, L0 + 2·contour hinge, 3 seeds.
- Direct and distilled share initial weights per seed, steps and selection.
- Distillation uses the target plus an agreement-masked teacher term (λ = 1).

Medians, full span:

| model | main err (late third) | confirm err | main lag-1 F1 / change amount | main thin parts / gaps kept |
|---|---|---|---|---|
| C10 B+C, 10,320 (reference) | 0.0040 (0.0060) | 0.0134 | 0.911 / 0.81 | 0.928 / 0.816 |
| A: C9 1,304 distilled (historical) | 0.0253 (0.0380) | 0.0591 | 0.535 / 0.52 | 0.528 / 0.387 |
| B: C11 1,304 direct | 0.0175 (0.0249) | 0.0470 | 0.677 / 0.65 | 0.700 / 0.569 |
| C: C11 1,304 distilled | 0.0218 (0.0349) | – (dropped, budget) | 0.588 / 0.54 | 0.643 / 0.521 |
| C11 3,884 direct (one intermediate size) | **0.0062 (0.0129)** | 0.0287 | 0.876 / 0.79 | 0.894 / 0.802 |

**Facts:**
- **The C10 recipe helps at the same size:** with it, the 1,304 model beats the historical C9 1,304 model.
- **But 1,304 fails every quality gate:**
  - specks 17–50 where the original has 0;
  - the arm–body gap fills in.
- **Distillation was worse than direct training** at identical steps (0.0218 vs 0.0175 on main).
- **3,884 is close to C10 on main:**
  - Every C10-level gate passes except full-span error (0.0062 vs 0.0060) and late-third error (0.0129 vs 0.0090).
  - It passes C9-level.
- **3,884 fails C9-level on confirm** (0.0287 > 0.0266).

**Conclusion (pre-registered rule):** no smaller architecture keeps C10 quality on both tasks. 10,320 remains the
reference. 3,884 is a reduced-quality option for main only.

## 2. Longer autonomous playback (intervals fixed from the source before training)
- **L120 = 1760–1879:** one continuous scene, no cut. The dancer emerges from a line and swings the cane.
- **L240 = 1760–1999:** contains the 1880 wipe and the ~1911 shot change.
  - No cut-free 240-frame interval contains L120. Bad Apple is mostly morphs and transitions; see the sheets.

| run | full err | thirds err | lag-4 change F1 / amount per third | sustained gate |
|---|---|---|---|---|
| L120, 10,320, s0 | 0.0304 | 0.042 / 0.032 / 0.017 | 0.85/0.75, 0.81/0.62, 0.95/0.87 | **pass** |
| L120, 10,320, s1 | 0.0323 | 0.042 / 0.036 / 0.019 | 0.85/0.74, 0.76/0.56, 0.94/0.86 | **pass** |
| L120, 3,884, s0 | 0.0611 | 0.075 / 0.058 / 0.050 | 0.66/0.65, 0.54/0.37, 0.74/0.49 | fail (under-motion) |
| L120, 3,884, s1 | 0.0537 | 0.074 / 0.050 / 0.037 | 0.67/0.63, 0.64/0.45, 0.87/0.68 | fail |
| L240, 10,320, 1 seed, from scratch, 15k steps | 0.0656 | 0.069 / **0.100** / 0.028 | 0.58/0.45, 0.61/0.41, 0.87/0.68 | fail |

**Facts:**
- **120 frames work only for the large model.** Its motion is sustained, but error is 0.030, ~2× C10's confirm-task
  error on the same scene.
- **The small model under-moves** at 120 frames.
- **240 frames failed:**
  - The warm start from the L120 model failed numerically: its state blows up beyond its trained 120 frames, and
    training could not recover.
  - The single bounded repair, training from scratch, did not reach the gate. The middle third, with the scene
    changes, is the worst part.
- **The extension stops at 120.**

**Numerical stability:** the long runs needed lr 5e-4. At 2e-3 the first L120 run went NaN.

## 3. Storage (ncab-v1)
- **Format:** `ca/ncab.py` writes it; `player/nca_core.mjs` reads it (Node and browser).
- **What the file holds:** weights plus a packed first frame plus a small header, all deflated.
- **Byte breakdown:**

  | model | weights | first frame | header | file f32 | file f16 | nca-v1 JSON |
  |---|---|---|---|---|---|---|
  | 10,320 (C10) | 41,280 | 216 | 217 | 38,839 | 19,630 | 56,234 |
  | 3,884 | 15,536 | 216 | ~440 | 14,833 | 7,599 | – |
  | 1,304 | 5,216 | 216 | ~430 | 5,225 | 2,819 | – |

- **Float16 weight storage** (state and compute stay float32):
  - Error changes by ≤ 0.0002 in every model tested.
  - Binary output differs in 3–88 pixels per replay (out of 95–102k).
  - **float32 stays the default**; `--f16` is offered.
- **Lossless recording of the same frames** (1 bit/px, zlib-9):

  | interval | zlib-9 | lzma |
  |---|---|---|
  | main | 1,903 B | 1,580 |
  | confirm | 4,053 B | 3,060 |
  | L120 | 7,532 B | 5,500 |

  - Every model file is larger than this recording, and the models are approximate. **There is no compression advantage.**

## 4. Pipeline (`python -m ca.pipeline`, documented in `ca/pipeline.py`)
- **Commands:**
  - `probe`: decoder-validated metadata.
  - `prep`: decodes only the interval; grayscale by ITU-R 601 luma; box resize that keeps the aspect ratio; explicit
    threshold and fps. Cached by the sha256 of (source, settings).
  - `train`: preset `demo` (CPU, 1,304) or `quality` (GPU, 10,320); explicit flags override the preset; resumable.
  - `export`: ncab plus JSON, with a manifest.
  - `replay`: numpy vs Node.
  - `video`: comparison video.
  - `run`: the whole chain, with `status.json`.
  - `import`: an existing export, no training.
- **Tests:** `tests/test_pipeline.py` covers:
  - colour clip (aspect 96×64 → 32×21 / 48×32);
  - cache separation by width and threshold, and a cache hit with the same settings;
  - output fps;
  - Bad Apple prep equals the campaign target bit for bit;
  - a non-video file is rejected;
  - export-only replay in an empty dir, Python and Node, matching the known hash.
- **Colour demo:** `runs/pipe/example_color_ball_demo` (CPU, demo preset, 95 s of training, err 0.0017, numpy = Node).
- **Provenance:** `BadApple!!.mp4` comes from the cloned Bad-Apple-Starter repo. Only its sha256 is recorded; the
  upstream identity is not verified.

## 5. Local UI (`ui/`, 127.0.0.1 only)
**Features:**
- Upload or choose a video (decoder-checked; size, dimension and interval limits; hashed paths; no shell).
- Preview with interval, width, threshold and fps controls.
- Training: start on CPU, live progress from the real checkpoint files, cancel and resume own jobs, failure and
  "died" states.
- Load or download or re-import exports.
- Synchronised original | CA | error panels; play, pause, step, scrub, speed, integer zoom, continuous-value view.
- Model panel: architecture, update schematic, training curve, checkpoint list. No fake rule list.
- The CA panel is computed in the browser from the export alone (nca_core in a Web Worker), cached, and never fed
  original frames.

**Verified by the UI agent with a real browser (Playwright, headless):**
- The browser frame hash equals the Node verifier for `example_c10_main` (`f2401adb…`).
- Upload → preview → training → export download → re-import, and cancel/resume, all done via the API.

**GPU training from the UI** (gpuq submit and cancel) is implemented but **untested**.

**Design:** `ui/DESIGN.md` records what was taken from the larperCord site and how it was changed.
- The site is GPL-3.0, so no code was copied. Only OFL fonts are bundled, with their licences.

## 6. Verification (numpy vs independent Node replay)
- 38 export replay checks (evaluations plus the storage f32/f16 files, some models checked more than once).
- 5 distinct models show a binary mismatch; all others are 0 px:
  - L120 large s0: 1 px;
  - L120 small s1: 1 px;
  - L240: 2 px;
  - confirm 1,304 s2: 1 px;
  - main 3,884 s0 as f16: 1 px.
- The continuous difference **grows with time**: max |Δv| per quarter of L120 ≈ 6e-6 → 2e-4 (`node_drift.json`).
- **Agreement between implementations is not accuracy against the original**; the error columns above measure that.

## 7. Not done or limitations
- Segmented (playlist) playback: not implemented. The trained intervals are not adjacent.
- Boolean-circuit expression display: not built; no Boolean model was part of this campaign.
- Distillation on the confirm task: dropped for budget.
- No 240-frame success.
- **Queue resets:** the shared gpuq queue was reset twice by another session.
  - 3 queued jobs were lost and resubmitted.
  - Running jobs finished as orphans; their GPU time comes from their logs.

## 8. Cost

**GPU:**

| measure | value |
|---|---|
| device occupancy (time a GPU held ≥ 1 of my jobs) | ≈ 4.3 GPU-h |
| sum of per-job wall times | 6.16 h |

- The per-job sum is inflated by GPU sharing and is **0.16 h over the 6 h cap**.
- Ledger: `runs/c11/gpu_ledger.jsonl`.
- Included in both figures: two failed runs (NaN and warm start), 3 min of a dropped run, and the pilots.

**CPU:** 0.18 core-h for experiments plus 0.58 core-h for pipeline and UI development (`runs/c11/cpu_ledger.jsonl`).
- Not metered: the UI agent's own browser tests.

## Reproduce
```sh
D=$(experiments/freeze.sh)    # frozen code snapshot used for the runs: runs/c11/frozen_main.txt
gpuq -G 1 python $D/experiments/exp40_c11.py main c11/p1/main_12x32_D 12 32 30000 cos:2e-3:2e-5 runs/c11/cfg/p1_D3884.json
gpuq -G 1 python $D/experiments/exp40_c11.py L120 c11/L120/big16x64_r2 16 64 30000 cos:5e-4:5e-6 runs/c11/cfg/two_seeds.json
python experiments/exp44_export.py c11/p1/main_12x32_D             # -> runs/c11/exports/*.ncab
python experiments/exp41_c11_eval.py runs/c11/eval/x.json main runs/c11/exports/p1_main_12x32_D_D_s0.ncab --node
python experiments/exp43_storage.py runs/c11/eval/storage_main.json main <export>
python experiments/exp42_c11_videos.py NAME TARGET "Title=export" --diff 1
python -m ca.pipeline run my.mp4 runs/pipe/mine --start 0 --frames 24 --width 32 --preset demo
python tests/test_pipeline.py
.venv/bin/python ui/server.py
```
- `ui/make_examples.sh` rebuilds the examples' comparison frames after a fresh clone (the cache is git-ignored).
