# Checkpoint 13: whole-song demo

**Plan, fixed criteria, run log:** [CAMPAIGN_13.md](CAMPAIGN_13.md).
**Machine-readable results:**
- `runs/c13/eval/accept.json` (C13 criteria);
- `runs/c13/eval/diag_stability.json`;
- `runs/c13/eval/wipe_pilot.json`.

**Status of this page:**
- Written when the seed-0 whole-song run finished.
- The seed-1 run and the transition-weighted run (`s0-tw2`) were still training; §5 is filled in when they finish.

## 1. Whole song: complete for seed 0
**Export:** `runs/featured/whole_song_s0.ncab`, a copy of `runs/c13/song_noise_s0/export_s0.ncab`.
- sha256 `facb43d9911399ee…` (full value in `runs/featured/manifest.json`).
- 110 phases × 10,320 float32 params, plus the first frame, plus a learned low-res initial hidden array (1,620 values).
- 4,228,712 bytes.

**Replay:** frames 0–6571 from the file alone. Hidden state is carried across all 109 boundaries; there is no reset and
no target injection.

| check | result |
|---|---|
| numpy replay vs original | error **0.0152**; worst 240-frame window 0.041; 100 % of windows ≤ 0.06 |
| frames inside failing 16-frame windows (> 0.09) | **1.5 %**: 1499–1530 (peak 0.159), 2849–2874 (0.107), 3641–3682 (0.103) |
| motion (lag-4 change F1 ≥ 0.5, mass 0.5–1.5) | 100 % of windows where the original moves |
| thin parts / thin gaps kept | 0.83 / 0.78 |
| small specks / holes vs original | 2.0× / 2.7× |
| spurious one-frame flips | 0.39 per frame |
| phase-boundary error ≤ 2 × global | pass |
| fades (192 uniform original frames) | 100 % reproduced |
| max \|state\| over the song | ≤ 44 (no growth) |
| Node vs numpy | 92 binary pixels differ in 6,572 frames (0.0008 %), max per frame 0.12 %; max \|Δv\| 1.6e-3 with no trend (`max_dv_by_tenth`) |
| browser (headless Chromium, UI engine) vs Node | **bit-identical** SHA-256 `277fff0f…` over all binary frames (6,572 frames, 110 phases, 314 s) |
| **C13 criteria** (§3 of the plan, fixed before any whole-song result) | **all pass** |
| C12 gates (kept side by side, unchanged) | fail: the worst 16-frame window is > 0.09 |

**Training notes.** Chunked-joint training as in C12: noise σ = 0.02 and 8,000 steps per chunk. New in C13: chunk states
were carried with the numpy reference arithmetic, so each chunk's **true replay error was known during training**. All
28 chunks: replay error = training error.

## 2. What the diagnosis showed (before the run)
- **The C12 whole-song failure was numerical.** In GPU (JAX) vs CPU (numpy) replays of the same no-noise export, the
  first binary difference is at frame 340, differences pass 1 % at frame 980, and the state reaches ~1e17.
- **The noise-trained model is stable:** max |Δv| ≤ 3e-5 over 0–2115 and a state of at most |31|.
- **The fast wipe at 1499–1530 is a separate, quality problem:** there is no GPU/CPU difference there.

## 3. Wipe pilot (chunk 6 only, one seed)
| variant | wipe error | worst 16-frame window |
|---|---|---|
| C12 baseline | 0.100 | 0.159 |
| transition-weighted loss (×2) | 0.078 | 0.120 |
| boundary moved to the wipe onset | 0.094 | 0.158 |
| both | 0.074 | 0.121 |

- The weighting helped; moving the boundary did not.
- The weighted recipe is run on the whole song as `s0-tw2` (§5).

## 4. Product, documentation, publication
- **UI:** three views, Demo / Rule lab / How it was built.
  - The demo plays the featured model from `runs/featured/manifest.json`, with a sha check.
  - The rule panel is generated from the implementation: blue structure, red phase parameters, change summary per phase.
  - A cell inspector is checked bit-exactly against the engine.
  - Internal terms are kept out of the default views (scanned automatically); provenance is in "Model details".
- **Browser engine speed:** ~21 frames/s, so playback waits at the computed edge (shown).
- **Pipeline CLI:** `ncab-v2` support (`info`, `replay`, `import`, phased `train/run --phases`). Tests: 6/6 pass.
- **Docs:**
  - README (hgit-style header, logo, demo, how it works, results, why);
  - [How we got here](research/journey.md) (the decision flow);
  - [timeline](research/timeline.md);
  - figures drawn by code (`tools/make_figures.py`);
  - logo from a real source frame (`assets/logo/README.md`).
- **Published:** https://github.com/VectorSophie/Bad-CA (`master`). First push `0c891d0`, merged with the remote GPL-3.0
  LICENSE commit, no force push.

## 5. Seed and recipe checks
*(filled in when the runs finish)*
