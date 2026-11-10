# CA-universal

CA-universal is a local research tool that tries to turn a video into a cellular automaton: a starting grid plus
local update rules that, run on their own from the first frame, replay the video. It preprocesses a video to a small
binary grid, trains a CA on it, exports the result to a self-contained file, and replays that file in Python, in
Node and in a local browser UI, without ever showing the CA the original frames after the first one. Bad Apple!! is
the test video. The pipeline accepts other videos too, including colour footage.

> **Status (2026-11-11).** The pipeline, the export formats, the two independent players and the local UI exist and
> are tested. Quality is verified only on the Bad Apple intervals listed under "Results so far". The longest
> autonomous replay that passes every evaluation gate is **frames 0–1467 (48.9 s, including the black opening)**,
> using 26 externally scheduled rules with the state carried across rule changes. A whole-song run (110 rules)
> **failed** as a replay after about 1,080 frames. The models that work are continuous-state Neural CAs with
> 10,320 float32 parameters per rule. They are not small rule sets, and every model file is larger than a lossless
> recording of the same frames. Explicit, human-readable rules exist and run exactly, but on real motion they are far
> worse than the Neural CAs (experimental). The final goal, the whole video replayed autonomously with automatically
> optimised phases and readable rules, has **not** been reached. Details: `docs/CHECKPOINT_12.md`.

## Try it

Every command below was run on this checkout on 2026-11-10 (CPU only, no training). The Python environment is the
local `.venv` (Python 3.12, numpy, scipy, PyAV, Pillow, JAX). There is no requirements file yet. Node 24 runs the
players.

```sh
# local UI (127.0.0.1 only; default port 8765). Stop with Ctrl-C.
.venv/bin/python ui/server.py --port 8790
#   -> "CA-universal replay bench on http://127.0.0.1:8790/"
#   open http://127.0.0.1:8790/?run=example_c10_main

# video -> CA pipeline CLI
.venv/bin/python -m ca.pipeline --help
#   -> subcommands: probe, prep, train, export, replay, video, run, import

# pipeline tests (about 9 s; one needs the Bad Apple file, see "Data")
.venv/bin/python tests/test_pipeline.py
#   -> ok test_color_and_cache / ok test_bad_apple_path / ok test_rejects / ok test_export_only_replay

# replay an export in Node, from the file alone; prints a SHA-256 over the binary frames
node player/nca_replay.mjs runs/pipe/example_c10_main/model.ncab
#   -> {"sha256":"f2401adb7b9f…","T":56,"H":36,"W":48,"C":16,"S":4,"weight_dtype":"f32"}
node player/rule_replay.mjs runs/c12/samples/life_then_grow.rule.json
#   -> {"sha256":"a97f536bf2d4…","T":30,"H":24,"W":24,"k":0,"S":1}

# UI inspector check: browser-side rollout and cell traces vs the reference step (about 45 s)
node ui/test_inspector.mjs
#   -> PASS lines for both samples and example_c10_main, then "ALL PASS"
```

Train a small model on your own clip (CPU, `demo` preset, 1,304 parameters; minutes):

```sh
.venv/bin/python -m ca.pipeline run my.mp4 runs/pipe/mine --start 0 --frames 24 --width 32 --preset demo
```

The `quality` preset (10,320 parameters) needs a GPU. On this machine GPU jobs go through the shared `gpuq` queue.

## What it actually does

| stage | what exists | where |
|---|---|---|
| Preprocess | decode only the requested interval (PyAV), luma (ITU-R 601), box resize keeping aspect ratio, explicit threshold and fps; cached by sha256 of source + settings | `ca/pipeline.py` `prep` |
| Train (single rule) | continuous-state Neural CA, JAX, resumable, contour-weighted loss | `ca/pipeline.py` `train`, `ca/nca_train.py` |
| Train (phases) | K rules on an external schedule, joint BPTT across phase boundaries, or greedy/chunked | `ca/phase_train.py`, `experiments/exp45_c12.py` |
| Segment | automatic phase boundaries from the video (MDL dynamic programming, cut and wipe detection) | `ca/segment.py` |
| Explicit rules | decision lists over integer neighbourhood features, ≤ 3 hidden bits; learner and exact executor | `ca/rules.py`, `player/rule_core.mjs` |
| Export | `nca-v1`, `ncab-v1`, `ncab-v2`, `dlrule-v1` (see `docs/format.md`) | `ca/nca.py`, `ca/ncab.py`, `ca/phase_nca.py`, `ca/rules.py` |
| Replay | numpy reference and an independent Node/browser player that read only the export | `player/` |
| Evaluate | full-interval autonomous replay vs the preprocessed binary original; shape, motion and transition metrics; gates | `ca/diag10.py`, `experiments/exp46_eval12.py` |
| Inspect | local UI: synchronised original / CA / error panels, phase timeline, rule list and diffs, per-cell inspector, rule editing | `ui/` |

Replay is autonomous in every reported result: the first frame (plus, where stated, stored initial hidden state)
goes in, and nothing from the target is read afterwards. There are no resets, no frame injection, and no coordinate
or frame-number inputs to the cells. A phase schedule, where used, is part of the export and is counted in its size.

## Results so far

All errors are the fraction of wrong pixels over the full interval of an autonomous replay from the export, at
48×36, against the binary original. Sources: `docs/CHECKPOINT_10.md`, `docs/CHECKPOINT_11.md`,
`docs/CAMPAIGN_12.md`, `docs/CHECKPOINT_12.md`.

| interval (Bad Apple frames) | model | error | verdict |
|---|---|---|---|
| 4500–4555 (56 frames) | 1 rule, 10,320 params (C10) | 0.0040 (median of 3 seeds) | passes |
| 1800–1859 (60 frames) | 1 rule, 10,320 params (C10) | 0.0134 (median) | passes |
| 4500–4555 | 1 rule, 3,884 / 1,304 params (C11) | 0.0062 / 0.0175 (median) | fail the C11 gates |
| 1760–1879 (120 frames) | 1 rule, 10,320 params (C11) | 0.0304 / 0.0323 (2 seeds) | passes the sustained-motion gate |
| 1760–1999 (240 frames, a wipe and a shot change) | 1 rule, 10,320 params | 0.0656 (C11), 0.072 / 0.065 (C12 pilot) | fails |
| 1760–1999 | 4 phases, uniform or automatic boundaries (C12 pilot) | 0.027 / 0.028 (uniform), 0.028 / 0.033 (auto) | uniform passes in both seeds, auto in one |
| 4500–4555 | explicit dlrule-v1, 14 phase programs, 117 rules (C12 pilot) | 0.0995 | experimental; far from the NCA |
| 1760–1999 | 1 rule with the same parameter count (41,232; C12) | 0.047 / 0.045 | fails; 4 phases beat it |
| 1760–1999 | 6 phases, uniform / automatic (C12) | 0.019 / 0.021, 0.021 / 0.022 | pass |
| 0–1467 (48.9 s, black opening included) | 26 phases, chunked-joint, training-only state noise, 1 seed (C12) | 0.0104 | **passes; longest** |
| 0–2115 | 37 phases, same recipe, 1 seed (C12) | 0.0157 | fails one window (fast wipe at 1507–1522) |
| 0–6571 (whole song) | 110 phases, chunked-joint without noise, 1 seed (C12) | 0.339 | **fails** after ~1,080 frames (not robust to float differences) |

What this does **not** show yet:
- **That phases help everywhere.** They beat a same-size single rule on 1760–1999 (with a wipe and a shot change),
  but on the cut-free 1760–1879 two phases gave no clear gain. Two seeds per arm.
- **That automatic segmentation helps.** At K = 2, 4 and 6 it was never better than equal-length phases.
- **The whole song.** Only one seed. The whole-song attempt failed; the noise-trained recipe that fixes the
  divergence has been run to frame 2115 only.
- **Compression.** Every model file is larger than the same frames stored losslessly (1 bit/px + zlib-9).
- **Generality.** Other videos run through the pipeline (a synthetic colour-ball demo trains to error 0.0017),
  but quality has only been studied on the Bad Apple intervals above.

## Replay types

1. **Fixed rule:** one rule for the whole interval.
2. **External schedule:** a time-varying CA. The export stores which rule owns each frame. State is carried across
   phase boundaries unchanged. This is the main path for long intervals.
3. **Internal-state phase switching**, where the cells' own state selects the rule: not implemented.
4. **Clip concatenation with resets:** diagnostic only, never counted as autonomous playback.

## Limits

- Resolution: every result is 48×36 binary. Preprocessing loss (480×360 → 48×36 box → threshold 128) is
  separate from CA error (`experiments/exp48_preproc_video.py` renders it).
- A parameter count is not a rule count. The Neural CA weights are not readable rules, and the UI never shows them
  as rules.
- A black first frame with zero hidden state cannot form a picture under a shift-invariant rule with a zero
  boundary. The song run therefore stores a learned low-resolution initial hidden array, counted in the file.
- Python and Node replays agree bit for bit on most exports. In C11, 5 of the checked models differed by 1–2
  pixels, and the continuous difference grows with replay length. That agreement is reported separately from
  accuracy.
- The `ca.pipeline` CLI trains, exports and replays single-rule (v1) models only. Phased v2 models come from
  `experiments/exp45_c12.py`. The UI lists and plays both.
- Local use only. The UI binds to 127.0.0.1 and checks Host and a custom header. Upload hardening for any public
  deployment has not been done.

## Experimental

- **Explicit rules (`dlrule-v1`).** Learned directly from frames, or extracted from a trained NCA. Exact on the
  positive controls (Conway's Life; a hidden-bit rule). On real motion they collapse to "hold" or flicker unless
  split into many short phases. See `docs/rule-models.md`.
- **Low-rank per-phase deltas** (`mode=delta`): 0.049 / 0.054 on 1760–1999, worse than independent phases.
- **Greedy phase training:** worse than joint training on the wipe and shot-change phases.
- **Rule editing in the UI:** edits replay from frame 0 by default. Continuing from the current state is labelled
  as a hybrid.
- **GPU training from the UI** (via `gpuq`): implemented, exercised once with a short check job.

## Documentation

| file | contents |
|---|---|
| `docs/architecture.md` | modules, data flow, players, UI |
| `docs/rule-models.md` | Neural CA, phased NCA and explicit rule models |
| `docs/format.md` | every export format, field by field, and the byte accounting |
| `docs/evaluation.md` | how replays are scored, metrics, gates, baselines |
| `docs/roadmap.md` | next steps, grounded in documented failures |
| `docs/research/README.md` | index of every campaign plan and checkpoint |
| `ui/README.md`, `ui/DESIGN.md` | UI endpoints and limits; design sources |

## Data and licence

- **Bad Apple!! video and music are not part of this repository and must not be added to it.** The experiments use
  `refs/Bad-Apple-Starter/BadApple!!.mp4` from a local clone of `C-mykk/Bad-Apple-Starter` (commit `7b04ba8e`;
  that repository has no licence file). Only the file's sha256 is recorded
  (`d8f21232313506da4f8d631ca289793bcf2f0a33440d68ccb02f4ac52e963cd0`; h264 480×360, 6,572 frames at 30 fps). Its
  upstream identity is not verified. `refs/` is git-ignored. To reproduce, supply your own copy at that path.
- Committed run artefacts (exports, small frame grids, videos under `runs/`) are derived from that file at 48×36
  for research comparison.
- `tests/data/*.mp4` are synthetic clips generated by this project.
- The UI bundles OFL-licensed fonts with their licence files (`ui/static/fonts/`). No code was copied from the
  GPL-3.0 design reference (`ui/DESIGN.md`).
- This repository has no licence file yet.

Documentation layout follows the README/doc structure of VectorSophie/hgit.
