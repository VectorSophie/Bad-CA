# Track G: external video, data sufficiency and phase dependence

Branch `research/generalization-shape`. Code: `experiments/gen_data.py`, `gen_train.py`, `gen_eval.py`,
`gen_ledger.py`; registry `ca/datasets.py` and `data/manifests/*.json`. Results: `runs/gen/summary.json` and the per-run
`eval.json` files. **Historical results were not rerun or changed.** Every comparison below uses one fixed
architecture (C = 16, Fh = 64, S = 4) unless stated.

## 1. The external video: *Fantasmagorie* (Émile Cohl, 1908)

| item | value |
|---|---|
| source | Wikimedia Commons, `File:Fantasmagorie_(1908).webm`; LicenseShortName "Public domain", Copyrighted False, AttributionRequired false (Commons API, 2026-11-13); credited to a PAL DVD edition |
| file | 33,761,285 bytes; Commons SHA-1 `7ba4199a…aa7`, matched; SHA-256 `262d0fb9…7b127`; kept unchanged in `data/source/` (git-ignored) |
| stream | VP9 profile 0, yuv420p, 720×540 (4:3), time base 1/1000, constant 25 fps (every Δpts = 0.04 s), 98.2 s, **2,455 decoded frames, 0 decode errors**, no audio |
| content | frames 0–741 are the DVD's modern title cards; the **animation is frames 742–2428 (1,687 frames, 67.5 s)**, fixed by a luma rule before any training |
| duplicates (source) | 13 % exact and 31 % near-identical consecutive frames, almost all in the title cards and black; inside the animation, grain makes every frame differ (0.7 % near-duplicates) |

**Preprocessing:** compared on contact sheets (`data/derived/fantasmagorie/preproc_compare.png`) and by numbers.

| method | foreground fraction | empty frames | change / frame | one-frame blips |
|---|---|---|---|---|
| Bad Apple recipe (box 48×36, ≥ 128) | 0.023 | 42 | 0.017 | 0.0056 |
| per-frame Otsu | 0.128 | 0 | 0.065 | 0.0197 |
| line extraction (192×144, minus 15×15 median, ≥ 24, area pooling) | 0.175 | 0 | 0.064 | 0.0175 |
| **lines + 3-frame temporal majority (selected)** | 0.172 | 0 | 0.039 | 0.0016 |

- **The Bad Apple recipe erases the 1–2 px drawn lines**, so it was rejected.
- **The temporal majority was selected** because 27–32 % of every variant's changes were one-frame blips (grain,
  flicker, gate weave). A model must not be rewarded for learning film noise. The unfiltered film is kept and tested
  (§4).
- **Limitation:** gate weave (whole-frame jitter) was not stabilised.

## 2. Redundancy (48×36 binary)

| | Bad Apple | Fantasmagorie |
|---|---|---|
| decoded frames used | 6,572 at 30 fps | 1,687 at 25 fps |
| identical / near-identical (≤ 0.2 % px) consecutive frames | 4.2 % / 9.8 % | 0.8 % / 7.2 % |
| distinct frames | 6,280 | 1,674 |
| mean (median) change per frame | 3.0 % (1.7 %) | 3.9 % (2.8 %); 6.4 % before the temporal majority |
| foreground | 46 % | 17 % (lines) |
| atlas (Path V rules, unchanged) | 410 windows; multi_motion 97, deformation 83, cut_wipe 80, … | 105 windows; **appear_disappear 36**, major_transform 29, deformation 26, cut_wipe 8 |

- Decoded frames are not independent examples. Both videos have few exact duplicates, but neighbouring frames differ
  by only 2–4 % of pixels.
- Fantasmagorie's motion is dominated by lines appearing and being redrawn, not by moving silhouettes.

## 3. C13 reproduction check (G1)
- The featured export's file SHA-256 `facb43d9…` equals `runs/featured/manifest.json`.
- An independent Node replay gives a whole-song mean error of **0.0152**, matching the record.
- The replay's output hash `277fff0f…` is now recorded; no previous output hash existed.
- No retraining.
- **Reusable as-is for another video:** `ca/phase_train.chunked` (chunked BPTT, numpy-carried state, noise), the
  ncab-v2 export, and the Python and Node players.
- **Video-specific:** preprocessing and the phase count.

## 4. Is Bad Apple data-limited? (G4: one shared rule, many windows)
**Protocol** (`gen_train.py win`):
- one rule (10,320 parameters), no phases;
- 18-frame windows: a 2-frame observation prefix, then 15 generated frames;
- 6,000 steps × batch 16, the same for every budget;
- 64-frame blocks: train, train, val, train, train, test; 16-frame guard gaps;
- test is the same for every budget;
- "novel scene" = no training window in the same cut-delimited scene.

Test results (generated-frame error / hold / share of windows beating hold / lag-1 change F1 / one-step error vs hold):

| run | train windows (distinct frames) | Bad Apple (68 test windows) | Fantasmagorie (16 test windows) |
|---|---|---|---|
| spread 10 % | 27 / 7 | 0.174 / 0.155 / 0.32 / 0.05 / 0.0346 vs 0.0363 | 0.137 / 0.110 / 0.06 / 0.04 |
| spread 25 % | 68 / 18 | 0.161 / 0.32 / 0.05 | 0.140 / 0.25 / 0.05 |
| spread 50 % | 137 / 36 | 0.166 / 0.35 / 0.05 | 0.128 / 0.12 / 0.04 |
| **spread 100 %** | 274 (3,087) / 72 (810) | **0.151 / 0.155 / 0.56 / 0.05 / 0.0322 vs 0.0363** | 0.120 / 0.110 / 0.31 / 0.04 |
| spread 100 %, seed 1 | | 0.148 / 0.54 | 0.125 / 0.25 |
| contiguous 10 / 25 / 50 % | fewer scenes | 0.191 / 0.171 / 0.153 | 0.163 / 0.160 / 0.146 |
| novel scenes only (contiguous 10 → 50 %) | | 0.193 vs hold 0.154 → 0.173 vs 0.178 | none (too short) |
| half frame rate (stride 2) | | 0.225 / 0.201 (worse) | 0.164 / 0.119 (worse) |
| **96×72** (100 %) | | **0.151 / 0.156 / 0.60 / 0.07 / 0.0292 vs 0.0370** | **0.089 / 0.093 / 0.62 / 0.03** |
| unfiltered film (no temporal majority) | | — | 0.128 / 0.118 / 0.44 |
| zero-shot: rule from the other video | | 0.338 / 0.155 (Fantasmagorie rule) | 0.126 / 0.110 (Bad Apple rule) |

**Observed:**
- **More data helps a little and does not change the picture.**
  - Bad Apple goes from worse than hold (0.17–0.19) to hold level (0.151 vs 0.155) at 100 %.
  - Motion F1 stays at 0.04–0.05 everywhere; one-step error falls 11 % below hold.
  - Fantasmagorie (4× fewer windows) stays worse than hold at every budget at 48×36.
- **More scenes is not clearly better than more frames from fewer scenes.** Contiguous 50 % (0.153) is better than
  spread 50 % (0.166), and the seed difference is 0.003. Unexplained.
- **Resolution helps more than data.** At 96×72 Fantasmagorie beats hold in 62 % of windows (worse than hold at
  48×36). Bad Apple's one-step error is 21 % below hold (11 % at 48×36), and F1 rises 0.05 → 0.07.
- **Halving the frame rate hurts both videos.** Each step must then span twice the motion.
- **Zero-shot transfer fails both ways.**

**Interpretation (not proven):**
- Bad Apple is mildly data-limited for one-step prediction: gains are still present at 100 %.
- There is no sign that more frames of this kind would make a shared local rule generate motion autonomously.
- Every result, here and in Readable Rules, Path D and Stage 0, points to the next frame being largely undetermined by
  a local view of the past (explanation G), with spatial resolution as a secondary bottleneck (explanation D).

## 5. How much of C13's success is the phases? (G3)
**Setup:** 240-frame clips (Bad Apple 1760–1999, which contains a wipe and a cut; Fantasmagorie 160–399), uniform
phases, 4,000 steps per chunk, σ = 0.02. Reconstruction is continuous from frame 0, as in C13.

| clip | K | parameters | reconstruction err (hold) | lag-1 F1 | 32 unseen frames: err vs hold | fresh start (hidden reset) vs carried |
|---|---|---|---|---|---|---|
| Bad Apple | 1 | 10,320 | 0.065 (0.401) | 0.11 | **0.082 vs 0.113** | 0.162 vs 0.053 |
| Bad Apple | 4 | 41,280 | 0.032 | 0.34 | 0.181 vs 0.113 | 0.279 vs 0.036 |
| Bad Apple | 16 | 165,120 | **0.011** | **0.72** | 0.435 vs 0.113 | 0.242 vs 0.011 |
| Bad Apple | 1, Fh 256 (parameter-matched to K = 4) | 41,232 | 0.085 (lr 2e-4) | 0.07 | 0.097 vs 0.113 | 0.150 vs 0.059 |
| Fantasmagorie | 1 | 10,320 | 0.116 (0.256) | 0.06 | 0.208 vs 0.179 | 0.079 vs 0.045 |
| Fantasmagorie | 4 | 41,280 | 0.077 | 0.12 | 0.247 vs 0.179 | 0.248 vs 0.082 |
| Fantasmagorie | 1, Fh 256 (parameter-matched to K = 4) | 41,232 | 0.103 (lr 2e-4) | 0.07 | 0.212 vs 0.179 | 0.062 vs 0.045 |
| Fantasmagorie | 16 | 165,120 | **0.032** | **0.50** | 0.389 vs 0.179 | 0.226 vs 0.032 |

**Observed:**
- **Phases buy fit, not parameters alone.** One rule with the parameters of K = 4 fits *worse* than K = 4 on both
  videos: Bad Apple 0.085 vs 0.032, Fantasmagorie 0.103 vs 0.077. This agrees with C12 (one large rule 0.045–0.047 vs 4–6 phases 0.019–0.033). Caveat: it needed a
  lower learning rate (lr 5e-4 diverged for both videos; those runs are kept as `*_DIVERGED`).
- **More phases fit better and extrapolate worse.** The last phase rule of K = 16 is 3–4× worse than hold on the next
  32 frames. Only Bad Apple's single rule beats hold there.
- **The fits depend on carried hidden state.** Restarting a phase from the true frame with empty hidden channels is
  3–20× worse than the continuous run.
- **These are fitted trajectories, not local laws of motion.**
- **Fantasmagorie is harder to fit than Bad Apple** at equal K (0.032 vs 0.011 at K = 16), consistent with its line
  art and redrawing.
- **96×72 at the same steps fits worse** (Bad Apple 0.021, Fantasmagorie 0.055 at K = 16). Higher resolution needs
  more training compute; it did not make fitting easier.

## 6. Full film (G2): the C13 recipe on all 1,687 animation frames

**Setup:** automatic phases at C13's density (K = 28, ~60 frames per phase), 7 chunks, 8,000 steps per chunk,
σ = 0.02, numpy-carried state, learned low-res initial hidden.

**Deviation from C13:**
- Chunk 1 diverged at C13's lr 5e-4: non-finite losses from step ~400, best checkpoint = the initial weights. It is
  recorded in `runs/gen/fit/fa_full_c1_lr5e-4_DIVERGED/NOTE.txt`.
- The job was stopped. Chunk 0 (trained at 5e-4) was kept, and chunks 1–6 were trained at lr 2e-4.

**Reconstruction** (continuous replay of the export from the first frame, no frames injected; `runs/gen/fit/fa_full/eval.json`):

| | Fantasmagorie (this run) | Bad Apple C13 featured (for scale) |
|---|---|---|
| frames | 1,687 | 6,572 |
| phases | 28 | 110 |
| mean error (hold of frame 0) | **0.066** (0.218) | 0.0152 |
| lag-1 / lag-4 change F1 | 0.16 / 0.39 | lag-4 ≥ 0.5 in ≥ 90 % of moving windows (C13 criterion M) |
| IoU / boundary F1 | 0.68 / 0.93 | — |
| frames with error > 0.09 | 489 of 1,686 (29 %) | 1.5 % of frames in failing 16-frame windows |
| 240-frame windows with error ≤ 0.06 | 31 % (max 0.116) | ≥ 90 % (max 0.041) |
| one-frame blips (prediction / target) | 0.0002 / 0.0016: no learned flicker | — |
| file / frames zlib | 1.08 MB / 97 kB (11×) | 4.2 MB / 231 kB (18×) |

**By atlas tag** (error / true change per frame):

| tag | windows | error | true change |
|---|---|---|---|
| translation | 1 | 0.021 | — |
| major_transform | 29 | 0.056 | 0.035 |
| appear_disappear | 36 | 0.061 | 0.037 |
| deformation | 26 | 0.068 | 0.032 |
| multi_motion | 4 | 0.071 | — |
| growth | 1 | 0.092 | — |
| **cut_wipe** | 8 | **0.126** | 0.077 |

**Per chunk:**

| chunk (frames) | error | note |
|---|---|---|
| 1–217 | 0.030 | |
| 217–632 | 0.117 | dense hat and plume drawings |
| 632–871 | 0.059 | |
| 871–1089 | 0.048 | |
| 1089–1390 | 0.074 | |
| 1390–1624 | 0.030 | |
| 1624–1687 | 0.052 | |

The worst 16-frame window starts at frame 616 (0.207).

**Replay consistency:** the Node player and the numpy reference replay the export independently.
- They differ in 109 of 2,915,136 pixels (3.7e-5 overall, at most 0.23 % in any frame, first difference at frame 221).
- Threshold noise of the same kind as C13's (92 px over the song), within C13's criterion S.
- Node output SHA-256: `dc114ffa…`.

**Fresh start per phase** (true frame, hidden reset): 0.134 vs 0.053 carried, about hold (0.142). The phase rules do
not work without the carried hidden state.

**Verdict:**
- **The architecture is portable.** Retrained from scratch, it replays the whole film from its first frame, with
  recognisable drawings, no learned flicker and boundary F1 0.93 (`assets/release/fantasmagorie_nca.gif`).
- **Quality is far below Bad Apple** and would not pass the C13 acceptance criteria: 4.3× the error, and 31 % vs
  ≥ 90 % of 240-frame windows under 0.06.
- **Failures concentrate** at cuts and wipes and in dense line drawings.
- This is one model per video; nothing is shared between the two videos.
