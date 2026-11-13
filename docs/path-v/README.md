# Path V: velocity-carrying CA and a failure-aware evaluation

Branch `research/velocity-ca-v0` (from `context-s0`, so the Stage 0 code and results are reused unchanged). CPU only,
GPU 0. Everything here is a **hand-written** rule driven by **oracle** controls unless stated otherwise.

## Verdict (2026-11-13)

| question | verdict |
|---|---|
| oracle velocity feasibility | **PARTIAL.** With oracle regional codes the velocity CA follows real motion far better than hold (macro error 0.077 vs 0.220 at 3×3, events every 4). But at a matched control budget it is only on par with a whole-image block shift, and its edge comes from the growth term, not from carrying velocity |
| motion persistence | **NO-GO overall, PARTIAL for translation.** One event fitted on 8 frames beats hold on the 23 unseen frames in 75–83 % of translation clips, but in 0–33 % of every other class |
| "velocity carried by cells" vs a velocity field | **no measurable benefit.** Lagrangian minus Eulerian, paired over 72 clips: \|Δerr\| ≤ 0.005 |
| generalisation within successful classes | not tested for learning (nothing learned); the oracle results are consistent across two starts of the same scene (r = 0.84) |
| learning (V2) | **skipped**: the gate was "motion that shift or keyframe baselines cannot explain", and it was not met |
| Boolean circuit (V3) | **skipped**: no working rule worth converting; the rule is given as readable pseudocode (`experiments/pv_vca.py`) |
| suitability for BadCA | a clear, inspectable *oracle-controlled animation mechanism* (close to a codec), **not** an emergent rule; useful mainly as an honest teaching demo and as a strong baseline |

## 1. Difficulty atlas (`experiments/pv_atlas.py`, `runs/path_v/atlas.json`, `atlas_examples.png`)
- **Windows:** 410 windows of 16 frames, `[16 i, 16 i + 16)`.
- **Descriptive metrics:**
  - change mass, cut / wipe / fade flags;
  - foreground area change and monotonicity, centroid displacement;
  - component births and deaths, IoU first → last;
  - the share of change explained by the best global shift (g), the best 3×3 regional shift (r) and dilation or erosion
    (m), each the best over lags 1, 2, 4.
- **Tags** are measured, not semantic. Rules v2 were frozen in commit `323549b` before any method was scored.
  - v1 tagged only 1 window as translation and filed fast rigid moves under major_transform; it is kept as
    `atlas_v1_rejected*`.

| tag | windows | frames | Neural CA error (featured, continuous) | Neural CA lag-1 F1 |
|---|---|---|---|---|
| cut_wipe | 80 | 1,280 | **0.030** (p90 0.057) | 0.58 |
| multi_motion | 97 | 1,552 | 0.015 | 0.59 |
| deformation | 83 | 1,328 | 0.010 | 0.48 |
| growth | 42 | 672 | 0.013 | 0.52 |
| major_transform | 42 | 672 | 0.014 | 0.56 |
| static | 34 | 544 | 0.001 | 0.69 |
| appear_disappear | 16 | 256 | 0.014 | 0.59 |
| translation | 16 | 256 | 0.008 | 0.75 |

- **Weighting:** frame-weighted Neural CA error 0.0152, macro over tags 0.0131.
- **The Neural CA's ten worst windows** are 9 cut/wipe windows and 1 growth window, headed by 1504, 3664, 3648 and
  2848.
- **Caveat:** the tags are heuristics. Visual checks showed sensible examples, but e.g. the "multi_motion" tag also
  catches some dancing and sparse-dot windows.

## 2. The velocity CA (`experiments/pv_vca.py`)
**State per cell (19 bits):**
- `fg` (1 bit);
- velocity `vy, vx` and growth rate `g`, each in [−4, 4] in quarter-px (or quarter-layer) per frame;
- sub-pixel phases `py, px, pg` in [0, 4).

The background colour is fixed per run from the start frame, and background cells carry nothing.

**One frame, radius 1:**

```
phase:      t = p + v;  step = floor(t / 4) in {-1, 0, 1};  p = t - 4*step     # displacement floor((2 + n v) / 4)
transport:  each cell takes the first of: itself if it stays, else a neighbour whose step lands here
            (order N, S, W, E, diagonals); nothing lands -> background (hole); extra movers are absorbed
growth:     t = pg + g; o = floor(t / 4); o = +1 -> background 4-neighbours become fg (copy state);
            o = -1 -> the cell dies if it touches background
control:    at an event every fg cell in region r takes code (vy, vx, g)_r, phases reset to 2 (lagrangian);
            the eulerian baseline re-reads the code of the region a cell is in every frame
```

`tests/test_pv_vca.py` checks the following synthetic behaviour:
- exact translation at every speed;
- persistence without events;
- opposing regions that split an object;
- the grid boundary;
- mid-trajectory events;
- deterministic collision loss;
- growth and shrink equal to 4-neighbour dilation and erosion;
- lagrangian vs eulerian across a region boundary;
- both polarities;
- the block-shift baseline.

**Documented limits:**
- objects cannot enter from outside the frame;
- collisions lose mass;
- speed is ≤ 1 px/frame;
- +v and −v step on alternating frames (a rounding convention).

## 3. V0: oracle motion fields (`experiments/pv_v0.py`, manifest `runs/path_v/v0_manifest.json`)

**Clips** (frozen before fitting, seed 20261114):
- per atlas tag (translation, growth, deformation, multi_motion, major_transform, cut_wipe): 2 dev and 6 test clips
  of 32 frames;
- each test clip also gets a second start 8 frames later in the same scene;
- 72 test clips in total, scored once. The dev clips were used only for debugging.

**Control** is an ORACLE:
- sequential greedy, coordinate descent per region, minimising the pixel error of the segment, with state carried;
- the runtime gets only the start frame plus the schedule, and replaying the schedule reproduces the fit bit for bit
  (`tests/test_pv_campaign.py`).

**Settings:**
- grids 1×1, 2×2, 3×3;
- events every 4, 8, 16 frames, or once.

**Methods and control cost per region-event:**

| method | description | bits |
|---|---|---|
| vcaL | full code | 9.51 |
| vcaL_v | velocity only | 6.34 |
| vcaE | Eulerian field | 9.51 |
| block | integer whole-image shift | 4.64 |
| codebook | Stage 0 K = 8, global | 3 |
| keyframe | signed-distance interpolation, spacing 8 / 16 / 31; bits = zlib size of the keyframes | — |
| hold | | 0 |

### Test results, macro over the 6 tags
F1 is exact-pixel lag-1 change F1.

| config | bits / 31 frames | err | F1 | IoU | boundary F1 |
|---|---|---|---|---|---|
| hold | 0 | 0.220 | 0.000 | 0.447 | 0.507 |
| codebook K = 8, every 4 | 24 | 0.173 | 0.042 | 0.511 | 0.566 |
| block 3×3, every 4 | 334 | 0.089 | 0.147 | 0.663 | 0.729 |
| block 3×3, every 8 | 167 | 0.101 | 0.111 | 0.623 | 0.667 |
| vcaL 2×2, every 4 | 304 | 0.090 | 0.139 | 0.654 | 0.693 |
| vcaL 3×3, every 8 | 342 | 0.086 | 0.127 | 0.664 | 0.706 |
| vcaL 3×3, every 4 | 685 | **0.077** | **0.173** | 0.684 | 0.732 |
| vcaL 3×3, once | 86 | 0.125 | 0.057 | 0.577 | 0.606 |
| vcaL_v 3×3, every 4 (no growth) | 456 | 0.140 | 0.108 | 0.556 | 0.631 |
| keyframe 16 | 1,433 | 0.094 | 0.067 | 0.675 | 0.733 |
| keyframe 8 | 2,844 | 0.050 | 0.173 | 0.792 | 0.853 |
| Neural CA (featured, continuous, 4.2 MB; calibration only) | — | 0.021 | 0.519 | 0.889 | 0.942 |

**Per tag**, vcaL 3×3 every 4 (err / F1):

| tag | err | F1 |
|---|---|---|
| translation | 0.073 | **0.28** |
| multi_motion | 0.080 | 0.21 |
| growth | 0.068 | 0.18 |
| major_transform | 0.069 | 0.13 |
| deformation | 0.047 | 0.10 |
| cut_wipe | 0.124 | 0.14 |

**Findings:**
- **Matched budget.** For every vcaL configuration, the best block / keyframe / codebook configuration with at most
  the same bits:
  - vcaL wins on error by 0.002–0.025 at ≥ 150 bits, but block shift often has the higher F1 (e.g. vcaL 3×3/8,
    342 bits: 0.086 / F1 0.127 vs block 3×3/4, 334 bits: 0.089 / F1 0.147);
  - velocity-only vcaL_v is *worse* than block shift at every budget;
  - so the advantage comes from the growth term.
- **Against keyframes.** vcaL 3×3/4 (685 bits) beats keyframe 16 (1,433 bits) on error and F1. Keyframe 8
  (2,844 bits) is better still.
- **Lagrangian vs Eulerian**, paired over 72 clips: mean Δerr between −0.005 and +0.003. Lagrangian is better in
  28–56 % of clips. On 1×1 grids they are bit-identical, as they must be.
- **Persistence** (one event fitted on frames 1–8, scored on unseen frames 9–31). Error / hold / share of clips
  beating hold, vcaL 3×3:

  | tag | err | hold | beats hold |
  |---|---|---|---|
  | translation | 0.240 | 0.275 | 75 % |
  | growth | 0.220 | 0.176 | 25 % |
  | deformation | 0.129 | 0.099 | 8 % |
  | multi_motion | 0.305 | 0.295 | 33 % |
  | major_transform | 0.156 | 0.117 | 33 % |
  | cut_wipe | 0.301 | 0.297 | 25 % |

  F1 on unseen frames is ≤ 0.05 everywhere.
- **Difficulty strata** (vcaL 3×3/8):
  - easiest 25 % of clips by hold error: 0.048 vs hold 0.075 (keyframe 16: 0.033);
  - hardest 10 %: 0.148 vs hold 0.494.
  - The oracle field handles hard clips relative to hold; it is not winning only on static windows.
- **Same scene, two starts:** \|Δerr\| 0.019, correlation 0.84.
- **Prospective milestone (F1 ≥ 0.30):** not reached. The best class is translation at 0.28, with oracle events every
  4 frames.

## 4. V1: one targeted change (`experiments/pv_v1.py`)
- **Hypothesis:** the 1 px/frame cap is why velocity-only VCA trails block shift.
- **Test:** 2 substeps per frame, same alphabet and bits.
- **Result:** no gain. The variant is slightly worse in 5 of 6 tags (e.g. growth 0.079 → 0.090) and better only on
  cut_wipe (0.131 → 0.108); persistence is unchanged.
- **Rejected.** The likelier cause is that VCA moves only foreground cells, while the block shift moves the whole
  picture. This is untested.
- **Persistent velocity registers:** already answered by lagrangian = eulerian (§3); carrying velocity does not reduce
  the event rate.
- **Not pursued further:** the extension budget, and therefore V2 learning, were not justified.

## 5. Failure isolation (`experiments/pv_long.py`, `runs/path_v/long.json`, `fig_long_*.png`)
96-frame rollouts across the end of 8 preregistered scenes; vcaL 3×3, oracle events every 8 frames.

| transition | continuous: divergence | error before / after | DIAGNOSTIC restart at the transition: error after | one event before (persistence): divergence |
|---|---|---|---|---|
| 355 (wipe) | 355 | 0.059 / 0.087 | 0.086 | 344 (37 frames after start) |
| 1218 | 1219 | 0.082 / 0.152 | 0.112 | 1183 (13) |
| 1686 (cut) | 1686 | 0.095 / 0.224 | 0.136 | 1686 |
| 2242 | 2259 | 0.092 / 0.165 | 0.108 | 2207 (13) |
| 2861 | — | 0.166 / 0.167 | 0.124 | 2827 (14) |
| 3993 | 3992 | 0.093 / 0.080 | 0.047 | 3963 (18) |
| 4565 | 4566 | 0.067 / 0.200 | 0.199 | 4544 (27) |
| 5319 | 5333 | 0.070 / 0.103 | 0.099 | 5283 (12) |

- **Continuous rollouts break at the transition.** Divergence coincides with it in 6 of 8 cases; the other 2 diverge
  14–17 frames later.
- **A diagnostic restart lowers post-transition error in 5 of 8.**
  - It does not help on wipes (355, 4565, 5319), where new content keeps arriving: transport cannot create it, and
    oracle growth can only "repaint" it slowly (visible in `fig_long_1686.png`).
- **Persistence also fails inside smooth scenes:** a single event diverges 12–37 frames after its start, *before* the
  transition in 7 of 8 segments.

## 6. Earlier experiments re-stratified (`experiments/pv_reanalysis.py`, `runs/path_v/reanalysis.json`)
Historical verdicts are unchanged; this only adds stratification.
- **Stage 0 (K = 8 oracle) by tag:**
  - one-step gap 0.25–0.42 everywhere;
  - beats hold in 75 % of multi_motion windows and 11 % of growth windows;
  - rollout F1 ≤ 0.05 in every tag;
  - the single translation window: gap 0.92, rollout F1 0.15.
- **Readable Rules G1:**
  - methods beat hold in 11–50 % of windows per tag; the few "successes" (beat hold with lag-4 F1 ≥ 0.2) come from
    about 15 distinct windows (50 seed-and-method instances), several of them cut windows;
  - this is not a hidden smooth-class success.
- **Path D4** (recorded windows): on windows the atlas tags static or deformation the circuits are 3–10× worse than
  hold, so they inject change. Only multi_motion windows inside scene 3998 are near hold.
- **New inference:** the D4 (503 / 476 / 432-gate) and D3 (64-gate) circuits run on all 410 windows.
  - They beat hold in at most 0–44 % of windows per tag, and in 0–2 % of deformation windows.
  - At most 2 windows anywhere reach F1 ≥ 0.2 while beating hold; the best windows are their own training scene
    (3998–4016).
  - **No overlooked growth success:** the growth-trained circuit beats hold in 1–5 % of growth windows elsewhere.
- **Conclusion:** earlier failures are **not** concentrated at cuts; they occur in smooth single-motion windows as well.
  The Neural CA's failures *are* concentrated at cuts.

## 7. Failure taxonomy (evidence → diagnosis)

| failure | diagnosis | evidence |
|---|---|---|
| a single event does not predict frames 9–31 except in translation clips | **A information insufficiency** (motion changes within ~12–37 frames), plus some D drift | persistence table, `once_pre` divergence before transitions |
| no gain from carried velocity; velocity-only VCA trails block shift | **B representation**: foreground-only transport cannot move background structure; no help from higher speed (V1) | lagrangian = eulerian; V1 |
| walking or articulated figures become blobs (`fig_failure_deformation.png`) | **B representation** (no articulation) and **E spatial discontinuity** (regional codes tear limbs) | per-tag F1 deformation 0.10 even with oracle events every 4 |
| collapse at cuts and wipes | **G abrupt transition** (transport cannot create a new scene; growth repaints slowly) | §5, restart diagnostics |
| old circuits worse than hold on static windows | **F metric**: they inject change; pixel error hides it in moving windows | §6 D4 by tag |
| D3/D4 circuits do not transfer to other windows of the same class | **A/B** (Stage 0: no shared local law) rather than **C optimisation** | §6; Stage 0 |

## 8. Visual artifacts (`runs/path_v/`)

| file | content |
|---|---|
| `fig_success_translation.png`, `fig_success.gif` | **selected** best translation test clip 2304: vcaL 3×3 every 8, err 0.065 (hold 0.422), F1 0.29, IoU 0.80. Recognisable for about 12–16 frames, then the arm is lost and the edges tear |
| `fig_success_translation_block.png` | the same clip with block shift |
| `fig_once_translation.png` | the same clip with a single event |
| `fig_failure_deformation.png` | 5264: a walking figure becomes a growing blob |
| `fig_failure_major.png` | 4816 |
| `fig_long_*.png` | timelines with event ticks, the transition, divergence marks, and the diagnostic restart |

## 9. Reproduce

```sh
.venv/bin/python experiments/pv_atlas.py
.venv/bin/python experiments/pv_v0.py run test            # manifest already frozen; ~4.4 core-h, 64 workers
.venv/bin/python experiments/pv_v1.py; .venv/bin/python experiments/pv_long.py
node player/nca_replay.mjs runs/featured/whole_song_s0.ncab /tmp/nca.f32   # ~4.5 min, for the NCA reference
.venv/bin/python experiments/pv_report.py /tmp/nca.f32; .venv/bin/python experiments/pv_reanalysis.py /tmp/nca.f32
.venv/bin/python experiments/pv_figs.py
.venv/bin/python tests/test_pv_vca.py; .venv/bin/python tests/test_pv_campaign.py
```

**Cost:**
- CPU 6.67 core-h metered (`runs/path_v/cpu_ledger.jsonl`), plus about 0.25 core-h unmetered (estimate: NCA replay
  0.075, tests, reports, figures).
- GPU 0. No interrupted or crashed runs, apart from two rejected atlas rule versions (recorded).
