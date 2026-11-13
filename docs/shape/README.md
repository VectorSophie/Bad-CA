# Track S: shape-aware CA (exploratory, CPU only)

Branch `research/generalization-shape`. All rules are **hand-written** and driven by **oracle** controls (chosen from
the intended frames, sequential coordinate descent per control unit, events every 4 frames on synthetic tasks and
every 8 on real windows). The runtime receives the start frame plus the schedule. GPU 0.

## Candidates (`experiments/shape_ca.py`)

- **S1 `lsca` (level-set CA):**
  - each cell keeps a clamped signed city-block distance φ ∈ [−6, 6], and the shape is φ < 0;
  - radius-1 rules: a normal speed moves the zero level one cell (grow/shrink with a sub-cell accumulator), a local
    re-distancing keeps φ consistent, and an optional smoothing flips cells that 6 or more of their 8 neighbours
    disagree with;
  - control per region of a 3×3 grid: (speed, smooth).
- **S2 `ffdca` (structural control + local transport):**
  - a 3×3-node lattice (2×2 cells) of velocity nodes;
  - every cell bilinearly interpolates its 4 surrounding nodes. **This read reaches a lattice spacing (16–24 cells),
    so S2 is a hybrid, not a radius-1 CA.**
  - then a local rule: sub-pixel accumulators, and each cell **pulls** one neighbour's value (no holes or collisions
    by construction), plus a 1-px speck/pinhole cleanup.
- **`ffdwarp`:** the geometric reference, the same lattice field applied as a backward warp of the event frame
  (not a CA).
- **Baselines:** block shift (Path V), the velocity CA (Path V, `vcaL`), morphology (per-region hold/dilate/erode).

### Bug found and fixed during this campaign
The first `lsca` let boundary cells reach φ = 0 and disappear, so it eroded one layer per frame even with zero growth.
`tests/test_gen_shape.py` (identity at zero growth) caught it.
- Results from that version are kept as `runs/shape/*_v1_lsca_bug.json` and `fig_synth_v1_lsca_bug.png`.
- The tables below come from the fixed code; only the `lsca` rows changed.

## Synthetic tasks (`runs/shape/synth.json`, `fig_synth.png`, `assets/release/shape_ring.gif`)
48×36, 16 frames, analytic shapes with known deformation:

| task | what happens |
|---|---|
| arm | a 2-px arm rotates 60° about a joint |
| ring | a 1.5-px elliptic ring squashes |
| grow | a disk grows while carrying a 1-px antenna |
| line | a 1-px segment bends into an arc |
| rigid | a bar rotates 30° while translating |

IoU / thin-structure recall / extra connected components (tearing):

| method | bits | arm | ring | grow | line | rigid |
|---|---|---|---|---|---|---|
| block | 167 | 0.80 / 0.31 / +0.33 | 0.42 / 0.56 / **+3.00** | 0.66 / 0.81 / 0 | 0.60 / 0.66 / +1.27 | 0.72 / 0.17 / 0 |
| vcaL | 342 | 0.83 / 0.42 / +0.93 | 0.46 / 0.53 / **+3.73** | 0.89 / 0.80 / 0 | 0.63 / 0.70 / +0.53 | 0.81 / 0.29 / 0 |
| morph | 57 | 0.78 / 0.18 / 0 | 0.14 / 0.18 / +0.33 | 0.82 / 0.34 / 0 | 0.48 / 0.52 / 0 | 0.61 / 0.15 / 0 |
| lsca (S1) | 150 | 0.78 / 0.20 / 0 | 0.19 / 0.25 / −0.40 | 0.89 / 0.32 / 0 | 0.47 / 0.50 / 0 | 0.64 / 0.19 / 0 |
| **ffdca (S2)** | 228 | **0.85 / 0.57 / 0** | **0.74 / 0.89 / 0** | **0.90 / 0.80 / 0** | **0.70 / 0.76 / 0** | **0.89 / 0.36 / 0** |
| ffdwarp (reference) | 228 | 0.86 / 0.59 / 0 | 0.64 / 0.87 / 0 | 0.91 / 0.76 / 0 | 0.68 / 0.72 / +0.60 | 0.87 / 0.18 / 0 |

**Observed:**
- **S2 is best or tied on every task, with zero tearing on all five.** It is the only local method that squashes the
  ring without breaking it. In `shape_ring.gif`, block shift and the velocity CA tear the ring into fragments.
- **S2 matches or beats the purely geometric warp.** Pulling one value per cell plus the local cleanup costs
  nothing here.
- **S1 adds nothing beyond morphology for shape change.** It is good for pure growth (grow: IoU 0.89) and keeps
  connectivity, but cannot transport (ring, rigid) and loses 1-px features (grow's antenna, thin recall 0.32).

## Real windows (`runs/shape/real.json`)
32 frames, oracle events every 8 frames.
- **Bad Apple:** the 24 V0 test clips tagged deformation or major_transform.
- **Fantasmagorie:** the first 12 cut-free deformation / major_transform windows under the Path V atlas rules, applied
  unchanged (`runs/shape/atlas_fantasmagorie.json`).
  - The film's atlas is dominated by appear_disappear (36 windows) and major_transform (29). Line art is redrawn rather
    than moved.

| method | bits | Bad Apple err / IoU / extra components | Fantasmagorie err / IoU / extra components |
|---|---|---|---|
| hold | 0 | 0.122 / 0.48 / 0.94 | 0.149 / 0.51 / 0.98 |
| keyframe every 16 | ~1,450 / ~2,480 | **0.051 / 0.70 / 0.63** | **0.081 / 0.70 / 0.90** |
| block | 167 | 0.073 / 0.60 / 1.51 | 0.128 / 0.55 / 1.10 |
| vcaL | 342 | 0.065 / 0.64 / 1.14 | 0.121 / 0.57 / 1.22 |
| lsca (S1) | 150 | 0.079 / 0.59 / 1.08 | 0.129 / 0.53 / 1.07 |
| **ffdca (S2)** | 228 | **0.060 / 0.65 / 0.94** | 0.126 / 0.56 / 0.94 |
| ffdwarp | 228 | 0.061 / 0.65 / 0.92 | 0.132 / 0.54 / 1.08 |

**Observed:**
- **Bad Apple:** S2 is the best CA method, with the lowest error and the fewest spurious components, at two thirds of
  the velocity CA's bits. Keyframes are clearly better, at 6× the bits.
- **Fantasmagorie:** every transport method is barely better than hold, and motion F1 is about 0.05. The film's
  motion is mostly appearance and redrawing, which no transport rule can produce.

## Interpretation and limits
- **Supported:** a small set of smoothly interpolated structural controls preserves connectivity and thin parts much
  better than independent per-region pixel motion. This is the gap Path V left open (tearing at region boundaries).
- **Not supported:** that S2 reconstructs real animation substantially better than existing baselines. The real-window
  gain over the velocity CA is small, and keyframes dominate at higher budgets.
- **Hypothesis (untested):** structural control would matter more at 96×72 or above, where limbs are several cells
  wide.
- **Honest label:** S2 is a hybrid geometric/CA system (non-local interpolation, oracle controls). It is not Conway-style
  emergence. No GPU was used; learning the lattice controls from observations was not attempted.
