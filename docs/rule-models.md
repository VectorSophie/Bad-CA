# Rule models

Three model families are in use. Only the first two currently replay real Bad Apple motion well. Earlier Boolean
families (LUT/SAT, CGP, DiffLogic circuits, Campaigns 01–08) are summarised in `docs/research/README.md`.

All of them share these replay rules:
- One local update, applied synchronously to every cell. Cells outside the grid read as 0 (zero boundary).
- S substeps of the update produce one video frame.
- The initial state is the first frame (visible), with hidden state 0 unless stored initial state is declared and
  counted in the file.
- Nothing from the target is read after the first frame. No resets, no frame injection, no coordinates or frame
  numbers as cell inputs.

## 1. Continuous-state Neural CA (`ca/nca.py`)

State: `s` of shape (C, H, W), float32. Channel 0 is visible, channels 1..C−1 are hidden.

One substep:

```
p = relu(conv3x3(s; Wp) + bp)      # Wp: (Fh, C, 3, 3), zero padding, all channels
s = s + Wu @ p + bu                # Wu: (C, Fh), 1x1 residual update
```

- Output frame = `s[0] > 0.5`. The continuous state is carried on unchanged; nothing is binarised inside the
  rollout. The continuous visible value is a raw state, not a probability.
- Receptive field per frame = S cells.
- Parameters = Fh·C·9 + Fh + C·Fh + C.

| size | C | Fh | S | params per rule | float32 weight bytes |
|---|---|---|---|---|---|
| reference (C10, `quality` preset) | 16 | 64 | 4 | 10,320 | 41,280 |
| C11 intermediate | 12 | 32 | 4 | 3,884 | 15,536 |
| small (`demo` preset) | 8 | 16 | 4 | 1,304 | 5,216 |

A parameter count is not a rule count. A 10,320-parameter model holds about 330 kbit of weights. It is not a small
or readable rule, and the UI never presents its weights as rules.

Training (C10 recipe, used since): Adam with per-tensor gradient normalisation, cosine learning rate, fixed steps,
loss = pixel loss L0 + 2 · LB, where LB is a hinge on the contour band of the original. Long intervals need
lr ≈ 5e-4 (2e-3 went non-finite on 120 frames). The best checkpoint is chosen by the binary error of an autonomous
rollout.

## 2. Phased Neural CA (`ca/phase_nca.py`, `ca/phase_train.py`)

A time-varying CA with an **external schedule**: the export lists half-open frame ranges `[start, end)` and the rule
that produces each frame t from frame t−1. Visible and hidden state carry across boundaries unchanged. This is
replay type 2.

| mode | parameters | per-phase rule |
|---|---|---|
| `fixed` | 1 set (K = 1) | the same rule everywhere |
| `indep` | K independent sets | K × 10,320 at C = 16, Fh = 64 |
| `delta` | shared base + per phase a rank-r update of Wp (as Fh × C·9) and Wu, plus full bias deltas | base 10,320 + K × 1,232 at rank 4 |

Effective per-phase weights for `delta` are combined in float64 and rounded to float32 once, identically in
Python and JS.

**Stored initial hidden (`init_g`, optional).** A learned array of shape (C−1, ⌈H/f⌉, ⌈W/f⌉), nearest-upsampled
by factor f to the initial hidden state. At 48×36 and f = 4 this is 15 × 9 × 12 = 1,620 floats. It is needed when
the first frame is uniform: a shift-invariant rule with a zero boundary keeps a uniform interior uniform, so the
picture has to come from stored information. It is counted in the file and reported separately; it is not a rule
achievement.

**Training.**
- `joint`: all phases together, full backpropagation through time from frame 0, so the gradient crosses phase
  boundaries and earlier phases can prepare hidden state for later ones.
- `greedy`: one phase at a time, starting from the actual carried state of the already-trained phases. No gradient
  into earlier phases. In the C12 pilot it was much worse than joint on the wipe and shot-change phases. The warm
  start (copy the previous rule) blew up when a rule ran past its trained horizon.
- `chunked` (`experiments/exp45_c12.py`): consecutive phases grouped into chunks of about 200–280 frames, joint
  inside a chunk, state carried between chunks. Used for the whole-song run (failed as a replay without noise; see `docs/CHECKPOINT_12.md` §6) and the noise-trained prefix runs.
- `--noise σ` (chunked): **training-only** Gaussian noise added to the whole state after every frame. Replay is
  unchanged. Without it, the chained song rules amplified float-order differences between the GPU training state and
  the CPU replay, and diverged after ~1,080 frames. With σ = 0.02 the replay equals the training error in every chunk
  (frames 0–2115; `docs/CHECKPOINT_12.md` §6).

**Segmentation (`ca/segment.py`).** Stage 1 analyses the video only. It finds K phases by an exact dynamic
programme minimising the code length of "next centre bit | 3×3 pattern of the previous frame" under one table per
phase, plus a BIC table cost. It also detects cuts (≥ 25 % of pixels flip, or a uniform frame starts or ends) and
wipes (sustained change ≥ 8 %). Each boundary gets a reason (`cut`, `wipe`, or `mdl` with its bit gain). The 3×3
table is a proxy: the CA reaches 4 px per frame. Whether a boundary helps a CA is only decided by training. In the
C12 pilot, automatic boundaries at K = 4 were not better than equal-length ones.

## 3. Explicit rules, dlrule-v1 (`ca/rules.py`, `player/rule_core.mjs`) — experimental

State per cell: visible bit v and k ≤ 3 hidden bits h0..h(k−1).

Features, all small integers computed from the current grid:

| feature | meaning |
|---|---|
| `v`, `h0`..`h(k-1)` | own state |
| `n8` | visible cells in the 3×3 Moore neighbourhood, centre excluded (0–8) |
| `r2` | visible cells on the radius-2 ring of the 5×5 square (0–16) |
| `N`, `S`, `E`, `W` | visible cells among the 3 neighbours on that side (0–3) |
| `m0`..`m(k-1)` | Moore neighbours with hidden bit i set (0–8) |

A program is an ordered list of rules `if [feature op const]... then {v, h}` with op in `<=`, `>=`, `==`. The first
rule whose conditions all hold sets the next state; if none fires, the cell keeps its state. A schedule chooses the
program per frame, as in type 2. Text form, e.g. Conway's Life:

```
R0: if v == 1 and n8 <= 1 -> v=0
R1: if v == 1 and n8 >= 4 -> v=0
R2: if v == 0 and n8 == 3 -> v=1
default: keep state
```

**Learning.** A best-first CART tree (Gini, integer thresholds, own code) predicts the next full state from the
features. Each leaf becomes one rule. Two routes (`experiments/exp47_rules.py`):
- *direct*: k = 0, S = 1, trained on consecutive true frames (teacher-forced pairs), one program or one per phase;
- *extract*: run a trained NCA substep by substep, quantise its state (v = visible > 0.5; k hidden bits from the k
  highest-variance hidden channels, thresholded at their median), learn quantised state → next quantised state at
  S = 4.

**Evaluation.** Always by executing the program autonomously. The teacher-forced one-step match is reported
separately and is not a result.

**Complexity** (`complexity()`): programs, rules, conditions, constants (= conditions + 2 × rules), hidden bits,
rules JSON bytes, and the zlib-9 size of rules + first frame + h0.

**Results (C12 pilot, `runs/c12/eval/rules_pilot.json`).**
- Positive controls: Conway's Life recovered exactly as 3 rules; a known hidden-bit rule recovered exactly.
- Bad Apple 4500–4555 (56 frames): one program collapses to "hold" (unweighted) or to flicker (change-weighted).
  K = 14 phase programs with 117 rules reach error 0.0995, against hold 0.183 and the C10 NCA 0.004.
- Extraction from the C10 NCA with 1–3 hidden bits: at best 0.153.
- So the engine and the learner work on rules that exist, but this representation and learner do not yet capture
  real motion.

## Replay types

| type | what it is | status |
|---|---|---|
| 1 | one fixed rule | implemented |
| 2 | external schedule, state carried across boundaries | implemented; main path for long intervals |
| 3 | the cells' own state selects the rule | not implemented |
| 4 | clips concatenated with resets | diagnostic only; never counted as autonomous |
