# Evaluation

A result counts only if it comes from an **autonomous replay of the export**, scored over the **full interval**
against the **preprocessed binary original**.

- The replay starts from the file alone: first frame, stored initial state if any, weights or rules, schedule.
- No reset, no injected frame, no coordinate or frame-number input. Training windows may be sampled; evaluation
  never is.
- Frame 0 is given and is excluded from every metric. Metrics use frames 1..T−1.
- Metrics run on arrays, never on rendered video.
- Teacher-forced one-step accuracy is a diagnostic. It is always reported separately and never as a result.

## What is compared with what

| comparison | meaning | reported as |
|---|---|---|
| CA replay vs binary original | accuracy of the CA | every error and shape metric below |
| Python replay vs Node replay of the same export | agreement between two implementations | binary mismatch pixels; max/mean continuous difference per quarter of the replay |
| binary original vs source video | preprocessing loss (480×360 → 48×36 box → threshold 128) | separate; `experiments/exp48_preproc_video.py` renders the three stages side by side |

Implementation agreement is not accuracy. A model can agree with itself bit for bit and still be far from the
video.

## Metrics

Defined in `ca/diag10.py` (`full_report`), `ca/motion9.py` (`lag_terms`) and `experiments/exp46_eval12.py`.

| metric | definition |
|---|---|
| pixel error | fraction of wrong pixels over frames 1..T−1; also split into FP (extra white) and FN (missing white) |
| thirds | the same error over each third of the interval (late-third growth shows drift) |
| worst 16-frame window | maximum mean error over any 16 consecutive frames |
| band / interior / background error | errors inside the 2-px contour band of the original, inside the shape, and in the background |
| contour precision/recall | inner boundary of replay vs original, 3×3 tolerance |
| lag-1 / lag-4 change F1 | changes X_t vs X_{t−k}, split into on/off, matched with a 3×3 tolerance in the same direction. Lag 4 measures change that accumulates over 4 frames; a period-2 flicker cancels at even lags and earns nothing |
| change mass | predicted changes / true changes at that lag (< 1: under-motion; > 1: over-motion or flicker) |
| thin-part / gap recall | recall of the original's thin foreground (F minus its 3×3 opening: fingers, cane) and thin gaps (closing(F) minus F: e.g. arm–body gap) |
| specks / holes | 8-connected foreground and background components of ≤ 3 px, replay vs original |
| spurious blips | pixels that flip at t and back at t+1 in the replay where the original has no such blip |
| error runs | per pixel, runs of consecutive wrong frames (count, mean, max, share ≥ 4 frames) |
| per-phase error | mean error inside each schedule entry, and over the 3 frames before/after each boundary |
| transition error | mean error over frames within ±3 of every detected cut or wipe onset (`ca/segment.signals`) |
| missed fades | frames where the original is uniform (blank) and the replay differs from it in > 5 % of pixels |
| err / hold | pixel error divided by the error of holding frame 0 |
| replay cost | numpy replay time, frames/s, peak RSS; Node replay time |

## Gates (Campaign 12, fixed before the pilot: `docs/CAMPAIGN_12.md` §4)

An interval succeeds when all of these hold:

1. **Sustained motion** (the C11 gate): in every third, lag-4 change F1 ≥ 0.5 and change mass in 0.5–1.5; and
   late-third error ≤ 3 × early-third error, or ≤ 0.03.
2. **Full error** ≤ 0.035 (the 120-frame large-model level, 0.0304 / 0.0323).
3. **Worst 16-frame window** ≤ 0.09.
4. **Transitions:** error within ±3 of every cut/wipe onset ≤ 0.10, and no missed fade.
5. **Visual veto** only (a gate can be failed by looking, never passed by it).

Comparison rules:
- "Better" means lower full error in **both** seeds and not more than 10 % worse on the worst window. With two seeds
  this is a statement about that interval and those settings only.
- Phase benefit is claimed only if a phased arm beats the capacity-matched one-rule control (`fixed256`, same
  parameter count). That control has not produced a valid result yet (non-finite in the pilot; rerun pending), so
  no phase-benefit claim is made.
- Explicit rules are "usable" only if their full error is at most the C9 level for that interval (0.0129 on
  4500–4555).
- Every seed is reported. The representative seed is named.

Earlier campaigns used their own pre-registered gates. Each CAMPAIGN file states them before its runs.

## Baselines and known loopholes

Fixed baselines run before training on each task, so a metric that rewards cheating is noticed:
- Campaign 08 (`experiments/exp26_loopholes.py`): hold, all black, all white, invert every frame (flicker),
  shift right 1 px/frame, random 5 % flips, the true clip delayed by one frame.
- Campaign 09 (`experiments/exp32_window_baselines.py`): the same plus checkerboard flicker along the silhouette edge
  and local on-then-revert blips.

Loopholes found so far:
- **Background-dominated pixel error.** Holding still scores well on mostly static frames. Hence err/hold, the
  contour band and the motion metrics.
- **Edge flicker.** In C8 a tolerant change F1 credited a checkerboard oscillating along the edge. Hence lag-4 F1,
  change mass, blips and revert rates.
- **Under-motion.** Replays that change too few pixels look smooth but lag the shape. Hence change mass ≥ 0.5 per
  third.
- **Uniform first frame.** With a blank first frame and zero hidden state, the only asymmetry is the zero boundary.
  After the first update the interior is non-zero, so border effects can spread inward; C12 measured this
  (`runs/c12/eval/opening_symmetry.json`). Whether a rule can grow the opening from the border alone was not tested.
  Any stored initial state, such as `init_g`, is counted in the file and reported as stored information, not as a rule
  achievement.

## Cost accounting

- GPU: assigned GPUs × job wall time, summed per job (shared GPUs are not discounted). Queue waits are recorded
  separately. Failed and cancelled runs count.
- CPU: core-hours of metered processes.
- Ledgers: `runs/c<N>/gpu_ledger.jsonl`, `runs/c<N>/cpu_ledger.jsonl`.

## Reproduce an evaluation

```sh
.venv/bin/python experiments/exp46_eval12.py out.json path/to/model.ncab --node   # ncab-v2, C12 gates
.venv/bin/python -m ca.pipeline replay runs/pipe/example_c10_main/model.ncab       # v1: numpy vs Node agreement
```

The first needs the Bad Apple source at `refs/Bad-Apple-Starter/BadApple!!.mp4` (see the README, "Data").
