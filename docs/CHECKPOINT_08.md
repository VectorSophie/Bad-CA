# Checkpoint 08: can a small CA circuit follow Bad Apple's motion into the late part of a clip?

**Sources**
- **Plan:** `docs/CAMPAIGN_08.md`. The plan, λ rule, thresholds and selection rule were fixed before the
  runs. Later decisions were appended per those rules.
- **Machine-readable:**

  | file | contents |
  |---|---|
  | `runs/c8/results.json` | 61 runs |
  | `runs/c8/summary08.md` | readable tables |
  | `runs/c8/loopholes.json` | baselines |
  | `runs/c8/oscillation.json` | post-hoc diagnostic |
  | `runs/c8/cpu_ledger.jsonl` | every metered process |

- **Exports:** a `*.best.cab` next to every result.

## 1. Videos to open first
All files are in `runs/c8/media/`.
- **Format:** MP4 (H.264, verified by decoding back: 60 frames at 30 fps = 2.0 s, and at 7.5 fps = 8.0 s) +
  GIF, each at 1× and 0.25×.
- **Opening:** from the repository root, `xdg-open runs/c8/media/<name>_1x.mp4`, or mpv/VLC, or drag
  into a browser.

| # | file (`…_1x.mp4` / `…_0.25x.mp4` / `.gif`) | shows |
|---|---|---|
| ① | `main_ba48_4500_60_1_orig_c7_c8` | original / C7 best (6 gates) / **C8 representative** (arm B, seed 4, 10 gates) |
| ② | `main_ba48_4500_60_2_representative_orig_c8_diff` | original / C8 representative / diff (red = missed white FN, blue = extra white FP) |
| ② | `main_ba48_4500_60_2_best_orig_c8_diff`, `…_worst_…` | best seed (0) and worst seed (2) of the same arm |
| ①② | `second_ba48_1800_60_*` | the same set for the second clip (arm B: representative seed 2, best 0, worst 1) |
| ③ | `control_{pcmove,pccycle}_{k1_S2,k1s_S4}_3_orig_recovered` | positive controls: original / recovered / diff |
| pre | `c7_best_vs_hold` | the first video, made before any search: original / C7 best / hold |

- **Panel labels:** clip, source frame number and time, arm, seed, active gates, k, S.
- **Footers:** per-segment F1tol and err/hold, plus the pre-registered level.
- **Replays:** every one is **decoded from the exported file**. There is no interpolation, editing or re-timing.

## 2. Did the motion get followed into the late segment?
**No.**
- **Success** (all three segments followed): **0 of 32 runs** of the 2×2 on both clips, and 0 of 3
  curriculum runs.
- **Partial** (early followed, late not): main clip A 3/5, B 3/5, C 0/5, D 2/5; second clip B 2/3, the
  other arms 0/3; curriculum 3/3.
- **The non-zero late scores are flicker, not motion.**
  - The motion objective did raise late F1tol (main-clip B median 0.173 vs A 0.000).
  - The videos and the **post-hoc revert diagnostic** (`exp29_oscillation.py`) show this comes from a
    checkerboard band **oscillating in place** along the silhouette edge, next to where the real
    movement happens.
  - Late changes of those seeds flip back on the next frame **45–100%** of the time. For the original
    clip it is **17%** (main) and **10%** (second).
  - The tolerant F1 credits such flicker because it sits near true changes. The pre-run loophole check
    covered only *global* flicker and random flips, so this **metric loophole** was missed.
  - The criteria are not changed after the fact: by the pre-registered rule these seeds are
    "partial/fail", and none is called motion.
- **What the circuits do reproduce:** as in C7, the coarse early silhouette change (early F1tol
  0.25–0.40; error 0.49–0.61× hold overall).
- **Re-evaluating the C7 best** under the new metrics: early 0.253, mid 0.034, late 0.000. That is
  **partial**, with the late segment frozen (period 1).

## 3. Objective change vs structure change

| arm | structure | objective | main: success / partial / fail | median late F1tol (range) | median err/hold | median gates | evals per 600 s |
|---|---|---|---|---|---|---|---|
| A | base k1_S2 | pix | 0/3/2 | 0.000 (0.000–0.019) | 0.49 | 11 | 87k |
| B | base k1_S2 | mot λ=2 | 0/3/2 | 0.173 (0.000–0.298) | 0.61 | 11 | 72k |
| C | ext k1s_S4 | pix | 0/0/5 | 0.000 (0–0) | 0.52 | 6 | 56k |
| D | ext k1s_S4 | mot λ=2 | 0/2/3 | 0.000 (0.000–0.229) | 0.61 | 6 | 50k |

- **Objective (B vs A):** a detectable effect on **late F1tol** by the pre-registered rule.
  - Median +0.17, and 4/5 of B's seeds are above A's median.
  - The pixel error is **worse** (0.61 vs 0.49).
  - The gained "change" is local oscillation (§2). **Net: the objective changes the behaviour, from
    freezing to flickering in place, but not toward motion.**
- **Structure (C/D vs A/B):** **no benefit.**
  - C and D are equal or worse on every measure.
  - S=4 roughly halves throughput (56k vs 87k evaluations).
  - Because there was no improvement, the pre-registered ablation (neighbour hidden alone vs S=4
    alone) was **not run**. The two changes can only be reported jointly: "no gain at this budget".
- **λ pilot:** the min-segment F1tol was 0.025 / 0.02 / **0.14** for λ = 0.5 / 1 / 2, so λ=2 was fixed.
  Larger λ buys late F1 by giving up pixel error (err/hold 0.48 → 0.63).
- **Curriculum** (arm B with prefix 20 → 40 → 60, 3 seeds): late F1tol 0.04–0.19, 3/3 partial, the same
  flicker signature (revert 0.40–0.99).
  - It evaluates about 2× more candidates in the same time (137k).
  - **No improvement.**

## 4. All seeds and a representative failure
- **Every run:** `runs/c8/summary08.md` and `results.json`.
- **Representative failure:** main-clip **arm B seed 0**, the "best" late F1tol of 0.30.
  - It fails the mid segment (F1tol 0.15, tolerant precision 0.10, mass ratio 3.0).
  - Its late changes revert 90% of the time.
  - In the diff video the arm and head are entirely red (missed), while a flickering checkerboard sits
    at the left edge.
- **Representative "frozen" failure:** main **B seed 2** (late period 1).
- **Second clip, arm D seed 1:** late F1tol 0.28 and precision 0.29, but late error is 1.12× hold, so
  the late segment is **not followed**.

## 5. Rule size, storage, compute
- **Rule size:** active gates 2–12 per run (median 6–11 by arm).
- **Files:** 89–100 bytes on the main clip (best baseline 1,675 bytes lzma) and 123–137 bytes on the
  second clip (3,031).
  - **These are not compression results.** The replays are approximate and do not follow the motion.
- **Replay cost:**
  - (T−1)·S·H·W = 59·2·1728 = 204k cell updates (base), 408k (ext);
  - Node decode a few ms per file.
- **Search compute:** 600 CPU-s per run, 72k–87k evaluations (base) and 50k–56k (ext).
- **Independent verification:** all 61 exports (controls, pilot, main, second, curriculum) replay
  **byte-identically to the Python simulation** in Python and in the Node player from the file alone.
  - Binary states, one fixed rule, initial state = first frame + hidden 0.
  - "Python = Node" holds for all of them; "replay = original" holds **only** for the 18 exact control runs.
- **Total:** **6.55 CPU core-h** of 12; **GPU 0** (CGP is numpy grid simulation of tens of gates; no
  accelerator benefit).

  | part | core-s |
  |---|---|
  | main + second clip | 19,217 |
  | controls + λ pilot | 2,430 |
  | curriculum | 1,802 |
  | smoke, loopholes, videos, diagnostic | about 90 |

## 6. Next focus and reasons
1. **Fix the measurement before more search.**
   - The tolerant F1 has an exploitable hole: local in-place flicker.
   - The next objective and success rule should include the revert rate, or temporal persistence:
     a change should persist for ≥ 2 frames, as in the original, where 83–90% persist.
   - Add local flicker as a loophole baseline.
   - Without this, any search will keep finding flicker.
2. **The capacity question is still open.** Every circuit found tracks the coarse shape for about 15–20
   frames and then freezes or flickers.
   - The controls show CGP can discover sustained motion and periodic change when a small circuit
     generates them.
   - For Bad Apple it is unknown whether no ~10-gate rule exists or whether the search can't find it.
   - The most informative next step: give the search a **mid-clip** start state, as a separate,
     explicitly non-autonomous diagnostic, to see whether ~10-gate rules can follow the later motion
     *locally*.
3. **Not recommended:**
   - the extended structure (no gain, half throughput);
   - curriculum as implemented;
   - more seeds of the same arms.

## Reproduction
- **Snapshots:**

  | snapshot | used for |
  |---|---|
  | `runs/frozen/df872b43bf6d` | controls, λ pilot |
  | `runs/frozen/ef180e8ee68b` | main + second clip |
  | `runs/frozen/d66459609ecf` | curriculum |

- **Search:** `python experiments/exp27_cgp2.py TARGET {k1_S2|k1s_S4} 32 SEED 600 {pix 0|mot 2|motcur 2} OUTDIR`, run
  through `experiments/c8_meter.py`. The job lists are in `runs/c8/jobs_*.txt`.
- **Analysis:**
  - `python experiments/summarize08.py`
  - `python experiments/exp28_videos.py` (videos)
  - `python experiments/exp26_loopholes.py`
  - `python experiments/exp29_oscillation.py`
- **First video (C7):** `python experiments/exp25_c7video.py`.
