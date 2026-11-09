# Checkpoint 07: widening the method space (four pilots)

**Sources**
- **Plan and survey:** `docs/RESEARCH_PHASE_07.md`, written before the pilots; the follow-up was
  appended before it ran.
- **Machine-readable table:** `runs/c7/results.json`, 53 runs, each with its result file, export and
  media. The readable version is `runs/c7/summary07.md`, built by `experiments/summarize07.py`.
- **Exports:**
  - every rule is a container file (`*.best.cab`);
  - every one replays **byte-identically to the method's own Python rollout** in Python **and** in the
    independent Node player (53/53);
  - whether that rollout matches the video is evaluated separately (exact / approximate).

**Setting:**
- autonomous replay: one shared local rule, visible init = first frame, hidden init = a uniform 0,
  boundary 0, fixed S;
- no frame injection.
- All Bad Apple targets start at non-uniform frames, skipping the black opening (43 black frames;
  182 uniform frames in total). **The full-song symmetry problem is not addressed.**

## Answers first

**1. Did we find a more promising method than before?**
- **Partly.** For *simple* rules, evolutionary search directly on the hard rollout (CGP) is the best
  of the four pilots. It needs no relaxation and no SAT.
  - It recovered the planted circuits exactly (pcA 4/4 seeds, pcB 2/4).
  - On the 2-second clip `ba48_4500_60`, circuits of **3–10 active gates** reach **0.48–0.70× the
    hold-frame error**, across all 12 seed/budget runs.
  - Each such rule fits in an 84–96-byte file.
- **But no method reproduced the actual motion of a long clip.**
  - Change-IoU, the overlap between predicted and true changes, stays at **0.003–0.044** for every
    method on both 60-frame clips.
  - The error gain comes from reproducing the **coarse silhouette change** (the dress becomes an upright
    figure within about 10 frames), then holding still.
  - So the gain is real, but it is "coarse shape", not "motion".

**2. Simplest rule, most meaningful motion**
- **Exact (synthetic):** `translate24` (a moving block, 12 frames) is reproduced exactly by a
  **1-condition** second-order rule, `next = now W`: 20 bytes, against 55 bytes for zlib. This is a
  trivial motion.
- **Bad Apple (approximate):** a **6-gate** CGP circuit on `ba48_4500_60`, 89 bytes, error 0.479× hold.
  - **Rule:** visible′ = NOT(C OR (W AND (S XNOR (SE OR h)))) and hidden′ = (NW AND W) OR h.
  - **How it runs:** visible inverts at every generation, so at S=2 only the observed frames look
    upright, not the intermediate ones; hidden is a spreading latch.
  - **Result:** it narrows the dress silhouette into an upright column over about 10 frames, close
    to the real transition (`runs/c7/c4_cgp/ba48_4500_60__k1_S2__G32__s3.best.gif`), and then freezes.
  - Horns, arms and the later movement are missing.
  - It is fast at first, so it is worse than hold at frame 1 (horizon 1).
- **Bad Apple 5 frames:** the 3-condition tree `if nowC → 1; elif prevE and prevS → 1; else 0` beats
  hold on ba32_3000_5 (0.82×, change-IoU 0.14). It is a **growth-only** rule: white never turns black.

**3. What was possible, exact and approximate**
- **Exact:**
  - every positive control: Life B3/S23 (C2), the second-order tree (C3), pcA/pcB (C4), pcA_long over
    18 frames (C5);
  - the synthetic translation (C3, C4).
  - **No exact Bad Apple replay in this phase.** None of the four methods targets exact Bad Apple, and
    the exact 5–12-frame rules from CP02–04 remain the only exact ones.
- **Approximate** (`ba48_4500_60`):

  | method | err/hold | rule size |
  |---|---|---|
  | C4 | 0.48–0.70 (best 0.48) | 3–10 gates |
  | C3 | 0.67 at best | 1,466 conditions; small trees ≈ 1.0 or worse |
  | C2 | 0.79 (S=2) | 18 bits |
  | C5 | **1.00** (collapsed to hold) | – |

  - Change-IoU ≤ 0.044 for all of them.
  - The spatially tolerant error, where a pixel counts only if its value is absent from the target's
    3×3 window, drops to 0.032 for the best C4. Most of the remaining error is near edges, not
    wrong content.
- **Transfer** (`ba48_1800_60`, identical procedure; thin, fast details):

  | method | err/hold | rule size |
  |---|---|---|
  | C4 | 0.75–0.86 | 2–8 gates |
  | C3 | 0.79 | 53 conditions |
  | C2 | 0.87 | 18 bits |

  - Change-IoU ≤ 0.01.
  - **The procedure applies to a new clip unchanged, but the gain shrinks**, and motion is again not
    reproduced.

**4. Trade-offs: compression, simplicity, replay length** (not combined into one score)
- **Exact:** small files beat the tested baselines only on clips that are themselves simple CA:
  - translate24: 20 vs 55 bytes;
  - pcA: 27 vs 116;
  - pcB: 29 vs 90;
  - Life control: 141 vs 710.
  - On Bad Apple the exact rules (CP02–06) remain larger than the baselines. Exact replay length
    stops at 12 frames (CP04).
- **Approximate:** CA files of 84–176 bytes against 1,675 (lzma, ba48_4500_60) and 3,031 bytes
  (ba48_1800_60).
  - **This is not compression.** The replays are not faithful (change-IoU about 0), so the byte
    comparison is meaningless.
- **Simplicity vs quality on the same clip:**
  - C4 with 3–10 gates beats C3 with 1,466 conditions.
  - Within C4, G=8/16/32 give overlapping distributions (best 0.48–0.51 for each), so more gates did
    not help at 600 s.
  - The "autonomous length" with useful content is short: the gain is concentrated in the first
    about 10–15 frames. The error curve then tracks hold, because the CA has frozen while the video
    keeps moving.

**5. Next focus and backup**
- **Focus: discrete program search (CGP) on the hard rollout, with an objective that rewards
  motion.** Evidence:
  - It is the only method that found 3–10-gate rules beating hold substantially on both long clips.
  - It passed its controls.
  - It has no soft/hard gap.
  - It is cheap (600 CPU-s per run).
  - Its failure mode is now specific: it optimises mean pixel error, which a frozen coarse shape
    satisfies.
  - **Next experiment:** a fitness that includes change-IoU / tolerant error and a horizon curriculum
    (extend frames when the current prefix is matched). Also try the structures that gave motion
    capacity in theory: neighbour hidden and S=4.
- **Backup: the second-order CA with supervised tree induction (C3).** Evidence:
  - it learns in seconds;
  - its rules are directly readable (`if nowW → 1`);
  - it is exact on the motion controls.
  - Its limit is structural: S=1 with radius 1 cannot follow Bad Apple's motion (the light cone,
    CP01). Its large trees overfit into drifting blobs.
  - **Next:** let it supply features or seed genomes for CGP (supervised, then refined on the
    rollout), or use radius-2 features.
- **Deprioritised:**
  - C2 Life-like: isotropic and memoryless, it cannot express direction (the best rule only erodes or
    smooths).
  - C5 DiffLogic on long horizons: the **soft loss fell 2.6×** (20k → 7.2k) while **hard rollout error
    rose** (seed 1000: 0.25 → 0.54), so the CP03 soft/hard gap recurs at 60 frames. The 2025 Gumbel
    straight-through fix ("Mind the Gap") is untested here; it was shown only on feed-forward CIFAR.

## Per-method summary

| method (paradigm) | positive control | common: ba32_3000_5 | common: ba48_4500_60 (err/hold, change-IoU, size, file B) | search cost |
|---|---|---|---|---|
| C2 Life-like, exhaustive 2^18 | **pass** (B3/S23 exact; 4 equivalent rules) | no rule beats hold (S=1, 2) | S=2: B0134/S0124, 0.79, 0.025, 18 bits, 151 B. S=1: B4/S345678, 0.90, 0.007 | 142–271 CPU-s per clip × S |
| C3 2nd-order + CART/DAgger | **pass** (a 3-condition tree, smaller than the generator's 5, exact) | 3 conditions, 0.82, 0.14, 62 B | 1,466 conditions, 0.67, 0.036, 7,032 B; depth ≤ 6 trees ≥ 0.83 | 0.4–7 CPU-s |
| C4 CGP (1+4)-ES on hard rollout | **pass** (pcA 4/4, pcB 2/4 exact) | best 0.89 (4 gates); 2/4 seeds stay at hold | best 0.48 (6 gates); 12 runs 0.48–0.70; change-IoU 0.009–0.044; 84–96 B | 600 CPU-s per run |
| C5 DiffLogic long (GPU) | **pass** (pcA_long exact, seed 1000, 31 s) | not run (CP03) | **1.00** (hold; 0 live gates); soft loss ↓ while hard error ↑ | 1,502 GPU-s + 9 s compile |

- **Seed distributions** (C4, `ba48_4500_60`, err/hold):

  | gate budget | 4 seeds |
  |---|---|
  | G=8 | 0.51, 0.52, 0.57, 0.70 |
  | G=16 | 0.50, 0.50, 0.54, 0.56 |
  | G=32 | 0.48, 0.54, 0.56, 0.57 |

  ba32_3000_5 (G=16): 0.89, 0.93, 1.0, 1.0. The best and the full distribution are both in
  `results.json`.

## Representative media (original | replay | diff: red = missing white, blue = extra white)
- `runs/c7/c4_cgp/ba48_4500_60__k1_S2__G32__s3.best.gif`: 6 gates, coarse morph, then frozen.
- `runs/c7/c3_secondorder/ba48_4500_60.best.gif`: 1,466-condition tree, a drifting blob.
- `runs/c7/c2_lifelike/ba48_4500_60__S2.best.gif`: Life-like erosion.
- `runs/c7/c5_difflogic/ba48_4500_60__k1_S2__main/best.gif`: hold (collapsed).
- `runs/c7/c4_cgp/ba48_1800_60__k1_S2__G16__s3.best.gif`: transfer clip.
- `runs/c7/c3_secondorder/translate24.best.gif`: 1-condition exact translation.
- `runs/c7/c3_secondorder/ba32_3000_5.best.gif`: 3-condition growth rule.
- `runs/c7/c2_lifelike/life_b3s23__S1.best.gif`: control.
- Each also has a `.png` contact sheet.

**Metric limits:**
- The mean error rewards a frozen coarse shape, and change-IoU is near 0 for every method.
- The tolerant metrics can hide a systematic lag; the plain error is always reported with them.
- Horizon = 1 for most long-clip runs, because the CA changes faster than the video in the first
  frames.

## Costs
- **CPU:** **3.80 core-h** of the 12 ceiling. That is user+sys via `os.wait4`, 1 thread per job, wall
  recorded separately.

  | part | core-s |
  |---|---|
  | C2 (search 453, smoke 11) | 464 |
  | C3 | 24 |
  | C4 (main 10,585, smoke 56) | 10,641 |
  | follow-up (C2 142, C3 7, C4 2,404) | 2,553 |
  | summaries | < 1 |

  The CPU spent in the GPU jobs' host processes (hard rollouts on the CPU) is part of their gpuq
  wall time and is not in the CPU ledger.
- **GPU:** **0.44 GPU-h** of the 2 ceiling, via gpuq.

  | job | what | wall |
  |---|---|---|
  | 56 | control | 39 s |
  | 57 | **failed:** out of memory at compile (needed 141 GB for 4 seeds × 118 generations) | 18 s |
  | 58 | main, with per-generation gradient checkpointing added (`soft_rollout(remat=True)`) | 25.3 min |

  The OOM was fixed on the first retry.
- **Unspent:** about 8 core-h and 1.6 GPU-h. They were not spent to chase a result.

## Reproduction
- **Snapshots:**

  | snapshot | used for |
  |---|---|
  | `runs/frozen/6a4835116d1d` | C2 |
  | `runs/frozen/d289b47f7f4a` | C3, C4 |
  | `runs/frozen/69759ef82062` | C5 control and the OOM run |
  | `runs/frozen/8372981afd47` | C5 main, follow-up |

- **Commands:**
  - `python experiments/exp21_lifelike.py TARGET S`
  - `python experiments/exp22_secondorder.py TARGET`
  - `python experiments/exp23_cgp.py TARGET k1_S2 G SEED 600 c7/c4_cgp`
  - `gpuq -L … -G 1 python experiments/exp24_difflogic_long.py ba48_4500_60 k1_S2 1500 main`
  - each wrapped by `experiments/c7_meter.py` for CPU accounting.
- **Targets:** `ba48_4500_60` and `ba48_1800_60` come from the canonical mp4 via `ca/video.py` (box
  resize to 48 px wide, threshold 128).

## Limits of this phase
- One clip per role, single short budgets.
- C3 was tested only at S=1, which the light cone rules out for exact Bad Apple.
- C4 used only k=1, S=2.
- No learned hidden init, positional input or multi-scale structure was tried; all are discussed in
  the research document as separate settings.
- No UI.
