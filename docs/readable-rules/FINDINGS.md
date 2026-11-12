# Findings

## How numbers are defined
- **Grid and data:** 48×36 binary frames (threshold 128).
- **Error:** the fraction of wrong pixels over frames 1..T−1 of an autonomous replay of the exported file from the
  original frame at the window start. Hidden bits start at 0 unless stated.
- **Motion:** lag-4 change F1, with 1-pixel tolerance, and change mass (predicted / true changed pixels).
- **Seeds:** "best" is the best of 3 search seeds. The mean over the 3 seeds is in the json files.
- **CPU:** process CPU seconds.

## 1. Execution: the correct rule given
| check | result |
|---|---|
| Life B3/S23, shift, period-4 blink (1 hidden bit), excitable (1 hidden bit) run from their generator rule | exact |
| Python vs Node on random dlrule-v2 programs (k = 0, 1, 2; derived programs) and on all UI examples | bit-identical |
| UI cell inspector (fired rule, next state) vs the engine | identical at all sampled cell-substeps |

The pilot's "library" runs on the controls (error 0 for Life and shift) are executor checks. **They are not discovery**,
because the generator rule was among the starting candidates.

## 2. Blind recovery: the generator hidden from every method
Setup:
- **Generator runs:** 32×32 grid, 24 frames, S = 1.
- **Training run:** the seed-0 initial state.
- **Held-out runs:** two random initial states never seen in training.

Columns: error on the training run; error on the two held-out runs; conditions; CPU.

| control (generator size) | rollout search from the empty program (8,000 evals) | one-step decision list (k=0) | exact SAT (3×3 + own hidden bits) → decision list |
|---|---|---|---|
| shift (2 rules) | **0 / 0, 0** (1 condition, 18 s) | 0 / 0, 0 (2) | 0 / 0, 0 (2), k = 0 |
| Life (3 rules, 6 conds) | 0.221 / 0.214, 0.159: **fails** | **0 / 0, 0** (3 conditions = B3/S23) | 0 / 0, 0 (3), k = 0 |
| blink (needs memory) | 0.374 / 0.39: fails | 0.50: cannot | **k = 0 UNSAT**; k = 1: 0 / 0.017, 0 (8 rules, 24 conditions) |
| excitable (needs memory) | 0.012 / 0.037, 0.044: stuck | 0.10: cannot | **k = 0 UNSAT**; k = 1: 0 / 0, 0.0001 (5 rules, 12 conditions) |

Notes:
- **Expressible vs found:** every generator is expressible in dlrule-v2, by construction (it is written in it).
  - Life is therefore a **search failure** of rollout search: chaotic rollouts give no gradient toward B3/S23. A
    one-step fit recovers it exactly, because Life's state is fully visible.
  - For blink and excitable, **UNSAT at k = 0 proves memory is necessary** (3×3 visible, S = 1, zero boundary).
  - With k = 1, SAT finds a rule that is exact on the training run. Rollout search does not.
- **Behaviourally equivalent rules:** SAT's rules use their own hidden encoding.
  - The excitable rule stores "has been reached" and reads neighbours' hidden bits (`m0`); the generator stores
    refractoriness instead. It is exact on training and agrees on held-out starts to 1e-4, but it is not identical.
  - The raw SAT lookup tables generalise worse than the decision lists made from them (blink 0.176 vs 0.017 on one
    held-out start), because unseen table entries default to 0.
- Recovered example (excitable, k = 1, found without the generator):
  ```
  R0: if m0 <= 0 and n8 <= 0 -> v=0, h=0
  R1: if h0 == 1 and m0 >= 1 -> v=0, h=1
  R2: if h0 == 0 and m0 >= 1 -> v=1, h=1
  R3: if m0 <= 0 and n8 >= 1 and v == 0 -> v=1, h=1
  R4: if m0 <= 0 and n8 >= 1 and v == 1 -> v=0, h=1
  ```

## 3. Synthesis on Bad Apple
### 3a. 16-frame clips: one fixed rule, no schedule, hidden 0
The search starts from the empty program ("blind") or from a one-step decision list ("tree"); there is no library. The
same rule is then run unchanged from the original frame of the next two 16-frame clips, which it never saw.

| clip | hold | best autonomous baseline | blind rule: error · F1 · mass · conditions | on the next clip (hold) | on the clip after (hold) |
|---|---|---|---|---|---|
| 4500–4515 | 0.090 | majority 0.084 | **0.047 · 0.60 · 0.54 · 5** | 0.072 (0.072) | 0.079 (0.053) |
| 1221–1236 | 0.301 | hold 0.301 | **0.085 · 0.72 · 0.63 · 11** (S = 2) | 0.078 (0.131) | 0.409 (0.391) |
| 1241–1256 | 0.068 | hold 0.068 | 0.045 · 0.51 · 0.81 · 5 | 0.574 (0.566) | 0.111 (0.087) |

The rule found blind on 4500–4515 is the most readable result (5 conditions, error about half of hold's, real motion).
It reads as: a cell turns on when the cell above is on and the neighbourhood is at least half full; it turns off in
sparse areas or when its east column is full.
```
R0: if u == 1 and n8 >= 4 -> v=1
R1: if n8 <= 3 and E <= 1 -> v=0
R2: if E >= 3 -> v=0
default: keep state
```
- **On the clip itself:** these rules beat every autonomous baseline and move. This meets the stage criterion on short
  clips.
- **Off the clip:** they **do not generalise**. On unseen clips they are near hold, or worse. In 1 of 6 cases they beat
  hold (1221: 0.078 vs 0.131).
- **Conclusion:** they are short clip-specific programs, not motion laws.

### 3b. 32–60-frame tasks: fixed vs phased
Each cell gives error, lag-4 F1, conditions and compressed bytes. Phased cells also give edits and schedule bytes.

| task | hold | Neural CA (whole-song export, its own replay) | fixed K = 1 | phased K = 4 | phased K = 8 |
|---|---|---|---|---|---|
| translate 1221:60 | 0.381 | 0.017 | 0.202 · 0.26 · 8 c · 253 B | 0.136 · 0.32 · 8 c, 8 edits · 136 B sched · 365 B | 0.113 · 0.46 · 15 c, 17 edits · 270 B sched · 469 B |
| contour 4500:56 | 0.183 | 0.013 | 0.104 · 0.23 · 6 c · 232 B | 0.068 · 0.45 · 38 c · 477 B | 0.059 · 0.54 · 43 c, 25 edits · 579 B |
| wipe 1499:32 | 0.279 | 0.102 | 0.225 · 0.48 · 9 c · 188 B | – | 0.175 · 0.47 · 38 c · 535 B |
| cut 1678:32 | 0.603 | 0.027 | 0.371 · 0.33 · 3 c · 277 B | – | 0.198 · 0.58 · 30 c · 579 B |

- **Context for the Neural CA column:** the NCA is about 4.2 MB with 110 phases, and starts these windows from its own
  carried state rather than from zero hidden.
- **Fixed rules over 32–60 frames:** they beat hold but do not move (F1 ≤ 0.48).
- **Schedules:** they halve the error and raise F1, at the cost of 5–7× the conditions and an external timeline.
- **One-step fit is not rollout:** the fixed decision lists predict the next frame well when given the true current frame
  (one-step error 0.026–0.074), but drift in autonomous rollout (0.104–0.254).

### 3c. Path B: extraction from the Neural CA (4500:56)
| abstraction (k bits) | collision rate | one-step agreement | autonomous error (hold 0.183) |
|---|---|---|---|
| PCA signs, k = 1–3 | 0.84 / 0.79 / 0.72 | 0.99 | 0.182–0.183 (= hold: no rule fires a change) |
| k-means, k = 1–3 | 0.82 / 0.76 / 0.66 | 0.99 | 0.183 |
| change-balanced, k = 2 (PCA / k-means) | 0.79 / 0.76 | 0.83 / 0.82 | 0.199 / 0.450 (worse than hold) |

- **Collision rate:** the share of samples whose abstract input occurs with more than one successor.
- **High one-step agreement is a trap:** almost no cell changes, so "keep" is right 99 % of the time.
- **What the data says:** quantising the teacher's 15 hidden channels to 1–3 bits destroys the information that
  decides changes.

## 4. Memory and collisions on Bad Apple (heuristic diagnostic)
- **What was measured:** the minimum one-step error of **any** deterministic rule whose inputs are the visible patch
  plus the cell's own last j values (j memory bits used as a shift register). Computed exactly for that representation
  as the sum of minority counts per identical input.
- **Limits:**
  - This bound is about one step only. It does not bound an autonomous CA with other kinds of hidden state.
  - With 5×5 patches and memory, many inputs are unique, so the bound is optimistic.
  - This is a small predictive-state style check in the spirit of local causal states. It does not implement that
    theory.

| task | 3×3, k = 0 → 3 | 5×5, k = 0 → 3 |
|---|---|---|
| translate | 0.048 → 0.044 | 0.038 → 0.032 |
| contour | 0.020 → 0.019 | 0.014 → 0.012 |
| wipe | 0.123 → 0.105 | 0.084 → 0.057 |
| cut | 0.057 → 0.055 | 0.038 → 0.034 |

- **Reading:** own-history memory barely helps, except on the wipe. Radius matters more.
- **The irreducible part is visible-state ambiguity:** the same local picture is followed by different pictures.
- **Synthetic controls:** there, memory was provably necessary (blink and excitable are UNSAT at k = 0).

## 5. Locality limits
**Proof (`ca.sat.light_cone_conflicts`).** It applies to any rule of radius r with S substeps, so reach R = r·S per
frame, with **any** hidden state and a constant hidden start. Frame t is a function of the radius-tR patch of frame 0.
If two cells have identical patches but different values at frame t, no such CA is exact.

Below is the first conflicting frame at each reach R. "–" means no conflict, which is necessary but not sufficient.

| task | R = 1 | R = 2 | R = 3 | R = 4 |
|---|---|---|---|---|
| translate | frame 1 (183 cells) | frame 1 (68) | frame 1 (8) | – |
| contour | frame 1 (31) | frame 1 (6) | – | – |
| wipe | frame 1 (12) | frame 1 (2) | frame 1 (1) | – |
| cut | frame 1 (30) | – | – | – |

What this proves:
- **Reach 2 cannot be exact:** dlrule with S = 1 (reach 2, because of `r2`) cannot be exact from a zero hidden start on
  translate, contour or wipe. The pilot's "tree" runs used S = 1.
- **The pilot's other runs:** the "library" runs (S = 2, reach 4) are not excluded.

**Heuristic:** the share of pixels that newly turn on farther than R from every on pixel of the previous frame:
- translate: 30 % at R = 1, 7.4 % at R = 2, 0.6 % at R = 4;
- wipe: 38 %, 13 %, 3.1 %;
- cut: 40 %, 26 %, 12 % (3.1 % at R = 8).

Cuts and fast wipes need either a long reach or pre-arranged hidden state. **No impossibility is claimed** beyond the
proof above.

## 6. Literature: what transfers
| work | verified scope | what BadCA uses / cannot use |
|---|---|---|
| AutomataGPT, Adv. Sci. 2026 ([10.1002/advs.202511352](https://doi.org/10.1002/advs.202511352)) | Transformer trained on about 1 M trajectories of 100 binary 2D radius-1 Moore rules on a torus. On unseen rules from the same family: 98.5 % perfect one-step forecasts, up to 96 % functional and 82 % exact rule recovery. Degrades with nonlocal interactions. | Uses the blind-recovery protocol (rule hidden, functional vs exact match), as in §2. Not applicable: inference within a fixed known class; Bad Apple has no generating rule in any class. |
| Elser, PRE 104, 034301 (2021) ([10.1103/PhysRevE.104.034301](https://doi.org/10.1103/PhysRevE.104.034301)) | Constraint projection that reconstructs 1D binary rules and the unseen intermediate states from observations at nonconsecutive times. Tested up to n = 6 inputs. | The same idea as our SAT encoding with unobserved substeps and hidden bits (§2 used S = 1). 2D video scale is untested there. |
| Rupe & Crutchfield, Chaos 28, 075312 (2018) ([10.1063/1.5021130](https://doi.org/10.1063/1.5021130)) | Local causal states (classes of past light cones with the same predictive future). Coherent structures in elementary CA. | §4 is only a light-cone collision count with own-history memory. A real causal-state reconstruction on video is not attempted. |
| Richardson et al., PLOS Comput. Biol. 2024 ([10.1371/journal.pcbi.1011589](https://doi.org/10.1371/journal.pcbi.1011589)) | NCAs trained on time series of images: Turing patterns, morphing between images. | Same family as our Neural CA (continuous, many parameters). It does not give readable rules. |

No novelty is claimed. This track only tests feasibility and simplicity.

## 7. Cost
- **CPU (`runs/readable_rules/cpu_ledger.jsonl`):** 1.92 core-h in total.
  - Rollout search pilots: 0.61 (library start) + 0.18 ("tree" start) + 0.19 (short clips).
  - Extraction: 0.18. Diagnostics: 0.08. Controls: 0.08.
  - Second round (§8): stability 0.43, Life landscape 0.12, SAT ladder 0.05, light cone < 0.01.
- **GPU:** 0.
- **Remaining:** 24 GPU-h and 62.1 CPU core-h.

## 8. Why search fails, and what SAT can prove (second round, 2026-11-12)
Script: `experiments/rr_next.py`; results in `runs/readable_rules/next/`.

### 8a. Life: the rollout objective is deceptive, not just under-budgeted
Reference: exact B3/S23 (error 0). Objective `h23` = full 23-step rollout error from seed 0; `h1` = one-step error.
- **Landscape:** hold 0.395; random programs mean 0.504 (10th percentile 0.237).
  Single edits of the exact rule: 9 are equivalent (error 0); the others score mostly 0.23–0.26 on `h23`
  (some 0.39–0.56) but 0.006–0.66 on `h1`.
- **Paths from hold to exact** (adding the three Life rules in each of the 6 orders): on `h23`, 4 of 6 orders get
  *worse* at an intermediate step (e.g. 0.395 → 0.578 → 0.56 → 0). On `h1` all 6 decrease monotonically.
- **Search from neighbours:** only 11 of 33 single-edit neighbours (9 of them equivalent) return to exact within
  2,000 evaluations.
- **Search from empty:** 2k evals 0.233 and 8k evals 0.231 (3 of 3 seeds); 32k evals reach exact in 2 of 3 seeds
  (75 CPU-s each).
- **Conclusion:** both. Near-miss rules sit on the same ~0.23 plateau as wrong rules, so the rollout objective gives
  no gradient toward the answer (structural). A large enough budget sometimes crosses it (budget). The one-step
  objective has no such plateau, which is why the one-step fit and SAT recover Life directly (§2).

### 8b. SAT ladder on Bad Apple crops
Setup: translate (1221) and contour (4500); crops 8×8 and 16×16 centred on the most-changing area, and the full
48×36 frame; families `tot` (own v, number of on Moore neighbours, own hidden: at most 2·9·2^k entries per output)
and `full` (any 3×3 + own hidden); k = 0, 1, 2 hidden bits; S = 1, 2 substeps; T = 2, 3, 4, 6, 8, 12, 16 frames,
stopping at the first non-SAT; 60 s per solve, 6 M clause cap. Hidden bits start at 0; the boundary is zero
(artificial at crop edges).

| | result |
|---|---|
| full frame, both tasks, all families, k, S | UNSAT at T = 2 (3 cases SIZE: over the clause cap) |
| 16×16 crops, all settings | UNSAT at T = 2 |
| 8×8 translate, all settings | UNSAT at T = 2 |
| 8×8 contour, S = 1 | UNSAT at T = 2 |
| 8×8 contour, S = 2 | `full` k=0: SAT to T = 3; k=1, 2: SAT to T = 4; `tot` k=2: SAT to T = 4 (186 entries), UNSAT at 6 |
| 16-frame replay of the longest SAT rules | 0.27–0.41 vs hold 0.36 |

- **No TIMEOUTs.** Every stop is a proof (or the size cap), not a solver limit.
- **Scope of every UNSAT:** impossible *for this rule family, radius 1 per substep, this S, hidden bits starting at
  zero, a zero boundary at the crop edge, and this crop*. It says nothing about a learned hidden start or a wider
  neighbourhood.
- **Why T = 2 already fails:** with hidden bits starting at 0 the first transition depends on the visible patch only,
  so k cannot help, and the light cone (8c) gives a direct witness. At crop edges, content entering from outside is
  a second cause.
- **Short exact rules do not generalise:** the T = 4 solutions replay 16 frames at or below hold quality.

### 8c. Light cone: premises, scope, witnesses
**Statement.** For any homogeneous, deterministic, synchronous CA with radius r per substep and S substeps per frame
(reach R = r·S per frame), any number of hidden channels **whose initial value is the same in every cell**, and a
zero boundary: the state of a cell at frame t is a function of the frame-0 patch of radius t·R around it. So if two
cells have identical radius-R patches at frame 0 but different values at frame 1, **no** such rule replays the clip
exactly.

- **Proof:** induction on substeps; nothing beyond the premises is used.
- **Not covered:** a learned or otherwise non-constant hidden start (the Neural CA has one: 1,620 values), a
  non-zero or data-driven boundary, non-homogeneous or stochastic rules.
- **R for `dlrule`:** `r2` reads radius 2, so R = 2·S (R = 2 at S = 1, R = 4 at S = 2). The SAT families read 3×3,
  so R = S.

| task | R = 1 witness (cell off / cell on, same patch) | R = 2 witness | R = 3 | R = 4 |
|---|---|---|---|---|
| translate 1221:60 | (3,18) / (7,11), all-zero patch | (5,19) / (8,14), all-zero 5×5 | (8,15) / (10,18) | none |
| contour 4500:56 | (18,12) / (7,17) | (27,7) / (7,17) | none | none |
| wipe 1499:32 | (14,44) / (17,47) | (19,28) / (19,30) | (19,29) / (19,30) | none |
| cut 1678:32 | (0,24) / (29,7) | none | none | none |

- **Empirical check (not part of the proof):** 0 of 300 random radius-2, S = 1 `dlrule` programs separate the R = 2
  witness pair, as the proof requires.
- **Consequence:** radius 2 with S = 1 (R = 2) is provably inexact on translate, contour and wipe; with S = 2 (R = 4)
  no witness exists, so the proof does not rule it out (nor does it show it possible).

### 8d. Autonomous-replay stability
One fixed rule (k = 0, S = 2, empty start, 4,000 evaluations, 3 seeds) fitted on 16, 32 and 64 frames of 1221 and
4500. Objectives: `full` (rollout error) and `mixed` (rollout plus one-step error). Rules are chosen on the training
window only; next-clip numbers are reported, never used for selection.

- **Where the gap opens:** the rollout error exceeds the one-step error by 0.05 from frame 2 on 1221 and from frames
  4–19 on 4500.
- **Training-window error by length** (hold in brackets):
  - 1221: 16 frames 0.092–0.106; 32 frames 0.087–0.094; 64 frames 0.28–0.36 (0.40), lag-4 F1 ≤ 0.29.
  - 4500: 16 frames 0.056–0.08; 32 frames 0.08–0.105; 64 frames 0.09–0.14 (0.187), lag-4 F1 ≤ 0.25.
- **Mixed objective:** slightly lower one-step error, no consistent gain on rollout error or F1
  (1221:16 mixed 0.092–0.101 vs full 0.096–0.106; worse on 4500:32).
- **Next clip (reported only):** mostly at or worse than hold; 1221:16 seed 0 is the exception (0.077 vs 0.131).

### 8e. Most Conway-like result so far
Still the blind 16-frame rule on 4500–4515 (§3a): `R0: if u == 1 and n8 >= 4 -> v=1; R1: if n8 <= 3 and E <= 1 ->
v=0; R2: if E >= 3 -> v=0` (5 conditions, error 0.047 vs hold 0.090), replayable in the UI rule lab. Nothing in this
round beat it.

## 9. Synthesis and scaling (third round, 2026-11-12)
Script: `experiments/rr_scale.py`; results in `runs/readable_rules/scale/` (`a1_controls.json`, `a1_badapple.json`,
`a2.json`, `c1.json`, `c1/`, `ac.json`, `ac/`); logs `runs/readable_rules/scale_*.log`. All CPU, no GPU.
"Seed" below always means the **learned initial hidden state** (k bits per cell at frame 0), stored in the export
and counted in bits; it never contains frames. Errors are pixel error over the replayed window; hold = repeat frame 0.

### 9a. A1: learned initial hidden state (SAT)
**Controls (Gate 1).** Generators `sprout` (k = 1: `h0 == 1 -> on; v == 0 and n8 >= 3 -> on`) and `timer` (k = 2).
Trained jointly on 4 runs (32×32, T = 12), tested held-out on 3 new runs (16×16, T = 10) with the whole rule pinned
and only the seed free.
- Coarse seeds (const, tiles, blocks) are UNSAT on both generators: they need a per-cell seed, as generated.
- `sprout`, `tot` family: held-out 3/3 exact, with and without Occam MaxSAT. Occam (minimise non-keep entries and
  seed ones) raises table agreement with the generator from 0.57 to 0.87 and cuts seed ones from 264 to 160.
- `sprout`, `full` family: fails held-out (unseen entries default to keep and are wrong).
- `timer`: fails held-out in every setting. A1 recovers a seed-driven rule only when its family is small.

**Bad Apple ladder.** translate 1221 and contour 4500; crops 8×8, 16×16 (most-changing area) and the full 48×36 frame;
`tot`/`full`; k = 1–3; S = 1, 2, 4; T = 2, 3, 4, 6, 8, 12, 16; seed representations from 1 bit (const) through
tiles and blocks to one value per cell; the smallest SAT representation is kept per T; Occam 20 s.

| status | count | meaning |
|---|---|---|
| SAT | 230 | exact on all T frames |
| UNSAT | 342 | proof: no rule of this family and seed representation exists |
| SIZE | 369 | encoding above the 6 M clause cap (mostly S ≥ 2 on 16×16 and the full frame) |
| TIMEOUT | 13 ladders | see the defect below; no row recorded |

**Defect (recorded, not hidden).** The 30 s SAT limit never fired: pysat's CaDiCaL 1.9.5 does not support
interrupts (46 `NotImplementedError` in the log). One solve ran 14,545 s. The meter killed the parent at 2 h, and the
pool workers ran on unmetered for about 4.4 h until stopped by hand. Thirteen of 108 ladders (mostly 8×8 at S = 4)
never finished. They are counted as TIMEOUT. The other 95 ladders finished; their UNSAT and SAT results are unaffected,
since SAT/UNSAT does not depend on the solver. Fixes:
- `solve` now uses Glucose4, which honours the timeout (checked: 5 s → TIMEOUT).
- `c9_meter.py` now kills the whole process group.

**What a learned seed buys:**
- **Feasibility:**
  - every zero-hidden setting was UNSAT (§8b);
  - with a seed, all three crop sizes reach SAT;
  - 8×8 and 16×16 contour are exact to T = 16;
  - 8×8 translate is exact to T = 16; 16×16 translate to T = 8; the full frame to T = 12 (contour) and T = 4 (translate).
- **Smallest seed per length:**
  - 8×8 crops: 1 bit (const, S = 4) up to T = 12. At T = 16: contour 4 bits (tile2, k1 S2), translate 9 bits (block3).
  - Full frame: one value per cell (3,456–5,184 bits) from T = 3; T = 2 needs 12 bits (block12).
- **But the information moves into the table:**

  | fit | seed bits | table bits | total | clip size |
  |---|---|---|---|---|
  | 8×8 contour, T = 16 | 4 | 545 entries × 2 bits = 1,090 | ≈ 1,094 | 960 raw |
  | full-frame contour, T = 12 | 5,184 | 630 × 4 = 2,520 | ≈ 7,700 | 19,008 raw; 3,056 zlib |

  The table bits are `entries_used × (k+1)`. Every exact fit costs as much information as the clip or more. These are
  look-up tables, not readable rules.
- **Out of window:**
  - only 46 of 224 SAT fits beat hold on the frames after their training window;
  - example: full-frame T = 12, 0.083 vs hold 0.035;
  - zeroing the seed breaks most fits (e.g. 0.143 for full-frame T = 12; 0.44–0.50 for the 8×8 T = 16 fits).

  The seed carries the replay, not the rule.

### 9b. A2: two-stage search vs rollout-only
- **Methods:**
  - `rollout_only`;
  - two-stage: 4 one-step climbs, pick by a short rollout, then refine with one of:
    - `full`: rollout error;
    - `staged`: short then full;
    - `mixed`.
- **Setup:** 4,000 evaluations; 3 seeds.
- **Windows:** separate search / select / final windows, no overlap. The final window is never used for selection.
  No library rules enter blind results.

| task | result |
|---|---|
| Life (B3/S23 from empty) | `staged` exact in 3/3 seeds and exact held-out; `rollout_only`, `full`, `mixed` stay at 0.233 (the §8a plateau) |
| shift | all methods exact |
| Bad Apple, 6 starts × 16/32 frames, training | err/hold 0.44–0.65 for every method; no clear winner |
| Bad Apple, final unseen window, 16 frames | beats hold: `rollout_only` 0/6, `full` 1/6, `staged` 0/6, `mixed` 0/6; mean err 0.126–0.141 vs hold 0.107; lag-4 F1 ≤ 0.28 |
| Bad Apple, final unseen window, 32 frames | 0–1 of 6 beat hold for every method; err 0.20–0.25 vs hold 0.111 |

- **Life:** the Life problem is the deceptive rollout objective, and `staged` (one-step first) fixes it.
- **Bad Apple:** a different problem shows up. Training fits are similar for every method, and none transfers to
  the next unseen window. Search is not the bottleneck there; generalisation is.
- **Adopted for C1/AC:** `staged`.

### 9c. C1: shared phased rules (training window, state carried across phases, no resets)
- **Phases:** K equal-length phases with an external schedule.
- **Modes:**
  - `indep`: one program per phase;
  - `shared`: base + edits, 0.002 penalty per edit;
  - `reuse`: a phase may reuse an earlier program if within 0.003.
- **Setup:** 2 seeds.
- **Table columns:** error (seed 0 / seed 1); distinct programs; conditions; file bytes including the schedule.
- **Schedule bytes:** 35 (K = 1), 137–141 (K = 4), 275–281 (K = 8), 547–563 (K = 16).

| task (hold) | K = 1 | K = 4 | K = 8 | K = 16 |
|---|---|---|---|---|
| 1221:120 (0.552) | 0.166, 1 prog, 2 cond | indep 0.101/0.107 | 0.100 | indep 0.080, 8 progs, 17–18 cond, 424–428 B; **reuse 0.084/0.080, 5–6 progs, 8–12 cond, 390–450 B** |
| 4500:120 (0.188) | 0.146/0.141 | 0.128–0.140 | 0.152–0.165 (worse than K = 4) | 0.138–0.150, 7–11 progs, 21–28 cond |
| 1760:240 (0.401) | 0.282 | 0.282 | 0.282 | 0.282 in 5 of 6 runs; seed 1 shared/reuse 0.147/0.149 with 15–16 progs, 65–67 cond, 704–714 B |

- **Worst 16-frame window:**
  - 1221:120 K = 16: 0.193;
  - 1760:240 at 0.282: 0.436, a collapse with lag-4 F1 0.03 (the rule kills motion and the phases cannot recover it).
- **Lag-4 F1:** best 0.37–0.39 (1221:120 K = 16); 4500:120 ≤ 0.26.
- **Shared/reuse vs independent:**
  - same error;
  - fewer distinct programs and conditions on 1221;
  - no gain on 4500.
- **Simplification:** where claimed, it is net of the schedule. 1221:120 K = 16 reuse is 390 B in total against 424 B
  independent.
- **Caveats:**
  - all C1 numbers are training-window numbers;
  - a phased rule with an external schedule has no meaning on an unseen clip, so none is reported;
  - one of the two 1760:240 runs found a working solution and the other collapsed (search variance).

### 9d. AC: joint rule + seed (training window)
- **Seed:** learned, 6×6 blocks, k = 1 (48 bits).
- **Order:** fixed + zero; fixed + learned; phased K = 8 + zero; phased K = 8 + learned.
- **Setup:** 2 seeds.
- **"zeroed":** the same export replayed with the seed set to 0.

| task (hold) | fixed, zero | fixed, learned (zeroed) | K = 8, zero | K = 8, learned (zeroed) |
|---|---|---|---|---|
| 1221:60 (0.381) | 0.331/0.330 | 0.330/0.242 (0.331/0.245) | 0.118/0.119, F1 0.42–0.44 | 0.118/0.115 (0.125/0.115) |
| 4500:56 (0.183) | 0.133/**0.091** | 0.097/0.129 (0.159/0.183) | 0.072/0.072 | 0.065/0.074 (0.098/0.101) |
| 1760:120 (0.319) | 0.194/0.195 | 0.195/0.191 (0.195/0.195) | 0.194/0.140 | 0.195/0.132 (0.195/0.145) |

- **Phasing:** K = 8 cuts error by 0.06–0.21 on 1221 and 4500.
- **The 48-bit seed:**
  - changes error by ≤ 0.01 when phased;
  - on 4500 it matters: zeroing it costs 0.03–0.06;
  - but there it does not beat the best zero-seed fixed rule.
- **Simplicity:** phased rules cost 7–40 conditions and 270 B of schedule.
- **No result beats its zero-seed counterpart by more than seed-run variance**, except 1221:60 fixed seed 1
  (0.242 vs 0.33). There the seed-zeroed replay is still 0.245, so the gain came from the different rule search,
  not the seed.

### 9e. Gates
| gate | status |
|---|---|
| 1. synthetic discovery, held-out | partly: Life (`staged`), shift, `sprout` pass; `timer` fails |
| 2. 16–32 frames, beating baselines with real motion | training windows yes; **unseen windows no** (A2: 0–1 of 6 beat hold) |
| 3. 60–120 frames | 1221:60/120 and 4500:56 phased beat hold clearly (training); 4500:120 barely |
| 4. 240+ frames, one export | 1 of 2 seeds (0.147 vs hold 0.401, 16 programs, 67 conditions) |
| 5. whole song | not started: Gate 2 fails on unseen windows and Gate 4 is one seed |

For comparison: the Neural CA replays the whole song at 0.0152.

### 9f. Best readable rules from this round
- **One fixed rule, no seed, no schedule:** 4500:56, 56 frames, error 0.091 vs hold 0.183, lag-4 F1 0.15
  (`runs/readable_rules/scale/ac/4500_56_K1_zero_s1.rule.json`). Five rules, 8 conditions, k = 1, S = 2:

  ```
  R0: if S >= 2 and m0 == 4 -> v=1, h=1
  R1: if W >= 3 and d >= 1  -> v=1, h=0
  R2: if N == 1             -> v=0, h=1
  R3: if n8 <= 2            -> v=0, h=0
  R4: if S <= 2 and W >= 2  -> v=1, h=1
  ```

  It covers 3.5× the frames of the §8e rule at about double the error. Training window only.
- **Phased:** 1221:120 K = 16 reuse, 0.080–0.084 vs hold 0.552, 5–6 distinct programs, 8–12 conditions, 547 B
  schedule. It is in the UI as `shared_rules_reused_frames_1221-1340`.

### 9g. Cost of this round
| method | CPU core-h |
|---|---|
| A1 controls | 0.12 |
| A2 | 0.86 |
| C1 | 4.22 |
| AC | 1.11 |
| A1 Bad Apple | 48.4–59.4 (59.4 booked; see the defect in 9a) |
| **this round** | **≤ 65.7** |

GPU: 0.
