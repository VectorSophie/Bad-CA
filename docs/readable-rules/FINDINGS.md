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

## 10. Generalisation and dynamics (fourth round, 2026-11-12)
Script: `experiments/rr_gen.py`; results in `runs/readable_rules/gen/` (`prereg_windows.json`, `lut.jsonl`,
`g1.jsonl`, `g2.jsonl`, `g3.jsonl`, `g3/`, `g4.jsonl`); logs `runs/readable_rules/gen_*.log`. CPU only.

**Pre-flight (before any fit).** `tests/test_meter.py` checks three things:
- SAT wall-time limit: a 3 s limit gives TIMEOUT after 4.5 s, encoding included;
- a wall-killed `Pool` run leaves no worker alive and books its CPU;
- SIGTERM to the meter does the same, and rows written before the kill are kept.

The meter now samples the CPU of every process in the run's session from `/proc`. The old `wait4` count would have
booked 0 CPU-s for the killed run; the sampled count is 16 CPU-s.

**Windows (fixed before fitting, `prereg_windows.json`).**
- **Scenes:** 24 scenes ≥ 80 frames between cut frames and wipe runs (`segment.signals`; a wipe run is ≥ 3 frames
  with change ≥ 0.08).
- **Windows per scene** (a = scene start, e = scene end):

  | window | frames |
  |---|---|
  | train | [a+2, a+18) |
  | adjacent | [a+18, a+34) |
  | distant | [a+50, a+66) |
  | cross | [e−8, e+8), straddling the cut or wipe |

- **Visual check.** Every scene was inspected on a contact sheet before any fit:
  - adjacent window: 17 same subject, 7 morph (no cut, but the subject changes shape or identity);
  - distant window: 13 same, 11 show different content although no cut or wipe was detected.

  The cut/wipe signals alone are not a reliable scene detector for Bad Apple, so groups below use the visual labels.
- **Methods:**
  - staged two-stage search, k = 0 and k = 1;
  - rollout-only, k = 0;
  - 3 seeds each, 4,000 evaluations, S = 2;
  - hidden state starts at 0.
- **Baselines:**
  - hold;
  - a shift baseline: the best constant integer shift (dy, dx) ∈ [−3, 3]² per frame, fitted one-step on the train window.

### 10a. Search-free bound: exact patch tables (`lut.jsonl`)
For radius R, the train window's majority table "radius-R patch at t → pixel at t+1" is the best any memoryless rule
with per-frame reach R can do on that window (S substeps of radius r give R = r·S). It is applied one step on each
unseen window; unseen patches keep their state. Errors are one-step errors, averaged over windows.

| R | train in-sample | adjacent (same): transfer / hold | distant (same): transfer / hold | cross: transfer / hold | patches unseen (adj.) |
|---|---|---|---|---|---|
| 1 | 0.0198 | 0.0207 / 0.0172 (3/17 beat hold) | 0.0308 / 0.0203 (1/13) | 0.0786 / 0.0721 (4/24) | 0.4 % |
| 2 | 0.0120 | 0.0220 / 0.0172 (4/17) | 0.0304 / 0.0203 (0/13) | 0.0779 / 0.0721 (2/24) | 7 % |
| 3 | 0.0062 | 0.0198 / 0.0172 (4/17) | 0.0244 / 0.0203 (1/13) | 0.0746 / 0.0721 (4/24) | 19 % |
| 4 | 0.0031 | 0.0185 / 0.0172 (5/17) | 0.0220 / 0.0203 (0/13) | 0.0725 / 0.0721 (6/24) | 32 % |

- **Each window has its own local law.** The in-window optimum on the unseen window is well below hold at every R
  (e.g. 0.0128 at R = 1, adjacent), but the training window's optimum does not carry over one step, even within the
  same scene.
- **Radius doesn't help.** A larger R fits the training window better but sees more unseen patches. It never turns
  transfer into a win.

  This bounds every memoryless search result below. Hidden state is the one thing the table does not cover.

### 10b. G1: window-to-window generalisation (`g1.jsonl`, 216 fits + 24 shift baselines)
**Training window.** Rollout error 0.086–0.092 vs hold 0.125; 69/72 runs per method beat hold. But the one-step
error (0.029–0.034) is *worse* than one-step hold (0.026) in 52–55 of 72 runs. The training win comes from matching
the net drift (e.g. a shape shrinking overall), not from predicting local motion.

**A. Transfer** (the rule unchanged, restarted from the unseen window's true first frame). The table gives mean error
vs hold, then runs beating hold (and runs beating it by more than 5 %), then lag-4 F1 and change mass.

| window group | staged k=0 | staged k=1 | rollout-only | shift |
|---|---|---|---|---|
| adjacent, same | 0.107 vs 0.085; 21/51 (12); F1 0.10, mass 0.42 | 0.103; 17/51 (9) | 0.123; 14/51 (11) | 0.100; 3/17 |
| adjacent, morph | 0.123 vs 0.127; 17/21 (13); F1 0.12, mass 0.14 | 15/21 | 16/21 | 0/7 |
| distant, same | 0.159 vs 0.101; **1/39** (0); F1 0.08 | 5/39 (1) | 2/39 (0) | 0/13 |
| distant, different | 0.209 vs 0.171; 11/33 (6) | 10/33 | 12/33 | 0/11 |
| cross | 0.275 vs 0.218; 17/72 (10); F1 0.11 | 19/72 | 17/72 | 0/24 |

- **Wins with real motion are rare:**
  - criterion: > 5 % better than hold, lag-4 F1 ≥ 0.2, change mass 0.3–3;
  - adjacent same-scene: 8 of 153 runs, from 3 windows (9, 16, 19), 5 of which also beat the shift baseline;
  - distant same-scene: **0 of 117**.
- **Trivial wins:**
  - most other "wins" are `if always -> v=0` (erase everything) on sparse scenes (starfield, a dot), F1 0.06, or
    erosion while a shape shrinks (the morph windows, change mass 0.14);
  - the all-erase rule on window 22 beats hold on the adjacent window (0.013 vs 0.023) and loses badly on the
    distant one (0.228 vs 0.155).
- **Same scene vs cut:**
  - per-window median improvement (seeds averaged), staged k = 0: adjacent same −8 %, distant same −22 %, cross −6 %;
  - permutation test same vs cross: p = 0.91 (adjacent) and 0.10 (distant);
  - rollout-only: distant same is significantly *worse* than cross (p = 0.006).

  **Same-scene windows are not easier.**

**B. Continuous** (from train[0], no injection, through all windows). The baseline is train[0] held, i.e. the same
information the rule has.
- **Adjacent same:** beats it in 36/51 (staged) and 40/51 (rollout-only), mean improvement 12–16 %.
- **Distant same:** 30/39 and 33/39, 13–20 %.
- **Cross:** 33–36 of 72, about 0 %.
- **But no motion survives:**
  - lag-4 F1 is 0.05–0.08 everywhere, and change mass 0.08–0.61;
  - the rollout beats a frozen frame by drifting toward the scene's average (shrinking or filling), not by moving
    with it;
  - the shift baseline is worse than the frozen frame (−2 % to −9 %).
- **Error take-off:** rollout error exceeds one-step error by 0.05 at a median of frame 4 (staged), 4.5 (k = 1) and
  5 (rollout-only); in 44–47 of 72 runs inside the 16 training frames; never in 6/72.

**Hidden state (k = 1, zero start):**
- adjacent transfer error −0.002 on average vs k = 0, better in 15/72;
- distant −0.007, better in 24/72;
- cross +0.009.

No consistent gain.

### 10c. G2: what the rules do (`g2.jsonl`, all 216 programs)
- **Rule kinds** (from the text):
  - other 201, grow (→ on when counts are high) 65, erode (→ off when counts are low) 63;
  - `always` 47, fill 7, kill 2.
- **Precision:** the share of a rule's changed pixels that match the video.
  - Every kind is mostly right on the training window: always 0.87, erode 0.66, grow 0.56, other 0.54.
  - On unseen windows, precision drops for every kind:
    - adjacent: always 0.40, erode 0.33, grow 0.31, other 0.25;
    - distant: 0.14–0.34.
- **The 32 adjacent same-scene wins (> 5 %):**
  - 25 contain an `always` rule;
  - they leave only 10 % of cells to the default "keep";
  - adjacent-window precision 0.60 (median), but distant 0.05.
- **The 121 failures:**
  - precision falls from 0.53 (train) to 0.14 (adjacent);
  - which rules fire barely changes between train and unseen windows (total-variation distance 0.075 adjacent,
    0.12 distant).

  The same rules fire in the same proportions but now make wrong changes. The local situations look alike; what
  follows them differs.
- **Ablation:**
  - in failures, removing a rule hurts the training window in 181/222 cases but *helps* the adjacent window in
    114/222;
  - in wins, removal helps adjacent in only 4/62.
- **Pruning:**
  - greedy condition removal (train error +0.001 allowed) drops 2.44 → 2.17 conditions on average and simplifies
    49/216 programs;
  - adjacent transfer is unchanged (0.1156 vs 0.1147).

  The rules are already near-minimal; their size is not what overfits.
- **Local statistics drift:**
  - measured as the total-variation distance between the (v, n8) histograms of rollout and video;
  - the rollout drifts slightly faster than holding the first frame: 0.030 vs 0.021 at frame 1, 0.102 vs 0.104 at
    frame 16, 0.308 vs 0.296 at frame 63;
  - the rollout is closer than hold in only 54–79 of 216 runs.

  The rules do not keep the video's local texture any better than a frozen frame.

### 10d. G3: phase placement (`g3.jsonl`, `g3/`)
**Setup:**
- Shared executor, export and `fit_clip` (staged base, `reuse` mode, edit penalty 0.002), S = 2, k = 0.
- K = 8 and 16; 3 seeds; the same search budget per phase for every placement.
- **Placements:**
  - **uniform:** equal lengths;
  - **cut-aligned:** boundaries at the clip's cut frames and wipe-run starts (`segment.signals`), strongest first;
    the longest phases are halved if there are fewer events than K − 1;
  - **optimized:** from cut-aligned, at most 4 greedy moves (±1/2/4/8 frames) or merges, each accepted only if it
    lowers the clip's MDL code length.
- **Data access:** every placement uses only the clip being fitted, which is also the scored clip. **These are fit
  (compression) results, not blind generalisation.**

Errors for seeds 0/1/2 (hold in brackets). "Progs" = distinct programs used; cond = conditions; file = total export
bytes including the schedule (275–283 B at K = 8, 547–567 B at K = 16).

| task | K | uniform | cut-aligned | optimized |
|---|---|---|---|---|
| 1221:120 (0.552), 1 hard cut | 8 | 0.101/0.100/0.097; F1 0.14–0.28; worst16 0.26–0.28; 3 progs; 5–11 cond; 323–385 B | **0.078/0.077/0.076; F1 0.47–0.52; worst16 0.130; 2 progs; 3–5 cond; 308–343 B** | same as cut |
| | 16 | 0.084/0.080/0.080; F1 0.32–0.42; 5–6 progs; 8–16 cond | 0.076 ×3; F1 0.51–0.52; 3 progs; 5 cond | same |
| 4500:120 (0.188), wipes only | 8 | 0.152/0.153/0.149 | 0.151/0.152/0.150 | 0.151/0.152/0.155 |
| | 16 | 0.138/0.149/0.145 | 0.145/0.143/0.144 | 0.145 ×3 |
| 2700:180 (0.555), cuts + wipes | 8 | **0.188/0.174/0.170** | 0.197/0.203/0.197 | 0.183/0.194/0.190 |
| | 16 | 0.155/0.181/0.180 | 0.173/0.180/0.175 | 0.180/0.181/0.184 |
| 1760:240 (0.401) | 8 | 0.282/0.282/0.280 (collapse) | same | same |
| | 16 | 0.282/**0.149/0.149** (15/12 progs, 65/51 cond) | 0.282/0.281/0.274 (collapse) | same |

- **Where cut alignment wins:** only where a single hard cut sits inside the clip (1221:120, cut at frame 42).
  - It wins on every axis there: error, F1, worst window, programs, conditions and bytes.
  - Cut-aligned K = 8 beats uniform K = 16: 0.077 with 2 programs and 3–5 conditions, vs 0.080–0.084 with 5–6
    programs and 8–16 conditions. **Fewer phases and rules for better motion.**
- **Elsewhere:**
  - with wipes only (4500) it is a tie;
  - on 2700 uniform is better by 0.01–0.03;
  - on 1760:240 cut alignment does not prevent the collapse, while 2 of 3 uniform K = 16 seeds escape it, at a cost
    of 51–65 conditions.
- **Paired over all 24 (task, K, seed):**
  - cut − uniform error: median −0.0005, mean +0.010, cut better in 12/24;
  - optimized: better in 10/24.
  - Both use fewer distinct programs (−1.5) and conditions (−5 to −6) on average, mostly because collapsed
    1760 runs use one program.
- **Optimized ≈ cut-aligned:** the MDL moves shift boundaries by a few frames and never merge (MDL always prefers
  more phases). The proxy does not track the fitted error.

### 10e. G4: radius-2 feasibility (`g4.jsonl`)
**Size estimate first.** The existing encoder enumerates unknown inputs: 2^(unknown bits) clauses per cell and
substep.
- A full radius-2 table at S = 2 has 25 unknown inputs at the intermediate substep, 2^25 ≈ 3.4·10^7 clauses per cell:
  **SIZE before encoding**, so it was not run.
- Two families are encodable:
  - **`full2`, S = 1:** any 5×5 pattern; every input is known, so one clause per cell. Reach R = 2.
  - **`tot2`:** a table over (v, n8, r2), 306 entries. Neighbour counts of unknown cells use exact unary counters,
    about 10^3 clauses per cell and frame; at most 3.1 M clauses here (cap 4 M).
- `tot1` (v, n8; radius 1) is the radius-1 comparison.
- Glucose4, 120 s per solve, zero boundary, k = 0, frame 0 given; T = 2, 3, 4, 6, 8, stopping at the first non-SAT.

**Rule-language match:**
- `tot1` and `tot2` are exactly `dlrule` restricted to {v, n8} and {v, n8, r2}, with no limit on the number of rules.
  All 35 such SAT models were exported as `dlrule` programs and replay exactly in the shared executor.
- `full2` has no `dlrule` equivalent: `dlrule` cannot name arbitrary 5×5 patterns.
- `dlrule` itself also has the directional counts (N, S, E, W, u, d, l, r), which these families lack. So a `tot2`
  UNSAT does **not** exclude `dlrule` programs that use direction.
- `tests/test_g4_sat.py` checks three properties:
  - soundness: a SAT model replays exactly;
  - completeness: the generator's table satisfies the encoding;
  - separation: radius 1 is UNSAT on a radius-2 generator.

**Status by family:**

| status | count |
|---|---|
| SAT | 47 |
| UNSAT | 47 |
| TIMEOUT | 4 (control1 T=2; 1678 and 4500 16×16 T=2; 4500 8×8 T=4) |
| SIZE | 0 run (full radius 2 at S = 2 is SIZE by estimate) |

**Last SAT T per family** (– = UNSAT at T = 2):

| task | crop | `tot1` S=2 | `tot2` S=1 | `full2` S=1 | **`tot2` S=2** |
|---|---|---|---|---|---|
| synthetic radius-2 rule (control0) | 16×16 | – | | | 8 (held-out error 0.19/0.13: table not unique, no Occam) |
| translate 1221 | 8 | – | – | – | **2** (137 rules), UNSAT at 3 |
| | 16 / full | – | – | – | – |
| contour 4500 | 8 | – | – | – | **3** (143 rules), TIMEOUT at 4 |
| | 16 | – | – | 2 | TIMEOUT at 2 |
| | full | – | – | – | – |
| cut 1678 | 8 | – | – | 2 | **3** (166 rules), UNSAT at 4 |
| | 16 | – | – | 2 | TIMEOUT |
| | full | – | – | 2 | – |
| wipe 1499 | 8 / 16 | 6 | 6 | 6 | 6 (all UNSAT at 8; the crop is a uniform fill) |
| | full | – | – | – | **2** (149 rules), UNSAT at 3 |

- **What radius 2 changes.** Radius 2 at S = 2 turns radius-1 UNSAT into SAT on 4 Bad Apple cases (8×8 translate,
  contour and cut; full-frame wipe). It gets no further than T = 3, and every solution is a 137–174-entry table, not
  a readable rule.
- **Light cone, cross-checked:** `full2` at S = 1 (R = 2) is UNSAT on the full frame for translate, contour and wipe,
  and SAT for the cut, exactly as the R = 2 witnesses of §8c predict.
- **Scope of the UNSAT results:** each holds for its grid (crop and zero boundary), hidden state starting at zero,
  k = 0, the given S, and the isotropic count family. It says nothing about direction-aware radius-2 rules or
  learned hidden starts.
- **Answer to "does radius 2 change the expressibility limit?":** only at T ≤ 3, on small crops, with tables. It does
  not change the generalisation picture: the R = 2–4 patch tables of §10a already fail to transfer.

### 10f. Answers and decision
1. **Do rules generalise within a scene?** No.
   - Transfer to the adjacent same-scene window: 21/51 staged runs beat hold; only 8/153 runs across all methods do
     so with real motion (F1 ≥ 0.2), from 3 windows.
   - Distant same-scene window: 1/39 and 0/117 with motion.
   - Same-scene windows are not easier than windows across a cut (p = 0.91, p = 0.10).
2. **Is failure due to error accumulation, overfitting or missing state?** Mostly overfitting to a window-specific
   local law; accumulation is the second effect.
   - On unseen windows the rules' one-step error is worse than one-step hold in 60/72 (staged), so they fail before
     any accumulation.
   - Even the best possible patch table at R = 1–4 does not transfer one step (§10a). No search method or radius can
     fix that.
   - Accumulation is also present: rollout leaves one-step behaviour at a median of frame 4.
   - One hidden bit gives no consistent gain. A learned hidden start was not re-tested (D8).
3. **Cut-aligned vs uniform phases?** Better only when a hard cut lies inside the clip (1221:120: error 0.077 vs
   0.099 at K = 8 with 2 programs instead of 3, F1 0.49 vs 0.20). Equal with wipes (4500), worse on 2700:180, no help
   against the 1760:240 collapse. Overall, cut better in 12/24 paired runs.
4. **Can a few shared rules explain several windows?**
   - Within one fitted clip, yes: 2–3 programs cover 1221:120 at K = 8–16 when phases follow the cut.
   - Across unseen windows, no: G1 rules do not carry over, and only 8 of 153 do with real motion.
5. **Does radius 2 change the limit?** Only for T ≤ 3 with large tables (§10e). No readable radius-2 rule was found.
6. **Grounds to extend to 120–240 frames?** None.

**Decision: NO-GO** under the brief's criteria (§8 of the 2026-11-12 brief).
- **GO is not met:** same-scene unseen windows are not beaten reliably, and motion F1 stays ≤ 0.10 on average.
- **PARTIAL is met on 1 of 4 tasks only:** cut alignment improves quality and size at equal K on 1221:120, but not on
  4500, 2700 or 1760, so it is not a general phase-placement effect.
- **What follows:** large-scale search stops. Readable Rules stays an experimental feature (Rule Lab) next to the
  Neural CA, which remains the featured demo.
- **Interpretation:** what the searches find are small programs that fit a particular window, mostly by matching its
  net drift. They are not reusable local laws of the video's motion.

### 10g. Cost of this round
| item | CPU core-h |
|---|---|
| LUT diagnostic | 0.003 |
| G1 | 0.78 |
| G2 | 0.006 |
| G3 (incl. one failed launch, 0.2 s) | 5.30 |
| G4 | 0.19 |
| unmetered pilots and tests | about 0.1 (estimate) |
| **this round** | **about 6.4** of the 24 allotted |

GPU: 0. Readable Rules cumulative: ≤ 74.0 of 128 core-h.
