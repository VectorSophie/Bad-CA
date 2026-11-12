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
