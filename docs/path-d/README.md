# Path D: Differentiable Logic CA

**Goal:** learn genuinely discrete, spatially shared, local Boolean circuits that run a CA by themselves, then make
them small enough to read. Experimental track, separate from the featured Neural CA (unchanged) and from Readable
Rules (closed with NO-GO, D9; nothing there was rerun).

**Status, round 2 (D2.5 / D3), 2026-11-12: PARTIAL** (§8–§12).
- A circuit learned from the visible channel alone reproduces Greenberg–Hastings waves exactly on unseen trajectories
  (simplified to 10 gates).
- On real Bad Apple, a 64-gate circuit reproduces frames 3998–4006 by itself (error 0.016 vs hold 0.057) and beats
  every baseline on the next 16 unseen frames, but captures only 4–9 % of their true changes.

**Status, round 1, 2026-11-12:**

| gate | result |
|---|---|
| D0 reproduction | **passed** |
| D1 dynamics | **passed** (synthetic, full-state supervision) |
| D2 simplification | **passed**, with a qualified result |
| D3 Bad Apple | **not started, by decision** (§7) |

Sources and exactly what was inspected: [REFERENCES.md](REFERENCES.md). Inspector: `/static/logic.html` on the UI
server.

## 1. Code
| piece | file | role |
|---|---|---|
| model | `ca/dlca.py` | faithful JAX port of the reference notebook. Hardened circuits become a flat netlist (`badca/logic-ca-v1`), run by an independent NumPy executor. |
| JS engine | `player/logic_core.mjs`, `player/logic_replay.mjs` | independent Node/browser executor. Python and Node rollouts are bit-identical (SHA-256) on every export checked. |
| simplification | `ca/logic_simplify.py` | exact passes, ABC synthesis (through the existing `ca/reduce.py` wrapper), SAT miter equivalence, unit tying |
| experiments | `experiments/pd_d0.py`, `pd_d1.py`, `pd_d2.py`, `pd_examples.py` | |
| inspector | `ui/static/logic.html` (+ `logic_examples/`) | inputs of a cell, the gates feeding its outputs with their values, step and run; a semantic statement only when one was verified |
| tests | `tests/test_path_d.py` | gate semantics, exact simplification on random netlists, exact Life, JAX = NumPy = netlist, Python = Node |
| results | `runs/path_d/` | `d0/`, `d1/`, `d2/`, `gpu_ledger.jsonl`, `cpu_ledger.jsonl` |

**Export format `badca/logic-ca-v1`:**
- channels C and boundary (zero or periodic); synchronous update;
- the 16 gate truth tables;
- a netlist with inputs `p*C + c` (3×3 position p in row-major order, centre 4), gates `[op, a, b]` and one output node
  per channel;
- the initial state, as packed bits.

Nothing else is needed to replay. No target frame is involved.

## 2. D0: faithful reproduction (`runs/path_d/d0/`)
The notebook's own configurations: Life (16 kernels, update 16×128 + taper, 3×3 periodic boards of all 512
patterns, lr 0.05, batch 20) and the synchronous checkerboard (8 channels, 20 steps from noise, loss on channel 0
at the last step). Only the epoch counts were varied.

| task | epochs | exact seeds | notes |
|---|---|---|---|
| Life | 3,000 (notebook) | 1/5 (seed 1) | **512/512** neighbourhoods; 300 steps on an unseen 128×128 soup bit-identical to true Life; 333 active gates (paper: 336) |
| Life | 6,000 | +2 (seeds 23, 3) | 323 and 445 active gates; seeds 0 and 2 keep 6–27 hard errors with soft loss ≈ 0 |
| checkerboard | 500 (notebook) | 0/5 | soft loss still falling at 500 |
| checkerboard | 2,000 | 2/5 (seeds 1, 2) | 100/100 unseen 16×16 starts exact at step 20, still exact at steps 20–100; 64×64 for 80 steps: error 0.000 |

- **Soft vs hard:** every failure is a soft-to-hard gap. The soft loss reaches ≈ 0 while the hardened circuit is
  wrong (checkerboard seed 23: soft 0.0002, hard 159). The JAX hard path and the NumPy executor agree bit for bit in
  every run.
- **Against the paper:**
  - Life is reproduced, in 3 of 5 seeds.
  - The synchronous checkerboard is reproduced, in 2 of 5 seeds, but not at the notebook's 500 epochs. The
    independent `logicars` log reaches hard loss 0 at epoch 511, so this looks like a borderline budget, not a
    method failure.
  - After exact simplification our checkerboard circuits have **22 and 28 gates; the paper reports 22**.
  - Not reproduced: asynchronous training, the lizard, the multicolour G.
- **Diagnosis of the failures:** the implementation is not at fault (Life is bit-exact and executors agree), and the
  method works for some seeds. What remains is optimization, meaning seeds and epochs.

## 3. D1: long-horizon learning (`runs/path_d/d1/`)
Five synthetic tasks with known local generators, each run with three objectives and three seeds:
- **Tasks:**
  - translate (move east 1 px per step);
  - blink (oscillation: cells toggle under a hidden mask);
  - majority (3×3 vote: boundary growth and erosion);
  - gh (Greenberg–Hastings excitable medium: traveling waves);
  - propagate (directional: a cell turns on if the cell above is on).
- **Objectives:** A one-step, B 16-step rollout, C horizon curriculum 1→2→4→8→16.
- **Kept equal:** the architecture, the 16 training worlds, and a budget of 32,000 CA-step evaluations per run.
- **Supervision:** the full state, hidden channels included. One-step training needs it.
- **Evaluation:** 16 unseen 32×32 starts; also a 64×64 grid and double density.

Each cell gives runs exact over 128 steps from all 16 unseen starts, out of 3, with the mean active gates.

| task | A one-step | B rollout-16 | C curriculum |
|---|---|---|---|
| translate | 3/3 (144) | 3/3 (43) | 3/3 (139) |
| blink | 3/3 (249) | 3/3 (67) | 3/3 (244) |
| majority | **3/3** (296) | 0/3: collapses to hold (error 0.377) | 0/3 (error 0.06–0.13) |
| propagate | **3/3** (99) | 1/3 | 3/3 (102) |
| gh (waves) | **3/3** (236) | 0/3: noise (change mass 12–17, F1 0.03) | 2/3 |
| **total** | **15/15** | 10/15 | 11/15 |

- **Gate D1:** passed. Learned circuits produce non-trivial autonomous motion on unseen starts, exact for 128 steps,
  on a 4× larger grid and at double density. Examples: excitable waves, directional growth, translation.
- **The proposed curriculum does not help:** it beats rollout training but not plain one-step training (11/15 vs 15/15).
- **Rollout failures are optimization, not representation:** the same architecture solves every task one-step.
- **Training time** at equal budget: one-step 27 s, rollout 27–40 s, curriculum 70–110 s.
- **Soft vs hard:** at horizon 16, one-step circuits have large soft rollout error (gh 0.08–0.11, propagate up to
  0.43) yet hard error 0. Soft behaviour is not a guide to hard behaviour, in either direction.
- **Exact behaviour is not the generator's rule:** the exact one-step gh circuits still differ from the generator on
  5,044–6,050 of its 19,683 valid neighbourhoods. Those configurations never occur from the training family (see D2).
- **Caveat that decides D3:** every success here used full-state supervision. Bad Apple gives only the visible
  channel, and the only objective that does not need hidden targets (B) is the weakest one.

## 4. D2: simplification (`runs/path_d/d2/`)
All methods start from the same learned circuit.
- **A exact:** our exact passes (wire removal, constant propagation, equal-input and one-input folding, NOT
  absorption, hashing, dead gates) + ABC (`abc -g`) + the passes again. Every result is SAT-proved equivalent (miter
  over all 9C inputs).
- **B one-step pruning:** greedy unit tying (a gate tied to its majority value, most-constant first, then exact
  cleanup), kept if one-step outputs on the visited patches change on ≤ ε of them.
- **C multi-step pruning:** same candidates, kept if the visible rollouts from 16 selection starts stay within ε of
  the original's at every step up to 128.
- **C-task:** checkerboard only; kept if the target still forms by step 20 and holds to 128.
- **Scoring:** 64 unseen starts.

**Exact simplification (A) alone, from the hardened netlist:**

| circuit | raw gates | A gates (depth) | readable as |
|---|---|---|---|
| Life (D0 s1) | 2,543 | 57 (10) | B3/S23, exact |
| checkerboard (D0 s2 / s1) | 4,608 | 22 (6) / 28 (6) | – |
| translate | 1,375 | **0** | output = west neighbour (a wire) |
| propagate | 1,375 | 1 | `c OR N` |
| blink | 1,598 | 1 | `ch0 XOR ch1` |
| gh (one-step s0/s1/s2, curriculum s0) | 1,598 | 17 / 15 / 14 / 17 | – |
| majority | 1,375 | 83 (10) | – |

**Beyond exact (ε = 0 unless noted):**

| circuit | A | B ε=0 | C ε=0 | B ε=0.001 | C ε=0.001 | notes |
|---|---|---|---|---|---|---|
| Life | 57 | 53 | 53 | 49: divergence 0.19, F1 0.46 | 53, identical | the 53-gate circuits are still SAT-equivalent: an exact gain ABC missed, not an approximation. Life visits every neighbourhood, so there is nothing to drop. |
| majority | 83 | 64 | 64 | 32: divergence 0.15 | 64 | 64 still exactly equivalent. C ε=0.01: 55 gates, divergence 0.004. |
| gh curriculum s0 | 17 | 9 | 9 | 7: noise (mass 81) | 9 | **the 9-gate result equals the Greenberg–Hastings generator on all 262,144 neighbourhoods**; the learned circuit did not (65,025 entries differed) |
| gh one-step s0/s1/s2 | 17/15/14 | 11/11/12 | 11/11/12 | 9–10: divergence 0.2–0.3 | unchanged | identical on 64/64 unseen rollouts from the training family, but **fail with refractory cells in the start (error 0.41)**, where the generator-equal circuit stays exact |
| checkerboard s2 | 22 | 22 | 22 | 12 | 22 | B ε=0.01 and C-task both reach **6 gates**; target formed and held in 64/64 unseen starts |
| checkerboard s1 | 28 | 28 | 28 | 24 | 28 | **C-task 7 gates** (64/64) vs B ε=0.01 20 gates; the 7-gate circuit disagrees with the original on 50 % of visited one-step inputs yet keeps the behaviour |

**Answer to "does behaviour-aware scoring beat ordinary simplification?":**
- **Exact simplification does most of the work:** 1,375–4,608 gates go to 0–83.
- **Strict pruning (ε = 0) gains only through unreachable states:** removing logic that matters only on neighbourhoods
  the dynamics never reach saves 2–19 more gates. B (on visited patches) and C (on rollouts) find the same circuits
  here, so multi-step scoring adds nothing at ε = 0.
- **Under a tolerance, multi-step scoring is what keeps dynamics alive:** one-step pruning with ε ≥ 0.001 destroyed
  the dynamics of Life, majority and gh (divergence 0.15–0.33, F1 down to 0.02). The multi-step criterion refused
  those same ties.
- **The one clear win is a task-level criterion:** checkerboard s1, 7 gates vs 20.
- **Every approximate result holds only for the start family it was scored on.** The gh one-step circuits show it:
  exact on 64/64 unseen starts without refractory cells, broken with them.

## 5. Readable rules that came out of this
These are printed straight from the exported netlists; the inspector shows the same gates.
- **Life, 57 gates.** The exact statement: *a dead cell becomes alive if it has 3 live neighbours; a live cell stays
  alive if it has 2 or 3; otherwise it is dead.* Checked on all 512 neighbourhoods. The 57 gates themselves are
  still a 57-gate circuit; the sentence is a verified description of its function, not of its structure.
- **Excitable waves, 9 gates, equal to the generator on every input:**
  ```
  g0 = ~(NW.e | N.e)   g1 = ~(SW.e | SE.e)   g2 = g0 & g1
  g3 = ~(W.e | S.e)    g4 = ~(NE.e | E.e)    g5 = g3 & g4
  g6 = ~(g2 & g5)                      # some neighbour is excited
  g7 = ~(c.e | c.r)                    # the cell is resting
  e' = g6 & g7        r' = c.e
  ```
- **Checkerboard from noise, 6 gates, 8 channels.** Approximate: verified on 64 unseen 16×16 starts, not equivalent
  to the learned circuit.
  ```
  g0 = ~(NE.1 | SW.1)   g1 = NW.3 | g0   g2 = ~(NW.5 | g1)
  g3 = NW.5 | NW.6      g4 = NW.2 | g3   g5 = 0
  ch0' = g2  ch1' = g4  ch3' = NW.6  ch6' = g1  ch2' = ch4' = ch5' = ch7' = 0
  ```
  After one step, channels 2, 4, 5 and 7 are constant 0. What remains is a signal copied from the north-west
  neighbour (ch3 ← NW.ch6, ch1 ← NW.ch6), combined with the north-east and south-west neighbours (g0). The visible
  channel is its complement. No higher-level label is attached: none has been proved.
- **Translation:** a wire. **Propagation:** `c OR N`. **Blink:** `XOR`.

## 6. Cost
| | used | ceiling |
|---|---|---|
| GPU | 108 min = **1.80 GPU-h** (`runs/path_d/gpu_ledger.jsonl`; D0 39 min, D1 pilot 6 min + campaign 63 min; one failed launch 0.3 min) | 4 GPU-h |
| CPU, metered | **0.34 core-h** (D2, `runs/path_d/cpu_ledger.jsonl`) | 12 core-h |
| CPU, unmetered | about 0.2 core-h (estimate): smoke tests, exact-simplification checks, examples, tests | |

Separate from the closed Readable Rules campaigns. No paid APIs.

**Incidents, recorded:**
- yosys (WebAssembly build) **hangs, rather than fails,** under the meter's default `prlimit --as=16G`. One pilot was
  stopped through the meter's SIGTERM path after 632 s: no process left, CPU booked. D2 now runs with
  `prlimit --as=64G`, and the yosys call has a 120 s timeout.
- `.venv/bin/yowasp-yosys` has a shebang pointing at the pre-rename venv path. It is now invoked as
  `[sys.executable, YOSYS]`, in `ca/reduce.py` too.
- A layer-taper bug in the first D1 pilot cost 19 s of GPU time.

## 7. Decisions
- **D3 (Bad Apple): not started.**
  - Every D1 success relied on supervising the full state. Bad Apple supplies only the visible channel, so a circuit
    with hidden state would need rollout training, which failed 5/15 even on synthetic tasks. It also collapsed to
    hold on Bad Apple in Campaign 7 (CHECKPOINT_07, C5).
  - A circuit without hidden state is a memoryless radius-1 rule. Readable Rules round 4 (FINDINGS §10a) already
    shows that the best memoryless patch table of radius 1–4 does not transfer even one step to the next window.
  - A D3 run would reproduce known failures, so the remaining GPU budget is kept.
  - Phases are not proposed either: the earlier phased results were compression, not dynamics.
- **Next for Path D, if continued (needs approval before running):**
  - hidden-state learning without hidden targets, on synthetic tasks first: gh with only the excited channel
    supervised;
  - rollout training with hard-forward / straight-through states, to attack the soft-to-hard gap that dominated D0
    and B.
  - Estimate: ≤ 1 GPU-h for both on the D1 tasks.
- **Path E (compositional primitives): kept as a future idea, not proposed.**
  - The simplified circuits share only trivial parts across independent tasks (wires, a single OR or XOR).
  - The one non-trivial component, the "some neighbour is excited" OR-tree of gh, appears in one task only.
  - There is no evidence yet of reusable motion subcircuits.
- **Featured demo:** unchanged. Nothing here earns replacement.

**Bottom line:** the most fun result is that an automatically learned circuit, simplified against its own dynamics,
can come out *as* the generating rule: 1,598 learned gates → 9 gates equal to Greenberg–Hastings. The honest limit
is that this works on synthetic worlds with full supervision, and approximate pruning only certifies the starting
family it was scored on.


---
# Round 2: from visible motion to readable logic (D2.5 / D3)

## 8. Setup
**Script and outputs:**
- Script: `experiments/pd25.py` (visible-only trainer and evaluator), also used by `experiments/pd_d3.py`.
- Results: `runs/path_d/d25/`, `runs/path_d/d3/`; pruning in `runs/path_d/d2/gh_burnin_s0*`.
- Ledgers: `runs/path_d/d25/{gpu,cpu}_ledger.jsonl`.

**What the learner sees:**
- **Only the visible channel.** k = 2 latent Boolean channels must be invented by training.
- **Hidden initialisation**, never from the generator's hidden state:
  - **zero:** hidden = 0 at the window start.
  - **burnin (observation prefix, P = 2):** start 2 frames earlier with hidden 0. For 2 steps the circuit runs with
    its visible channel overwritten by the observed frames, then runs free. It is one shared rule, nothing is learned
    per sequence, and its cost is 2 extra visible frames.
  - A per-sequence learned seed was **not** used: it cannot be applied to a new sequence.
- **Objectives:**
  - soft rollout;
  - `ste`: hard-forward, i.e. argmax gates and rounded states in the forward pass, straight-through gradients.
  - Both use the horizon curriculum 1→16 and the same budget (48 k step-evaluations).

**Export extension:** `prefix` in `badca/logic-ca-v1`. After step i < P, channel 0 is set to the stored observed frame
i. Both executors implement it, and Python and Node match bit for bit (`tests/test_path_d.py`).

**Data, Greenberg–Hastings (fixed before any fit):**
- 48×48 periodic grid, sparse starts (excited 0.004, refractory 0.004): fronts expand, then spirals persist with about
  0.12 of cells changing per step.
- The first regime tried (the D1 starts) dies out by step 20. It was discarded **before** training, because hold would
  have been trivially perfect.
- Splits: 16 train / 8 val / 16 test sequences of 160 steps; fresh-start windows at t0 = 4, 40, 100.
- Windows with no motion at all are skipped and counted: 18 of 48 test windows (sequences that die out).
- **Out of family:** a dense refractory-rich regime (starts 0.08 / 0.2), where every cell cycles with period 3.

**Baselines:**
- hold;
- the best memoryless 3×3 rule (a majority table fitted on training transitions; it cannot represent refractoriness);
- a **cycle rule** `e' = ~e & ~e_prev`: two frames of memory and **no neighbours**. This is the trivial way to fake
  the waves.

## 9. D2.5-A: visible-only synthetic learning (GH)
Test split, 30 windows with motion, 128 free-running steps. Hold 0.217, memoryless table 0.376, cycle rule 0.199
(at 16 steps).

| setting | seed 0 | seed 1 | seed 2 | dense regime (s0 / s1 / s2), err128 |
|---|---|---|---|---|
| A full state (D1, one-step) | exact | exact | exact | – |
| B zero hidden | err128 0.305, 0/30 exact | 0.352 | 0.328 | 0.41–0.42 |
| C burnin, soft | **exact 30/30**, F1 1.00 | 6/30; err16 0.048, err128 0.212 | 22/30; err128 0.030 | **0.007** / 0.371 / 0.322 |
| D burnin + ste | err128 0.137, F1 0.06, change mass 0.03 (frozen) | – | – | 0.333 |
| D' zero + ste | same as D (collapse) | – | – | 0.333 |

- **Gate D2.5-A: passed** with C, the observation prefix. This is non-trivial motion on unseen trajectories, beating
  hold, the memoryless table and the cycle rule.
- **Prefix:** without it (B) nothing generalises. With it, 1 of 3 seeds is exact everywhere, 1 nearly, 1 only short-term.
- **Straight-through training collapses** to near-static output in every case here and on Bad Apple (§11). It is not
  the fix for the soft-to-hard gap at this scale.
- **Soft vs hard at 16 steps (error):** C s0 0.000 / 0.000, s1 0.079 / 0.065, s2 0.011 / 0.009, ste 0.37 / 0.12.
- **Information cost:** 2 latent bits per cell, initial hidden all 0, plus 2 observed frames (2 × 2,304 bits). No
  per-sequence parameters.

## 10. D2.5-B: stability and simplification
Seed 0 of C:
- 1,823 hardened gates → **65** (exact passes) → **14** (ABC). SAT-equivalent.
- B ε=0 stays at 14.
- **C ε=0 reaches 10 gates.** Visible rollouts are identical on 64/64 unseen starts. 2 % of one-step outputs on visited
  states change, but only in latent values that never reach the visible channel. One-step scoring cannot see that.
- **Re-scored on the untouched test split:** exact 30/30 for 128 steps. Dense regime: 0.0065.
- B ε ≥ 0.001 breaks it (change mass 37, divergence 0.10–0.16).

**The 10 gates** (v = excited; h1, h2 = learned latents):
```
g0 = ~(N.v | W.v)    g1 = ~(NE.v | E.v)   g2 = g0 & g1    g3 = c.h2 | g2
g4 = ~(SW.v | SE.v)  g5 = ~(NW.v ^ S.v)   g6 = g4 & g5    g7 = ~(g3 & g6)
g8 = ~(NW.h1 | c.v)  g9 = g7 & g8
v' = g9      h1' = SE.v      h2' = c.v
```
Readable from the gates alone:
- **The circuit stores its memory in a neighbour.** `h2' = c.v` stores "I was excited". `h1' = SE.v` stores the
  south-east neighbour's excitation, so next step `NW.h1` (the north-west neighbour's copy of *its* south-east cell)
  is **this cell's own previous excitation**.
- `g8` = resting (neither excited now nor last step).
- `g7` = some neighbour among N, W, NE, E, SW, SE is excited, or exactly one of NW and S is.
- **The flaw:** it misses the case "only NW and S are both excited", so it is **not** equal to Greenberg–Hastings on
  every input. That configuration never occurs in the test family; the dense regime shows it (0.0065).
- Inspector: *Waves learned from the visible channel only*.

## 11. D3: real Bad Apple clips (visible only, one fixed circuit, no schedule, no injection after the 2-frame prefix)
**Clips**, fixed before training (`experiments/pd_d3.py::CLIPS`):

| clip | case |
|---|---|
| 1221 | translate |
| 4500 | contour |
| 3998 | growing silhouette |
| 1499 | wipe |

- **Windows:** train = [a, a+L], val = [a+L, a+2L] (model selection), final = [a+2L, a+3L] (untouched).
- **Baselines**, all fitted on train only: hold, shift, memoryless 3×3 table, and a Readable Rules staged `dlrule`.
- **Fresh-start transfer:** the window's visible frame plus its 2-frame prefix.
- **Continuous continuation:** from a, with no re-initialisation.

**Fresh-start errors:**

| run | gates (simplified) | train err / hold / best baseline | val err / hold / best baseline | final err / hold / best baseline |
|---|---|---|---|---|
| 1221 L16 | 430 | 0.156 / 0.299 / readable 0.097 | 0.202 / 0.132 / 0.065 | 0.420 / 0.416 / 0.416 |
| 1221 L8 | 482 | 0.308 / 0.316 / 0.125 | 0.240 / 0.089 / 0.089 | 0.256 / 0.105 / 0.073 |
| 4500 L16 | 252 | 0.123 / 0.094 / 0.076 | 0.163 / 0.073 / 0.061 | 0.187 / 0.060 / 0.060 |
| 4500 L16 ×3 budget | 640 | 0.099 / 0.094 / 0.076 | 0.115 / 0.073 / 0.061 | 0.161 / 0.060 / 0.060 |
| 4500 L8 s0 / s1 | 347 / 265 | 0.045, 0.049 / 0.067 / 0.057 | 0.048, 0.051 / 0.057 / 0.057 | 0.061, 0.058 / 0.056 / 0.055 |
| 3998 L16 | 263 | 0.069 / 0.078 / 0.040 | 0.076 / 0.093 / 0.078 | 0.029 / 0.013 / 0.013 |
| **3998 L16 ×3 budget** | 461 | 0.037 / 0.078 / 0.040 | **0.055 / 0.093 / 0.078, F1 0.62** | 0.041 / 0.013 / 0.013 |
| **3998 L8 s0 / s1** | 334 / 273 | **0.019, 0.013** / 0.057 / 0.027 | **0.027, 0.028** / 0.035 / 0.034 | 0.053, **0.044** / 0.054 / 0.051 |
| 1499 L16 / L8 (wipe) | 407 / 273 | ≈ hold | ≈ hold | worse |
| ste (4500, 3998; L8; 2 seeds) | 1–73 | collapse or flood (0.83–0.96), or near-static | – | – |

- **Continuous continuation fails everywhere.** After the training window the rollout is worse than hold (e.g. the pruned
  3998 L8 s1 circuit: val 0.041 vs 0.035, final 0.098 vs 0.054).
- **Anti-cheat on 3998 L8 s0:** births and deaths are balanced (pred / true: births 98/116, deaths 61/63 in train), and
  the silhouette centroid tracks the original. It is neither erosion, filling, nor a frozen frame.
- 4500 L8 over-erodes: 304 predicted deaths vs 185 true.

**Optional extension chosen:** reduce the circuit while keeping motion. Unit tying is scored on the **training window
only** (error may rise ≤ 0.003); val and final are re-scored afterwards. This is approximate, fitted-trajectory
pruning, not an equivalence.

| circuit | gates | train | val | final |
|---|---|---|---|---|
| **3998 L8 s1** | 271 → **64** (depth 9) | **0.016** (hold 0.057), F1 0.82 | **0.025** (hold 0.035, table 0.034, readable 0.039), F1 0.63 | **0.041** (hold 0.054, table 0.051, readable 0.068), F1 0.68 |
| 3998 L16 ×3 | 456 → 4 | 0.037 | 0.084 (hold 0.093) | 0.057 (hold 0.013) |
| 4500 L8 s0 | 346 → 38 | 0.048 | 0.058 (≈ hold) | 0.073 (worse) |

- **The 64-gate circuit** is the smallest successful one. It generates frames **3998–4006** by itself from frame 3996
  plus 2 observed frames. Started fresh on the following unseen frames 4006–4022, it beats every baseline.
- **But it captures only 9 % (val) and 4 % (final) of the true changes there.** It wins by changing little and mostly
  correctly, not by reproducing the motion.
- Its 64 gates are inspectable gate by gate in the inspector (*Bad Apple frames 3998–4022*). No sentence-level
  description has been verified, so none is given.
- **The 3998 L16 collapse to 4 gates** at the same training error shows the large L16 circuit was not using its gates
  on its own training frames.

## 12. Decision, cost, next
**Classification: PARTIAL.**
- Visible-only synthetic learning works: exact GH waves on unseen trajectories from the visible channel and a 2-frame
  prefix, 10 gates.
- On Bad Apple, a compact fixed circuit reconstructs an 8-frame fitted window with real, balanced motion (F1 0.82).
  Its numerical wins on unseen frames come with very little reproduced motion, and continuous continuation fails.
- **Not GO:** there is no reliable motion generalisation beyond the fitted frames.

**What remains unsolved:**
- generalising motion past the fitted window on video;
- continuous continuation;
- 16+ fitted frames with small circuits (the 1221 / 4500 L16 fits do not even beat baselines on their own frames);
- the soft-to-hard gap (straight-through training collapsed).

**Compute, this campaign:**

| | used | ceiling |
|---|---|---|
| GPU | 93.8 min = **1.56 GPU-h** (D2.5-A 30.8 min, D3 63.1 min; no failed GPU runs) | 6 GPU-h |
| CPU, metered | 0.03 core-h (pruning) | 8 core-h |
| CPU, unmetered | about 0.3 core-h (estimate): CPU smoke tests, evaluations, examples | |

**Cumulative Path D:** GPU 201.9 min = **3.37 GPU-h**; CPU about 0.9 core-h. Each GPU job held one GPU, so the totals
are summed allocated GPU time.

**Next recommended experiment (not run):** multi-clip training on 3998-like scenes. One circuit is fitted on several
short windows of the same silhouette (still a fixed circuit, still visible-only), to test whether the motion that is
missing on unseen frames is a data problem (one 8-frame window) or a capacity / objective problem. Estimate: ≤ 1 GPU-h.
