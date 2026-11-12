# Path D: Differentiable Logic CA

**Goal:** learn genuinely discrete, spatially shared, local Boolean circuits that run a CA by themselves, then make
them small enough to read. Experimental track, separate from the featured Neural CA (unchanged) and from Readable
Rules (closed with NO-GO, D9; nothing there was rerun).

**Status, 2026-11-12:**

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
