# Checkpoint 06: better search for a compressing CA circuit on ba32_3000_5

**Sources**
- **Plan:** `docs/CAMPAIGN_06.md`, written before the searches, with the pilot and the fixed budget
  appended afterwards.
- **Code:** searches ran from the frozen snapshots `2be472b71f78` (pilot) and `ab973e1cb669` (main and
  cross-checks).
- **Machine-readable:**

  | file | contents |
  |---|---|
  | `runs/c6/results.json` | 83 solver runs: status, CPU, wall, peak memory, CNF and target SHA-256, decode checks |
  | `runs/c6/cpu_ledger.jsonl` | every metered process |
  | `runs/c6/bugcheck.json` | the sign-error check |
  | `runs/c6/breakeven.json` | the real-file sizes |

- **Not committed:** the CNF files (493 MB) stay on disk, untracked. Their SHA-256 and the
  generating snapshot are recorded; encoding is deterministic.

**Fixed setting:**
- ba32_3000_5, target SHA-256 `b624dcdc…`;
- F_priv k=1 (n=10 inputs), S=2, boundary 0, hidden init 0;
- 2-input gates with arbitrary truth tables; the Campaign 05 container file.

**Success** means a file that replays exactly in Python and Node from its bytes alone, at **< 80
bytes**, the smallest baseline (change list = zlib = 80 bytes).

## Answers

**1. Solver swap or encoding change: which helped?**
- **Solver:** Kissat 4.0.4 against CaDiCaL 1.9.5, on the same DIMACS files.
  - Kissat was faster on 5 of the 6 decided UNSAT instances, by 1.1–1.6× (e.g. ba G4 431 vs 485 s,
    disc12 G4 91 vs 143 s).
  - It also decided **two harder instances that CaDiCaL could not** within 650 s: cube `so0=11` UNSAT
    in 103 s, and the chain+hidc G8 subspace UNSAT in 223 s.
  - That is a real but **modest** gain, on single runs.
- **Encoding (A, existence-preserving nodegen):**
  - same verdicts as the base;
  - speed mixed: disc12 G4 1.56× and ba G3 1.3× faster, ba G4 0.92× (slower).
  - No clear gain.
- **Cube-and-conquer:** it settled 12 of the 22 parts of the frontier instance. Without it, that
  instance was UNKNOWN as a whole.

**2. Did the effect reach the frontier?**
- **Only partly.** As whole instances, ba32_3000_5 G5 and G12 and disc12 G5 stayed **UNKNOWN** with
  both solvers (450 s).
- The G=12 instance was split into 22 cubes by the visible output's selection; these cover the
  whole space and are disjoint.
  - **12 cubes UNSAT:** output = any of the 10 input taps, gate 0 or gate 1.
  - **10 cubes UNKNOWN:** output = gates 2–11, each at the 650 s cap.
- So the frontier was narrowed, not crossed.

**3. Was a compressing exact circuit found?**
- **No.** No SAT model came up on ba32_3000_5 in any configuration. The result is **UNKNOWN**, not
  impossibility.
- The only SAT runs were the positive control pcB G3, solved by both solvers and verified exactly in
  Python and Node from the file (29 bytes).

**4. New UNSAT bounds: which family and range?**
All of these hold within F_priv k=1, S=2, boundary 0, hidden init 0 and the 2-input arbitrary-gate
basis.
- **No change to the full-space bound** from Campaign 05: no circuit with ≤ 4 gates, re-decided by both
  solvers here (C05 lex/v1 G4).
- **Under nodegen** (no buffers; constant and NOT gates may feed outputs only) **at exactly G' gates**,
  G' = 0–4: UNSAT for both ba32_3000_5 and disc12, by both solvers.
  - By the normalization argument in CAMPAIGN_06 §4A, this again means no circuit with ≤ 4 gates.
  - This is consistent with Campaign 05, not a new range.
- **Partial result at G=12:** inside the G=12 lex encoding, which covers every circuit with ≤ 12
  gates through padding, **every model must have the visible output on one of gates 2–11.**
  - The visible output cannot be an input tap, gate 0 or gate 1.
  - Cubes so0=0–10 are UNSAT by both solvers. so0=11 is UNSAT by Kissat only; CaDiCaL timed out.
  - Because of padding and lex ordering, a gate's position is not invariant. This excludes regions of
    the encoding, and is **not** a statement such as "the output needs depth ≥ 3".
- **Heuristic subspaces only:**
  - chain (gate j reads gate j−1) + hidc (hidden = visible centre) + nodegen, at exactly 8 gates: UNSAT
    (Kissat; CaDiCaL timed out).
  - witness: states fixed to the exp03c LUT solution, at exactly 5 gates: UNSAT by both solvers.
- **Proofs:** none was verified. Kissat in pysat cannot log proofs, and no DRAT checker was available,
  because the standalone build was blocked by the permission policy. UNSAT verdicts are instead
  **cross-checked** by the second solver where it finished; the exceptions are listed above.

**5. Space excluded by heuristics**
- **chain + hidc:**
  - every gate reads the one before it, so the circuit is a single chain with side inputs;
  - hidden = the visible centre (a one-step delay);
  - UNSAT at exactly 8 gates; chain or hidc alone at 12 gates was UNKNOWN (300 s).
- **witness:**
  - every unobserved state bit fixed to the existing LUT solution's trajectory;
  - this is a separate "uses an existing solution" track, never counted as video-only;
  - UNSAT at 5 gates, UNKNOWN at 8 and 12 (120 s).
- No exact solution was found in any heuristic subspace either.

**6. Budget and next investment**
- **Campaign 06:** **3.97 CPU core-h** of 8, 0 GPU.

  | part | core-s |
  |---|---|
  | solving | 14,145 |
  | encoding | 39 |
  | checks and tests | 57 |
  | break-even files | 6 |
  | driver | 30 |

- **Earlier campaigns, separate:** C05 5.78 core-h, C04 about 11.1 core-h + 0.68 GPU-h.
- **Unused budget:** about 3 core-h was **left unspent** under the fixed plan.
- **What the evidence says:**
  - The remaining question is whether any of the 10 open cubes (visible output = gate 2–11, ≤ 12
    gates) has a solution.
  - Each was unresolved after 650 s, so closing them needs either much more CPU per cube or a stronger
    reformulation.
  - Solver swaps give ≤ 1.6× on decided instances and occasional larger wins; they will not close
    the gap alone.
- **Worth doing next, if anything:**
  - **(a)** Split the 10 open cubes further, e.g. by the hidden output's selection, and give each
    a defined budget. This is the only route that can end in a whole-space UNSAT (proving that
    compression is impossible at G ≤ 12) or a SAT (success).
  - **(b)** A verified-proof setup (an allowed DRAT/LRAT checker), so any UNSAT claim becomes
    checkable.
- **Not recommended:** more heuristic subspaces, since none produced a solution, and lower-G
  minimality ladders, which do not bear on compression.

## Correctness work (part 2 of the request)
- **The C05 sign error was a pure variable renaming.**
  - Old and new CNFs are identical clause multisets under tt[j][q] → tt[j][3−q], on 8 instances,
    with and without lex.
  - So every C05 SAT/UNSAT verdict stands. Only the witness reading was wrong, and it was already
    fixed. **No verdict needed correcting.**
- **Semantics test (`tests/test_c6_semantics.py`):** the SAT encoding, the Python evaluator and the
  Node player agree on:
  - all 16 operations, including constants, buffers and NOT;
  - G=0 direct wiring and an S=2 chain;
  - every operation with a different rollout being rejected by the encoding.
  - It also confirms monotonicity in G and that nodegen keeps the planted minimum sizes.
  - The existing `tests/test_c5.py` was re-run and passes.
- **"Exactly G" vs "≤ G":**
  - the encoding means exactly G gates, all used;
  - for base/lex, buffer padding makes UNSAT at G imply none with ≤ G;
  - under nodegen, padding is impossible, so "≤ G" needs UNSAT at every G' ≤ G (done for G ≤ 4).
- **Break-even with real files:**

  | G | file bytes |
  |---|---|
  | 10 | 75 |
  | 11 | 77 |
  | 12 | 78 |
  | 13 | 80 |
  | 14 | 82 |

  - All parse in Python and Node, so **G ≤ 12** is required.
  - All 6 baseline files decode exactly again (best 80).

## Process notes
- **Blocked build:** building standalone solvers from freshly cloned source was blocked by the
  environment's permission policy. The clones were removed unbuilt, and the pysat-bundled solvers
  were used instead.
- **Driver bug, before any search:** a keyword-argument bug dropped the ledger records of two
  smoke-test encodes. Those CNFs were deleted and re-encoded so their CPU is counted.
- **Timing:** single-run wall and CPU on a shared machine, so differences under about 1.5× are not
  used as conclusions.
