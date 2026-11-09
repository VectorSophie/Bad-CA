# Campaign 06: better search for a small, compressing CA circuit on ba32_3000_5 (pre-run plan)

Written 2026-11-09, before any Campaign 06 search. Already run before this was written, and listed in
the ledger:
- the encoding-equivalence check of the C05 sign error;
- the semantics tests;
- an attempt to build standalone solvers. It was blocked by the environment's permission policy
  (running code fetched from outside), so nothing was built.

Pilot results and the fixed budget are appended below later; this part is not edited.

## 1. Fixed target
- **Clip:** ba32_3000_5 (frames 3000–3004, 32 px wide, box resize, threshold 128). The target
  SHA-256 is the same as in Campaign 05.
- **Rule family:** the same as Campaign 05.
  - F_priv k=1: visible 3×3 plus the cell's own hidden bit, n = 10.
  - S=2, boundary 0, hidden init 0.
  - Basis: 2-input gates with an arbitrary truth table.
  - File: the `ca/container.py` file (circuit payload).
- **Success** means all of the following at once:
  - a file that replays from its bytes alone in Python **and** the Node player;
  - every frame byte-exact;
  - **total bytes < 80**, the smallest existing baseline (change list = zlib = 80 bytes).
  - A gate count alone is never treated as success.
  - The break-even is re-checked with real files of G = 10–14 gates. From Campaign 05: G ≤ 12 gives
    78 bytes.
- **If nothing is found,** the result is **UNKNOWN**. A timeout is never read as impossibility.
- **Not in scope:** UI, training, longer clips, other videos.

## 2. Encoding and verdicts: done before this plan
- **The C05 sign error was a variable renaming.** The old pilot CNF equals the fixed CNF exactly
  (as a clause multiset) after renaming tt[j][q] → tt[j][3−q], on 8 instances (pcA, pcB, disc12,
  bar12, ba32_3000_5, with and without lex): `runs/c6/bugcheck.json`.
  - So every SAT/UNSAT verdict is unchanged.
  - Only the reading of the witness (the decoded circuit) was wrong, and that was already fixed in C05.
  - No verdict needs correcting.
- **`tests/test_c6_semantics.py`** confirms that the encoding, the Python evaluator and Node agree:
  - on all 16 gate operations, including constants, buffers and NOT;
  - on direct wiring (G=0) and an S=2 chain;
  - SAT is monotone in G;
  - every op with a different rollout is rejected.
- **What G means:**
  - The encoding asks for **exactly G gates, all used**.
  - A buffer gate pads any smaller circuit, so for the base/lex encodings, UNSAT at G means **no
    circuit with ≤ G gates** in this family.
  - Under nodegen (below) padding is impossible. A bound of "≤ G" then needs UNSAT at **every** G' ≤ G.

## 3. Solver comparison (first)
- **Solvers:** CaDiCaL 1.9.5 (current) against **Kissat 4.0.4** (alternative), both from the
  installed pysat 1.9.dev21.
  - Both get the **same DIMACS file**, with its SHA-256 recorded.
  - Each runs a single, non-incremental solve: no assumptions, phases or reused state. The encoding
    needs none of these.
  - Seeds: the solver defaults; pysat does not expose them, and this is recorded.
  - Proofs: Kissat in pysat cannot log proofs. CaDiCaL can, but no DRAT checker is available.
    **UNSAT proofs are therefore marked unverified.** Instead, UNSAT verdicts are cross-checked by
    the other solver.
- **Instances:**
  - **decided before:** pcB G2 (UNSAT) and G3 (SAT); disc12 G3 and G4 (UNSAT); bar12 k1_S4 G2 (UNSAT);
    ba32_3000_5 G3 and G4 (UNSAT);
  - **frontier:** ba32_3000_5 G5 and G12; disc12 G5.
- **Same limits for every run:** 1 thread, the same `prlimit --as` and `--cpu`.
- **Verdict rule:** a solver "wins" only if it is faster on most decided instances **and** decides
  at least one frontier instance, or is clearly faster on them. One fast run is not enough.

## 4. Search-space reduction (kept separate)
**A. Existence-preserving.** Each is checked against the base verdicts on small instances.
- **lex:** symmetry v2 from Campaign 05.
- **nodegen:**
  - no buffer gates (tt 12 = a, 10 = b);
  - constant (0, 15) and NOT (3, 5) gates may feed outputs only.
  - **Why this is safe for "at most G":** take any circuit with ≤ G gates and process its gates in
    topological order.
    - A buffer is removed, and its uses are redirected to its source.
    - A constant or NOT feeding a gate is absorbed: the consumer's truth table is rewritten, and its
      freed input is re-wired to any other earlier signal (the pool has ≥ 10 signals).
    - Gates that become unused are removed.
    - The function and the outputs are unchanged, a < b is restored by permuting the table, and the
      gate count never grows.
  - **What it cannot remove:** a constant or NOT gate that is itself an output, for example
    "hidden = 0" or "visible = NOT tap". Those stay allowed.
  - **Consequence:** padding is lost, so lower bounds need the full ladder G' ≤ G.

**B. Heuristics that can lose solutions.** Exact solutions found here are valid; UNSAT holds only
inside the subspace.
- **chain:** gate j must read gate j−1, so circuits are a single chain with side inputs.
- **hidc:** hidden output = the visible centre tap, so hidden is a one-step delay.
- **witness:** every unobserved state bit is fixed to the hidden/intermediate trajectory of the
  existing exp03c LUT solution.
  - This is reported as **"compression using an existing solution"**, never as video-only synthesis.

## 5. Cube-and-conquer (only if 3–4 do not cross the frontier)
- **Split variables:** the visible output selection `so[0][p]`, p ∈ pool (n + G values).
  - Exactly one is true in every model, so the cubes **cover the whole space** and are disjoint.
  - It decides what drives the visible output, a natural top-level structural choice.
- **Global UNSAT** is claimed only if every cube is UNSAT.
- **Workers:** 1 by default, at most 2.
- **Resumable:** the job list and every per-cube result are in the ledger, and finished keys are skipped.
- **Costs:** generation, solving and verification all count.

## 6. Order and budget
- **Ceiling:** **8 CPU core-hours** of new runs and verification. **GPU: none.**
- **CPU definition:** user + system CPU of every process, including its threads and children, via
  `os.wait4`, logged in `runs/c6/cpu_ledger.jsonl`. Wall time is logged separately.
- **Wall cap:** 3× the CPU cap per job, KILL. Earlier campaigns' costs are reported separately.

| part | core-h |
|---|---|
| correctness + pilot (solver comparison, A on decided instances, break-even files) | ≤ 2 |
| frontier search on the most promising path | ≤ 5 |
| final verification and analysis | ≥ 1 |

- **Stop rule:** the driver starts no new job once the ledger reaches 7.0 core-h.
- **Fixed after the pilot,** and not raised for the sake of results.
- **G choice:**
  - Start at G=5. Do not spend the budget proving small G.
  - The frontier phase **prioritizes finding a compressing solution**: G near the upper limit
    (10–12) with the best path.
  - Minimality proofs are secondary.
  - The choice is recorded.

## 7. Recording
- **Code:** frozen snapshots only.
- **Per process:** command, solver, CNF SHA-256, target SHA-256, CPU, wall, peak RSS, exit reason.
- **SAT circuits:** saved at once (`.sat.json`), replayed in Python, written to a file, and decoded by
  Python and Node from the file's bytes.
- **On success:** the winning file and the baseline files are stored side by side, each with both
  decoders' checks.

---
## Pilot results (appended after the pilot, before the main run; ledger 1.58 core-h)
Snapshot `2be472b71f78`. All runs used 1 thread, a 24 GB address space and a wall cap of 3× the CPU cap.

**Break-even with real files** (`runs/c6/breakeven.json`):

| G | file bytes |
|---|---|
| 10 | 75 |
| 11 | 77 |
| 12 | 78 |
| 13 | 80 |
| 14 | 82 |

- The best baseline is 80 bytes (change list = zlib), so success requires **G ≤ 12**.
- All these files parse in Python and Node.
- The baseline files decode exactly again.

**Solver comparison** (same DIMACS, SHA-256 recorded, lex encoding), solve CPU in seconds:

| instance | CaDiCaL 1.9.5 | Kissat 4.0.4 |
|---|---|---|
| pcB G2 UNSAT | 0.35 | 0.83 |
| pcB G3 SAT | 1.19 | 0.94 |
| disc12 G3 UNSAT | 4.65 | 4.14 |
| bar12 S4 G2 UNSAT | 17.5 | 15.7 |
| ba32_3000_5 G3 UNSAT | 29.4 | 24.9 |
| disc12 G4 UNSAT | 142.8 | 91.2 |
| ba32_3000_5 G4 UNSAT | 484.7 | 430.9 |
| ba32_3000_5 G5 | UNKNOWN (450) | UNKNOWN (450) |
| ba32_3000_5 G12 | UNKNOWN (450) | UNKNOWN (450) |
| disc12 G5 | UNKNOWN (450) | UNKNOWN (450) |

- Kissat was faster on 5 of the 6 decided UNSAT instances, by 1.1–1.6×, and slower on the trivial pcB G2.
- **Neither solver crossed the frontier.**
- Kissat is used from here on, as a modest single-run advantage, not a strong conclusion.

**A (nodegen + lex, Kissat):**
- **Same verdicts as the base:** UNSAT at G=0–4 on both disc12 and ba32_3000_5.
- **Speed vs lex alone:** mixed.

  | instance | nodegen + lex | lex alone | ratio |
  |---|---|---|---|
  | disc12 G4 | 58.3 s | 91.2 s | 1.56× faster |
  | ba32_3000_5 G3 | 18.8 s | 24.9 s | 1.3× faster |
  | ba32_3000_5 G4 | 466.7 s | 430.9 s | 0.92× (slower) |

- **New:** ladder-based bound "no nodegen circuit with ≤ 4 gates" on both clips, consistent with the earlier base bounds.

**B (heuristics, ba32_3000_5, Kissat):**
- chain + hidc at G=8: **UNSAT** in 223 s, inside that subspace only.
- chain at G=12 and hidc at G=12: UNKNOWN at 300 s.
- **witness** (all unobserved states fixed to the exp03c LUT solution): G=5 UNSAT in 0.3 s; G=8 and G=12 UNKNOWN at 120 s.

**Decision:** the solver and A did not cross the frontier, so the §5 condition is met and
cube-and-conquer runs.

## Fixed budget (main)
- **Priority: finding a compressing solution,** not proving minimality.
- **Instance:** ba32_3000_5 **G=12, lex, no nodegen,** Kissat 4.0.4.
  - Buffer padding makes this one instance cover **every circuit with ≤ 12 gates**, so every
    compressing size is covered.
- **Cubes:** `so0=p` for p = 21..0. That is the visible output's pool selection: gates first, last
  gate first, then taps.
  - 22 disjoint cubes **covering the whole space**.
- **Per cube:** CPU cap 650 s, 24 GB, wall cap 3×.
- **Workers:** 2.
- **Stop at the first SAT cube:** the file is verified and it is a success.
- **Global UNSAT** (no circuit ≤ 12, so compression impossible in this family) is claimed only if all
  22 cubes are UNSAT. Otherwise the result is UNKNOWN, with the excluded cubes listed.
- **Worst case:** 22 × 650 s = 3.97 core-h, so the ledger reaches at most 5.55 core-h, leaving ≥ 2.4
  core-h for verification and reporting. The stop rule is unchanged at 7.0.
- **Not run:** lower-G proof ladders (secondary), and further heuristic subspaces.
