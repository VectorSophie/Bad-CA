# Campaign 05: direct synthesis of small circuits, compared by total file size (pre-run plan)

Written 2026-11-09, **before** any Campaign 05 run. Results go to `docs/CHECKPOINT_05.md` and
`runs/c5/results.json`. A pilot section and a fixed-budget section are appended after the pilot; this
first part is not edited afterwards. Corrections go into a dated note at the end.

**Question.** Can synthesis under a "small program" constraint, from the start, find a CA
representation that is smaller than a recording? This replaces the old route of finding an exact LUT
and shrinking it afterwards.

## Starting point (CHECKPOINT_01–04)
- Every exact rule found so far came from LUT SAT, possibly followed by ABC reduction.
- Every such rule is larger than the best recording of its clip. For example, on ba32_3000_5 the
  smallest is 588 bits (keep-exception LUT) and the best ABC circuit is 2133 bits, against a
  195-bit change list.
- The circuits from ABC (≥125 gates on ba32_3000_5) come from an arbitrary exact LUT, not from a
  search for a small circuit. Whether a much smaller exact circuit exists is open.

## 1. What is fixed
- **Model:**
  - one fixed local rule shared by every cell;
  - Moore radius 1, with the taps given by the family;
  - fixed steps per frame (S), boundary constant 0;
  - initial state: visible = the clip's first frame, hidden = all 0.
  - Not allowed: frame injection, rule switching, time or coordinate inputs, per-scene resets.
  - Other initial states are **out of scope this campaign** (not run).
- **Lossless is primary.** Approximate replay is not run in this campaign, so there are no
  approximate numbers.
- **Cost = bytes of a decodable file** (`ca/container.py`, one bitstream, padded to whole bytes).
  - **Every file has:**
    - the method id;
    - W, H, T (Elias-gamma);
    - the first frame, coded by one function shared by all methods that store it separately (1 flag
      bit, then raw bits or frame-RLE, whichever is shorter);
    - the method payload.
  - **CA payload:**
    - the family id, k, S, a 2-bit boundary and a 1-bit hidden init;
    - then either a circuit (gate count, a 4-bit truth table per gate, two wire references of
      ⌈log2(pool)⌉ bits each, output references) or a keep-exception LUT.
    - The taps are implied by the family id and k, and that rule is part of the generic decoder.
  - **Recording payloads:** change list, XOR+RLE, per-frame RLE (existing coders).
  - **General-purpose payloads:** zlib (raw deflate, level 9), bz2 (level 9) and lzma (raw stream,
    preset 9e). Each runs on the packed bits of all T frames, and on frame 0 + packed XOR deltas;
    the smaller variant per compressor is reported.
  - **Generic decoder code is excluded the same way for every method:**
    - the CA player;
    - the RLE / change-list decoders;
    - the zlib/bz2/lzma libraries.
    - Nothing clip-specific may live in a decoder.
  - **Auxiliary only:** payload bits without the header and first frame (continuity with CHECKPOINT_02–04).
- **Reported separately from size:**
  - update count (cells × generations);
  - decode time (Python, Node);
  - search time (CPU core-seconds), peak memory.

## 2. Direct circuit synthesis (primary)
- **Circuit family C(n, G):**
  - G gates, each with 2 inputs and an **arbitrary 2-input Boolean function** (4 truth-table bits).
  - Gate j reads two distinct, ordered earlier signals (index a < b) from the pool [n tap inputs,
    gates 0..j−1].
  - Each output channel (visible, then hidden) selects any pool signal.
- **Basis choice:** arbitrary 2-input functions (16 ops) rather than AND-inverter or a fixed gate list.
  - It makes the SAT encoding uniform: 4 free truth-table variables, no gate-type selection.
  - It is exactly the format the existing bit encoder (`costs.rule_circuit`) and the Node player
    (circuit-v1) already decode and evaluate.
  - This campaign tries no other basis.
- **Encoding** (`ca/circuit_sat.py`):
  - **Gate selection:** one-hot selection variables for each gate input and each output.
  - **Circuit copy per cell and generation:** each copy of the circuit is evaluated with mux clauses
    of the form sel → (wire ↔ source) and (a,b) → (out ↔ tt[a,b]). Its inputs are the cell's tap
    states, which are constants when known (boundary, observed frames, hidden init) and variables
    otherwise. Copies with identical input literals are shared.
  - **Outputs:** each output equals the next state.
  - **Hard constraints:** every observed frame is exact; nothing else.
  - **Symmetry handling, v1:**
    - a < b inside a gate;
    - every gate must be used by a later gate or an output, so there are no dangling gates.
    - A buffer gate can pad any smaller circuit, so **SAT is monotone in G**. UNSAT at G therefore
      proves that no circuit with ≤ G gates exists in C(n, ·) under that structure, boundary,
      hidden init and S.
  - **v2,** only if the pilot shows a symmetry bottleneck: lexicographic ordering of independent
    adjacent gates. v1 and v2 results are both kept, and both count toward the budget.
- **Search:** ascending G = 0, 1, 2, … with a CPU cap per G, stopping at the first SAT.
  - **Statuses:**
    - SAT;
    - UNSAT (proof for that G and below);
    - UNKNOWN_cpu_limit or UNKNOWN_wall_limit;
    - UNKNOWN_memout;
    - ERROR (implementation).
  - **Minimality** is claimed only when every smaller G is UNSAT.
- **Saving and checking:**
  - Every SAT circuit is saved at once: gates, outputs, settings, the target SHA-256, the solver
    statistics and the hidden trajectory.
  - It is then rolled out in Python, as binary states with an autonomous run.
- **Baseline kept separate:** the existing route of LUT SAT, then ABC or a keep-exception LUT
  (CHECKPOINT_03/04 rules), put into the same container.

## 3. Structure ablation (limited, narrowed by pilot)
- **Base:** F_priv(k=1), visible 3×3 + own hidden, n = 10.
- **Comparisons:**
  - k ∈ {0, 1, 2};
  - F_shared(k=1), where neighbours' hidden bits are read too (n = 18);
  - S ∈ {1, 2, 4, 8}.
  - Each combination is chosen by certificates first:
    - **light cone:** any radius-1 rule, constant initialization;
    - **state repeat:** memoryless only;
    - **history conflict:** F_priv, S = 1.
  - Then by the pilot. Not all combinations are run.
- **Reported cost growth with n:** LUT encoding (exceptions over 2^n entries) vs circuit encoding
  (log2(pool)-bit references).
- **Interpretation, kept separate:**
  - memory (k);
  - communication range (shared hidden);
  - number of updates (S).

## 4. Targets, in order
1. **Feasibility review** of each synthetic clip × structure by certificates. An infeasible
   combination is reported as infeasible, never as an optimization failure.
2. **Positive control:**
   - **Source:** a clip produced by a small known circuit, k = 1, a few gates, a random first frame,
     12×12.
   - **Success:** the direct synthesis recovers some exact circuit with ≤ the known gate count.
3. **Synthetic clips:**
   - **moving object:** `translate`, 24×24, 12 frames;
   - **grow → shrink:** `disc12`;
   - **rotating bar:** `bar12`.
   - Hand-written rules (e.g. shift) are diagnostics or baselines only, kept separate from the
     video-only automatic results.
4. **Prefix generalization (diagnostic):**
   - a rule found on a short prefix is replayed on the full synthetic clip, and on a longer
     extension where one exists;
   - it says whether the dynamics' structure was captured, and is **not** compression evidence.
5. **Bad Apple:** ba32_3000_5, ba32_4500_5, ba32_4500_10 (same preprocessing as before: 32 px
   wide, box resize, threshold 128).
   - Longer clips only if one of these is promising, meaning its CA file is smaller than the
     smallest baseline.
   - No new clip selection is planned. If one happens, all candidates, including failures, are logged.

## 5. Judgement criteria
- **"Direct synthesis works":** the positive control is recovered exactly (Python rollout + Node file
  replay) at ≤ the known G.
- **"Smaller than the LUT→reduction route":** on the same clip, the direct-circuit file has fewer
  bytes than the smallest existing-route file (best of keep-exception LUT and ABC circuit, over all
  exact rules of that clip).
- **"Compression advantage on clip X":**
  - the CA file has fewer bytes than **every** baseline file on the same clip, i.e. the 3 recordings
    and 3 general-purpose compressors;
  - phrased as "beats the tested baselines on this clip", never as general or full-video compression.
- **Gate count vs size:** a smaller gate count and a smaller file are reported separately.
- **Single-run timing differences** under about 1.5× are not used as conclusions.

## 6. Verification
- **Every file is decoded from its bytes alone,** without the source clip:
  - **in Python:** every method;
  - **in Node:** every CA method, via `player/decode_file.mjs`, an independent bit reader and gate
    evaluator.
- **All T frames must match byte for byte** (SHA-256 over T×H×W bytes).
- **Two checks, labelled separately:**
  - **full truth-table equivalence:** only meaningful for LUT→circuit conversions;
  - **clip replay.**

## 7. Execution and budget
- **Ceiling:** **12 CPU core-hours, 1 GPU-hour.**
  - **GPU plan: 0.** There is no DiffLogic in this campaign and nothing goes to gpuq.
- **Definition of CPU core-seconds:**
  - the user + system CPU time of every process this campaign starts, **including its children**;
  - measured by the driver with `os.wait4` on each job (prlimit execs into the worker, so the waited
    pid is the worker);
  - interactive and verification commands are wrapped by the same meter
    (`experiments/c5_meter.py`);
  - everything is appended to `runs/c5/cpu_ledger.jsonl`.
  - **Threads:** one per solver process (CaDiCaL is single-threaded; OMP/BLAS threads = 1).
  - Concurrency is at most 16 processes at once.
- **Per-job limits:**
  - `prlimit --as` 16 GB (32 GB if a pilot shows it is needed, noted);
  - `prlimit --cpu` at the job's CPU cap;
  - a wall cap of 3× the CPU cap, KILL.
- **Allocation:**

  | part | core-h |
  |---|---|
  | implementation tests + pilots | ≤ 1.0 |
  | synthetic, positive control, certificates, prefix | ≤ 1.5 |
  | Bad Apple direct synthesis | ≤ 5.0 |
  | structure ablation | ≤ 2.0 |
  | files, baselines, verification | ≤ 0.5 |
  | reserve | 2.0 |

- **Stop rule:** search stops once the ledger reaches 9.5 core-hours, leaving room for verification
  and reporting.
- **Frozen code:** every search runs from a `tools/freeze.sh` snapshot, never edited during a run.
  Pilots, failures, encoding, preprocessing and verification are all on the ledger.
- **Untouched:** other projects, resident processes and all earlier results (runs/exp*).
- **Not done:** UI work, DiffLogic seeds, incremental re-runs.

---
## Pilot results (appended after the pilot, before the main runs; ledger 1.48 core-h)
Snapshots: `893056a8f9c7` (pilot), `f1f140b3c6fe` (after the bug fix and v3; used by every later run).

**Certificates** (`runs/c5/certs.json`):
- **Light cone:**
  - fails at S=1 for every Bad Apple clip, so no radius-1 rule works there with constant init;
  - fails at S ≤ 2 for bar12 and at S=1 for pcB.
- **State repeat:** disc12 cannot be reproduced by any memoryless rule (frames 1 and 5).
- **History conflicts** (F_priv, S=1, any k): disc12, bar12, pcB and every Bad Apple clip.
- **LUT-level feasibility from CHECKPOINT_02:**
  - ba32_3000_5 k0 S2 is UNSAT, and a LUT bound also bounds every circuit;
  - disc12 is feasible at F_shared k1 S1 and at k1 S2;
  - bar12 needs S=4.

**Positive control:**
- pcA (known 2 gates) recovered at G=2, with G=0 and G=1 UNSAT.
- pcB (known 4 gates) was SAT at G=3, below the known circuit, with G ≤ 2 UNSAT.

**Implementation error found by the pilot's own check:**
- the truth-table literal signs in `CircuitSAT._eval` were inverted (bit q referred to complemented inputs);
- the SAT/UNSAT statuses are unaffected (a bijection of the tables), but decoded circuits were wrong;
- pcB's pilot circuit failed its Python and Node replay, and the error was caught that way. pcA passed by symmetry (XNOR);
- fixed in `f1f140b3c6fe`, with a non-symmetric regression test (pcB, direct and v3).

**Scaling** (k1_S2, 60 s CPU cap): the largest G each run decided:

| clip | UNSAT up to (CPU) |
|---|---|
| ba32_3000_5 | G=3 (25 s) |
| disc12 | G=3 (4.8 s) |
| bar12 (k1_S4) | G=2 (19 s) |

- Everything above that, up to G=12, is UNKNOWN at 60 s.

**Bottleneck checks** (kept, all on the ledger):
- **v3 (one circuit copy per LUT entry, tied to the exp03 rollout CNF)** was **slower** on every instance it decided:

  | instance | v3 | v1 |
  |---|---|---|
  | disc12 G3 UNSAT | 27 s | 4.8 s |
  | ba32_3000_5 G3 UNSAT | 55 s | 25 s |
  | pcB G3 SAT | 15.5 s | 1.0 s |

  With v3, G=4–8 were UNKNOWN at 120 s and G=16 or 32 at 300 s.
- **Symmetry v2 (lex) vs v1:**
  - equal at G ≤ 3;
  - disc12 G4 UNSAT proved in 144 s with lex against 234 s without (one run, about 1.6×).
- **Conclusion:** the cost is in the search, not the encoding size. Proof time grows roughly 30–50× per gate, so UNSAT ladders can reach only G≈4–5.
- **Main runs therefore:** v1 encoding + lex.
  - Short UNSAT ladders give lower bounds.
  - Single-G SAT probes at G=8 and 12 cover the compression question. 12 gates ≈ the 195-bit change-list payload of ba32_3000_5, since a gate costs about 14 bits at n=10.

## Fixed budget (main)
All runs use the v1 encoding + lex, snapshot `f1f140b3c6fe`, with 16 GB, 1 thread and the wall cap at 3× the CPU cap. Each line below is a series; a ladder stops at its first SAT.

| clip | structure | G (CPU cap s) |
|---|---|---|
| ba32_3000_5 | k1_S2 (base) | 4 (1800); 8 (1800); 12 (1800) |
| ba32_3000_5 | k1s_S2 (neighbour hidden) | 1,2,3 (600 each); 12 (1800) |
| ba32_3000_5 | k2_S2 (2 hidden bits) | 1,2,3 (600); 12 (1200) |
| ba32_3000_5 | k1_S4 (more updates) | 1,2,3 (600); 12 (1200) |
| ba32_4500_5 | k1_S2 | 1,2,3 (600); 12 (1200) |
| ba32_4500_10 | k1_S2 | 1,2 (600) |
| disc12 | k1_S2 | 5 (1800); 12 (1200) |
| disc12 | k1s_S1 (neighbour hidden, S=1) | 1,2,3 (600) |
| bar12 | k1_S4 | 3 (1200); 12 (1200) |
| prefix diagnostics | translate24:p3 k0_S1, pcA:p3 k1_S1, pcB:p3 k1_S2 | 0..3 (60) |
| prefix diagnostics | disc12:p4 k1_S2 | 0..4 (300) |

**Totals:**
- **worst case** (every cap reached) 28,620 core-s = 7.95 core-h, so the ledger reaches at most 9.43 core-h, under the 9.5 stop;
- **remaining:** about 2.6 core-h for verification, files and the report.

**Fixed decisions:**
- Longer Bad Apple clips run **only** if a CA file beats every baseline on a clip above.
- **Not run:** v3; other gate bases; other initial states.
