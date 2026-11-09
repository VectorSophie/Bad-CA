# Campaign 04: SAT-centred (plan written before any run)

Date 2026-11-08. Results go to `docs/CHECKPOINT_04.md`. Every job runs from a frozen snapshot
(`tools/freeze.sh`). SAT and circuit work is CPU only, one thread per process, `prlimit` address
space and an outer `timeout -s KILL`. Only DiffLogic training goes through `gpuq -G 1`.
Family everywhere: F_priv k=1, S=2 (condition E), boundary 0, hidden init 0.

## Metrics (all parts)
- **Primary:** hard autonomous rollout error, exact yes/no, time to exact, rule size.
- **Statuses:** SAT / UNSAT (proof) / UNKNOWN_timeout. A timeout is never read as infeasibility or as
  evidence that a large rule is needed.
- **"Distance to an exact rule" is not measured against one known SAT solution.** Exact rules are
  many, and their hidden-state encodings can differ. Where a repair is run, its result is the **upper
  bound found** on the Hamming distance from the start rule to *some* exact rule. It is never called a
  minimum unless the solver proved optimality (UNSAT at E-1).

## Part 1: incremental vs cold-start SAT
- **Clip:** ba32, start frame 4500, lengths 5 / 10 / 20 / 40, the same for every arm (frames sliced
  from one decode).
- **Objective:** the first exact rule (feasibility), with no size objective.
- **Arms:**
  - **cold:** a fresh solver per length.
  - **incr-keep-step1:** one solver; frames added one at a time (L = 2..40); clauses and learned clauses kept.
  - **incr-prev-step1:** as incr-keep-step1, and after each solve every variable prefers its value in the previous model.
  - **incr-prev-sched:** as incr-prev-step1, but solving only at 5 → 10 → 20 → 40.
- **Phases:** the keep phase on LUT variables in every arm, as in exp03/exp03c. Phases are a
  **preference**, never a hard constraint.
- **Reported per length:** the step time (encode + solve for that length) and the **cumulative** time
  including every earlier length. Cold arms also get the sum over lengths. Lengths not reached are
  UNKNOWN.

## Part 2: SAT repair starting points
- **Starts:**
  - **keep;**
  - **Conway:** an explicit F_priv embedding, visible = Life(visible 3×3), hidden = keep;
  - **fixed random:** rng 0;
  - **learned hard rule:** a hardened DiffLogic LUT. R0 from exp07, plus the Part 4 rule if one exists.
- **Repair:** exp03c with REF = start.
  - The warm start is a **search preference**: decision phases prefer the start's value for each entry. It is **not a hard constraint**.
  - The anytime objective is Hamming distance to the start.
- **Measured at equal CPU budget:**
  - success;
  - time to the first exact rule (encode included);
  - the repair-distance upper bound over time;
  - final rule size (keep-exception bitstream, and the Part 3 circuit).
- **Learned starts:** the GPU cost that produced them is reported separately, from gpuq records.
- **Clips:** ba32_3000_5, where R0 exists. ba32_4500_10 is added if the pilot shows that every start
  is exact within seconds on ba32_3000_5, so that time-to-exact can actually separate the starts. That
  needs one R0 run on ba32_4500_10, at the same budget as exp07.

## Part 3: shrinking existing exact rules into logic/circuits
- **Tools:** existing ones, espresso (pyeda) for two-level and yosys/ABC (`abc -g`, 2-input gates)
  for multi-level.
- **Two equivalence levels, reported separately:**
  - **full truth-table equivalence:** all 2^10 inputs, every channel;
  - **clip-rollout equivalence:** only the LUT entries the clip's rollout consults must match, the rest are don't-cares.
- **Re-verification:** every reduced rule is re-verified by the independent Node player from the
  initial state (seed frame, hidden init, boundary), across the whole rollout. The player gets a
  circuit format of its own that evaluates gates directly, without Python's LUT.
- **Circuit size:** the decode-checked `costs.rule_circuit` bitstream. It counts every gate's op,
  every wire reference, the output references and constants (as gates).
- **Inputs:** the exact rules from CHECKPOINT_02/03 and this campaign.

## Part 4: binarization penalty (small, auxiliary)
- **Method BP:** R0 plus λ·Σ s(1−s) over every generation's state, with λ ramped linearly by step.
- **Setup:** the same 4 wiring seeds and 600 s as R0. Two λ_max values are piloted at 60 s and one goes
  to the main run.
- **Checkpoint selection:** by hard autonomous rollout. Lower penalty or soft loss is logged and **not**
  counted as success.
- **Its hardened rule** becomes another Part 2 learned start.
- **No seed scaling.**

## Pilot (before fixing budgets)
| part | what | budget |
|---|---|---|
| 1 | cold L5, L10; the 3 incremental arms up to L10 | 10 min wall each, CPU |
| 2 | keep / Conway / random / R0 repair on ba32_3000_5 | 120 s each, CPU |
| 3 | reduction smoke test on 2 rules | minutes, CPU |
| 4 | BP at λ_max 0.1 and 1, 60 s; R0 on ba32_4500_10, 60 s (timing) | ≈0.1 GPU-h |

**Ceiling for the whole campaign** (main budgets are fixed after the pilot in §Fixed budget below, and
not raised without a written note): **≤ 1.0 GPU-h, ≤ 20 CPU-h.**

## Pilot results (snapshot 70c0d482bb48)

**Part 1** (wall cap 600 s):
- **cold:** L5 15.7 s, L10 382.6 s (encode 26 s, solve 356 s). Peak RSS 5.3 GB at L10.
- **incr-keep-step1:** reached L9, cumulative 403 s.
- **incr-prev-step1:** reached L9, cumulative 282 s.
- **incr-prev-sched:** L5 at 15.6 s, then killed while solving L10.
- **Encoding:** about 2.9 s per added frame. Memory is about 0.4 GB per frame.
- **Solve time per step:** grows steeply, 36 → 63 → 271 s for L7 → L9 (keep).

**Part 2** (ba32_3000_5, wall cap 120 s): every start reached its first exact rule at 12.8–15.1 s, of
which about 11 s is encoding. Time to exact therefore **cannot separate the starts on this clip**, so
ba32_4500_10 is added as planned. On that clip the keep start took 383 s.

**Part 3:** the smoke test passed on 2 rules. The circuits for the min-exception rule are larger than
its keep-exception bitstream (2133 against 588 bits on the reach level). Custom ABC scripts fail
silently in the yowasp (wasm) build. Instead, the "heavier" flow repeats the default `abc -g` mapping
4 times (noted as a shortcut in `ca/reduce.py`).

**Part 4:** BP at λ_max 0.1 and 1 (ramp 800 steps) both ended at the hold error (0.0070) after 60 s,
like R0's own 60 s pilot. The pilot cannot choose between them on the hard metric, so **both go to
main**: two extra GPU jobs, instead of picking one by soft loss. BP runs about 13 steps/s against
R0's 28, because it rolls out every generation and evaluates the penalty. At an equal wall budget,
BP therefore makes fewer steps.

**R0 on ba32_4500_10** (60 s): hard error 0.0578 against hold 0.0616.

**GPU used by pilots:** 3 jobs × about 2.3 min wall ≈ 0.12 GPU-h.

## Fixed budget
| part | runs | budget per run | limits |
|---|---|---|---|
| 1 | cold L5, L10, L20, L40 | 1800 s wall each (as in Campaign 3) | 1 thread, `prlimit --as` 48 GB |
| 1 | incr-keep-step1, incr-prev-step1, incr-prev-sched, to L40 | **7200 s wall total** (= the cold arms' total, 4 × 1800) | 1 thread, 48 GB |
| 4 | BP λ_max 0.1 and 1 on ba32_3000_5, ramp 4000 steps | 600 s train, 4 seeds 1000–1003 | gpuq -G 1 |
| 2 | R0 on ba32_4500_10 (to produce the learned start) | 600 s train, 4 seeds 1000–1003 | gpuq -G 1 |
| 2 | ba32_3000_5: keep, Conway, random, R0, BP0.1, BP1 | 600 s wall each (as D2) | 1 thread, 24 GB |
| 2 | ba32_4500_10: keep, Conway, random, R0_4500_10 | 1800 s wall each | 1 thread, 24 GB |
| 3 | every exact exported rule | minutes | CPU |

**Estimate:**
- **GPU:** 3 main jobs × about 11.5 min ≈ 0.58 GPU-h, plus pilots 0.12, total **≈ 0.70 GPU-h**.
- **CPU:**
  - Part 1: about 2 h cold + 6 h incremental;
  - Part 2: about 3 h;
  - Part 3: under 0.5 h;
  - total **≈ 11.5 CPU-h**.

Both are within the ceiling.
