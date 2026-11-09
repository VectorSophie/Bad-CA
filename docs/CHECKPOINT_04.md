# Checkpoint 04: SAT-centred campaign

**Sources**
- **Plan:** `docs/CAMPAIGN_04.md`, written before the main runs; the restored original, see §Provenance.
- **Search code:** frozen snapshot `runs/frozen/9b8f0db4a7ba`.
- **Analysis and verification code:** snapshot `runs/frozen/64277b4ec640`.
- **Tables:** `experiments/summarize04.py exp09 exp10 exp11_all exp12`, saved to `runs/summary04.md`.

**Setting everywhere:** family F_priv k=1 (visible 3×3 plus the cell's own hidden bit, 10 inputs, 2 × 1024 LUT entries), S=2, boundary 0, hidden init 0. CPU solver CaDiCaL 1.9.5 (pysat), one thread per process.

**Statuses**
- **SAT** = an exact rule was found.
- **UNKNOWN_timeout** = no answer within the wall budget. This is **not** infeasibility, and **not** evidence that a large rule is needed.
- **BEST_SO_FAR_budget** = SAT, with the objective not proved optimal.

**Timing caveat:** all times are wall clock on a shared 128-core machine, with up to 10 searches running at once. The same cold L10 solve took 383 s in the pilot and 525 s in the main run. Single-run time differences below about 1.5× are not distinguishable.

## Part 1. Incremental vs cold-start SAT (ba32, start 4500, first exact rule, no size objective)

**Arms**
- **cold:** a fresh solver per length, 1800 s each.
- **incr-keep-step1:** one solver; frames added one at a time; clauses and learned clauses kept; keep phases.
- **incr-prev-step1:** as incr-keep-step1, but each variable prefers its value in the previous model (a preference, not a constraint).
- **incr-prev-sched:** solves only at 5 → 10 → 20 → 40.
- **Budget:** each incremental arm had 7200 s in total, equal to the cold arms' sum.

| L | cold: time for this length (s) | cold: sum 5..L (s) | incr-keep-step1: step / cumulative (s) | incr-prev-step1: step / cumulative (s) | incr-prev-sched: step / cumulative (s) |
|---|---|---|---|---|---|
| 5 | 16.1 SAT | 16.1 | 4.9 / 16.0 SAT | 5.8 / 16.4 SAT | 15.7 / 15.7 SAT |
| 10 | 525.0 SAT | 541.1 | 263.3 / 819.6 SAT | 850.6 / 1243.0 SAT | UNKNOWN_timeout (whole remaining budget) |
| 11 | – | – | 258.0 / 1077.6 SAT | 101.3 / 1344.2 SAT | – |
| 12 | – | – | 1282.5 / 2360.1 SAT | 3709.9 / 5054.1 SAT | – |
| 13 | – | – | UNKNOWN_timeout | UNKNOWN_timeout | – |
| 20 | UNKNOWN_timeout (1800 s) | UNKNOWN | not reached | not reached | not reached |
| 40 | UNKNOWN_timeout (1800 s) | UNKNOWN | not reached | not reached | not reached |

- **Cumulative cost is what counts,** and on that measure incremental did **not** beat cold start at L10. Cold reached L10 in 541 s (its 5 + 10 sum), against 820 s and 1243 s for the step-1 arms. The cold L10 run alone took 525 s.
- **What incremental bought** is the intermediate lengths L11 and L12, which the cold arm never attempted. It did not get further toward L20: both step-1 arms used the rest of their 7200 s on L13 without an answer.
- **Phase preference:** keeping the previous model's phases was slower than keep phases at L10 and L12 in this single run. The scheduled arm, which jumps 5→10 with previous-model phases, found nothing at L10 in about 7180 s, while cold L10 took 525 s. That is one run per arm, so it is a hint that previous-model phases can misdirect the search, not an established effect.
- **Memory:** peak RSS 5.3 GB (cold L10), 5.2 / 5.9 GB (step-1 arms at L12).
- **Node checks:** all 7 saved rules (L5 and L10 of every arm that reached them) were exported and replayed in the independent Node player; every hash matched the target (`runs/exp09/node_verify.json`). The L11/L12 rules were checked only by Python rollout at solve time, because the script saves rules only at the report lengths. **They are not Node-verified.**

## Part 2. SAT repair from different starting rules

**How the starting rule is used:**
- Exactness on the clip is the **only hard constraint**.
- The start enters as (i) decision phases preferring the start's value for every LUT entry, a **search preference**, and (ii) the anytime objective, Hamming distance to the start.
- So every number below is an **upper bound found** on the distance from the start to *some* exact rule. Exact rules are many and their hidden encodings differ, so this is not a distance to one known solution.
- No run proved optimality (that would need UNSAT at E−1), so none is a minimum.

**Starts** (all explicit F_priv k=1 LUTs, `experiments/exp10_repair_starts.py`):
- **keep:** each channel copies its own value.
- **Conway:** visible output = Life(visible 3×3), ignoring the hidden bit; hidden output = keep. With S=2 this is two Life steps per frame.
- **random:** `default_rng(0)`.
- **R0:** the hardened DiffLogic relaxation rule from exp07, or exp12 for ba32_4500_10.
- **BP0.1 / BP1:** the Part 4 rules.

**Main metrics:** hard rollout error is 0 for every SAT row (exact, Node-verified from the seed frame, hidden init and boundary).

| clip, budget | start | status | first exact s (encode s) | distance at first exact | best distance **upper bound** (at s) | rule bits (keep-exception) | GPU cost to produce the start |
|---|---|---|---|---|---|---|---|
| ba32_3000_5, 600 s | keep | SAT | 13.0 (11.8) | 237 | 99 (503) | **715** | – |
| | Conway | SAT | 13.4 (11.6) | 340 | 183 (288) | 1449 | – |
| | random | SAT | 15.4 (11.9) | 333 | 190 (249) | 2346 | – |
| | R0 | SAT | 14.8 (11.9) | 282 | **82** (540) | 1722 | 607 GPU-s (exp07 R0, 4 vmapped seeds) |
| | BP0.1 | SAT | 13.2 (11.9) | 239 | 83 (440) | 1117 | 614 GPU-s |
| | BP1 | SAT | 13.1 (11.8) | 234 | 96 (504) | 1104 | 614 GPU-s |
| ba32_4500_10, 1800 s | keep | SAT | **518.2** (26.5) | 573 | 572 (814) | **1940** | – |
| | Conway | **UNKNOWN_timeout_no_solution** | – | – | – | – | – |
| | random | SAT | 1562.5 (26.5) | 724 | 724 (1563) | 2258 | – |
| | R0 | SAT | 1671.7 (26.8) | 523 | 523 (1672) | 2173 | 608 GPU-s (exp12, hard err 0.0354 vs hold 0.0616) |

- **ba32_3000_5:** every start reached an exact rule 1.2–3.5 s after encoding, so time to exact does not separate the starts. The learned starts end with the smallest repair-distance bounds (82–96, against 99 for keep). Keep gives the smallest rule, because its distance objective is the size encoding itself.
- **ba32_4500_10:** keep was the fastest to exact, about 3× faster than random and R0, and Conway found nothing in 1800 s.
  - The learned start (R0, err 0.0354) did **not** shorten time to exact; it was the slowest of the three that succeeded.
  - Its first exact rule is the closest to its start (523) only in Hamming distance to itself, which is not a quality measure.
  - These are single runs; a 3× gap is above the timing noise noted above.
- **Learned starts cost GPU on top of CPU:** about 607–614 GPU-s each, not included in the CPU column.

## Part 3. Shrinking exact rules into 2-input-gate circuits (`experiments/exp11_reduce.py`, `ca/reduce.py`)

**Tools:** existing ones only. espresso (pyeda) for two-level, then yosys/ABC `abc -g` for 2-input gates. Flows: default, the default repeated 4 times (shortcut: the wasm ABC build ignores custom scripts), and each of these applied to `rule XOR keep`.

**Two equivalence levels, reported separately:**
- **full:** the circuit equals the LUT on all 2^10 inputs for every channel.
- **reach:** only the entries the clip's own rollout consults must match; all others are don't-cares. This is equivalence on this clip, not as a function.

**Every candidate is verified three ways:**
1. it is serialized to its decode-checked bitstream and decoded;
2. the decoded rule is rolled out in Python;
3. the independent Node player replays it **from the seed frame, hidden init 0 and boundary 0**. It evaluates the circuit gate by gate (circuit-v1) or reads the LUT (rule-v1), and must reproduce the target's SHA-256.

**Results:**
- **All checks pass** for all 49 exact rules: 33 from CHECKPOINT_02/03 and 16 from this campaign.
- **Circuit bits** include the header, the gate count, each gate's 4-bit op and its two wire references (⌈log2 pool⌉ bits each), the output references, and constants and inverters as gates.

Representative rows (full table: `runs/summary04.md`):

| rule | clip | consulted entries | full: LUT bits | full: best circuit bits (gates) | reach: LUT bits | reach: best circuit bits (gates) | best recording bits |
|---|---|---|---|---|---|---|---|
| exp03c min-exceptions | ba32_3000_5 | 193 | 588 | 2995 (168) | 588 | 2133 (125) | 195 |
| exp10 keep repair | ba32_3000_5 | 207 | 715 | 3535 (195) | 715 | 2133 (125) | 195 |
| exp10 BP1 repair | ba32_3000_5 | 217 | 1104 | 4275 (232) | 883 | 2415 (139) | 195 |
| exp10 random repair | ba32_3000_5 | 266 | 2346 | 18687 (856) | 1233 | 4035 (220) | 195 |
| exp03c min-exceptions | ba32_4500_5 | 422 | 1159 | 7373 (374) | 1159 | 5657 (296) | 678 |
| exp09 cold L10 | ba32_4500_10 | 629 | 1941 | 13311 (632) | 1941 | 10101 (498) | 1260 |
| exp03c X6 | bar12 | 762 | 2109 | 18377 (843) | 2109 | 14465 (680) | 825 |
| exp03c E | disc12 | 312 | 1143 | 6933 (354) | 1143 | 4975 (265) | 554 |

- **Circuits never beat the LUT encoding.** In every one of the 49 rules the best circuit has more bits than the same rule's keep-exception LUT encoding, by 1.4–9.0× at the full level and 2.7–7.4× at the reach level. Wiring references dominate: each gate costs 4 op bits plus 2 × 9–10 reference bits.
- **Reach cuts the circuit, not the LUT.** Using only the consulted entries changes the gate count by −74% to +8% (random repair: 856 → 220 gates; in one rule ABC's result was slightly larger, since the mapping is heuristic). The keep-exception LUT shrinks only when the exact rule changes unconsulted entries away from keep.
- **Smallest exact rule found, per clip:** ba32_3000_5 588 bits, ba32_4500_5 1159, ba32_4500_10 1859, disc12 1143, bar12 2109. **Every one is larger than the best recording of the same clip** (195 / 678 / 1260 / 554 / 825 bits). This holds whichever encoding, LUT or circuit, is used.

## Part 4. Binarization penalty (small auxiliary experiment, GPU via gpuq)

**Method BP** = R0 + λ·Σ s(1−s) over every generation's state, with λ ramped linearly over 4000 steps. Same 4 seeds (1000–1003), 600 s, checkpoint chosen by **hard autonomous rollout**.

| run | exact seeds | best hard err | hold err | selected at (it / s) | penalty at selection | steps |
|---|---|---|---|---|---|---|
| BP λmax 0.1, ba32_3000_5 | 0/4 | 0.0070 (= hold) | 0.0070 | 448 / 40 | 48.3 | 9916 |
| BP λmax 1, ba32_3000_5 | 0/4 | 0.0070 (= hold) | 0.0070 | 457 / 40 | 23.3 | 9849 |
| R0 reference (exp07) | 0/4 | 0.0057 | 0.0070 | 2619 / 100 | – | 16854 |

- **The penalty did not improve the hard rollout.** Both λ values never got below the hold error at any 20 s evaluation, so the selected checkpoint is an early one at 40 s.
- **Lower penalty and soft loss are logged but not counted as success.** At equal wall time BP also ran 0.59× the steps of R0, because it rolls out every generation.
- **As repair starts (Part 2)** the BP rules gave repair-distance bounds of 83 and 96. That is between R0 (82) and keep (99), so the penalty does not appear to help as a start either.

## Cost (whole campaign, including pilots)
- **GPU:** 6 gpuq jobs (49–54).
  - Pilots: 3 × 2.3 min.
  - Main: 3 × 11.35 min.
  - **0.68 GPU-h** in total; plan 0.70, ceiling 1.0.
- **CPU:**
  - Part 1: cold 4,141 s and incremental 3 × 7,200 s.
  - Part 2: 6 × 600 s + 4 × 1,800 s.
  - Pilots about 2,700 s; Part 3 reduction 350 s; verification a few minutes.
  - **About 11.1 CPU-h** in total; plan 11.5, ceiling 20.
- **Seed counts were not increased:** 4 per DiffLogic run, as in Campaign 3.

## Provenance and process notes
- **Files deleted and restored.** At 2026-11-09 10:25:56 KST the Campaign 4 code and `docs/CAMPAIGN_04.md` were deleted from the work tree, by the user's rollback of another session.
  - Only the 6 missing code files were restored, byte-for-byte from `runs/frozen/64277b4ec640`. Re-freezing the restored tree gives the same hash.
  - Their search-relevant code is identical to the snapshot the searches ran from (`9b8f0db4a7ba`). `exp11_reduce.py` differs only by the documented `target` fallback fix.
  - `CAMPAIGN_04.md` is the **original pre-run text**, recovered from a `cat` of the file captured in the session transcript. The file then had mtime 00:07:18, before the main runs started at 00:07:35, and the recovered file has the same byte size (7842).
  - Pre-restore file list, hashes and diff: `runs/c4_restore_20261109/`.
- **`runs/exp11b`**, created 02:10 KST, is of unestablished origin. It is preserved but **excluded** from every table here.
- **Errors in my process, kept separate from the findings:**
  - **Premature "done":** I first reported the runs as finished while they were still running, and checked partial results; those outputs were deleted and every check was redone after the runs ended.
  - **Waits that never ended:** my wait loops matched their own command lines in `pgrep`, so they kept waiting after the jobs had finished.
  - Neither error touched any search.

## Conclusions
1. **Exactness:** every exact rule came from SAT. Exact rules were found up to L12 for ba32_4500 (L5/L10 Node-verified, L11/L12 Python only). L20 and L40 remain **UNKNOWN**.
2. **Incremental SAT:** at equal cumulative budget, not faster than cold start at L10. It reached two more lengths (L11, L12), but no arm got past L12.
3. **Repair starts:**
   - They only matter on the harder clip. There keep reached exact fastest and Conway not at all, while random and R0 needed about 3× longer.
   - The learned start (GPU-trained, about 0.17 GPU-h) did not shorten time to exact. Its only measured advantage is a smaller repair-distance **upper bound** on the easy clip, and that distance is not a primary metric.
4. **Size:**
   - Circuits are always larger than keep-exception LUT encodings.
   - Reachable-only don't-cares shrink the circuits a lot but not below the LUT.
   - Every exact rule is larger than the best recording of its clip.
   - So compression by a rule is still **not shown** at these lengths.
5. **Binarization penalty:** no hard-rollout gain at this budget.

**Next (proposed, not run):** the open question is L ≥ 13. Neither more time nor incremental solving moved it.
- **(a)** A more compact SAT encoding: today about 0.4 GB and 2.9 s of encoding per frame. Encode the LUT as a circuit, or share clauses across cells with the same neighbourhood, and compare at equal budget.
- **(b)** A different family at equal size accounting, such as k=2 or S=3. CHECKPOINT_02 showed that some clips need a higher S.
