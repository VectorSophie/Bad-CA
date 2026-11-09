# Checkpoint 02: state-representation ablation (SAT + DiffLogic)

Date 2026-11-08. Every number below comes from a file under `runs/` produced by a script in `experiments/`.

## 0. Corrections to CHECKPOINT_01

Audit script: `experiments/exp01b_numeric_audit.py` → `runs/exp01b/audit.json`.

**Definitions.** All are per clip, for the Moore r=1 LUT family, boundary 0, 1 generation/frame, visible bit only, majority-vote LUT:
- `N = cells × (T−1)`: all observed one-step transitions.
- `changes = #{(t, cell) : f[t+1] ≠ f[t]}`.
- `minerr = Σ_contexts min(n0, n1)`: the **teacher-forced one-step** error of the best LUT in the family, attained by majority vote. This is an optimum over one-step predictions from *true* frames. **It is not a bound on autonomous rollout error.** Rollout feeds its own outputs back, so its error can be larger, and in principle a rule could even beat the majority rule's rollout. Autonomous rollout numbers are reported separately everywhere below.
- `explained_change = 1 − minerr / changes`: the net share of changed pixels the best memoryless LUT predicts (changes predicted, minus unchanged pixels it breaks, over changes).

**The "3%" claim.** It was the **unweighted mean over 5 clips** of `explained_change` at 64 px. Clips start at 10/30/62/100/150 s, 90 frames each, boundary 0. Exact value 3.25%. The **pooled** version, Σminerr/Σchanges over the same clips, is 2.48%. Per-clip range 0.60%–6.49%. Other widths, unweighted/pooled: 16 px 3.38/4.24%, 32 px 3.56/2.31%, 128 px 3.06/3.37%.

**"Floor = change rate" was an approximation, not an equality.** `minerr < changes` in every clip:

| 64 px clip | N | changes | minerr | changes predicted | unchanged broken | explained |
|---|---|---|---|---|---|---|
| t10s | 273408 | 10199 | 9935 | 532 | 268 | 2.59% |
| t30s | 273408 | 3711 | 3470 | 777 | 536 | 6.49% |
| t62s | 273408 | 13828 | 13484 | 928 | 584 | 2.49% |
| t100s | 273408 | 1007 | 966 | 76 | 35 | 4.07% |
| t150s | 273408 | 9425 | 9368 | 106 | 49 | 0.60% |

**"100% conflicting" replaced by counts** (24×24 synthetic clips):

| clip | N | changes | minerr | conflicting contexts / seen | changes predicted | unchanged broken |
|---|---|---|---|---|---|---|
| translate | 6336 | 132 | 0 | 0 / 26 | 132 | 0 |
| grow→shrink | 5760 | 288 | 260 | 60 / 62 | 64 | 36 |
| rotate bar | 6336 | 446 | 440 | 74 / 74 | 26 | 20 |

grow→shrink: 60 of 62 contexts conflict, and the best LUT nets 28 of 288 changes (9.7%). Rotate: every seen context conflicts, and the best LUT nets 6 of 446 changes (1.3%). The earlier phrase "100% conflicting" was wrong for grow→shrink.

## 1. Setup of the ablation

- **Rule family F_priv(k)** (primary). LUT inputs are the visible Moore 3×3 (bits 0–8, row-major, centre = bit 4) plus the cell's **own** k hidden bits (bits 9…). Outputs are the next visible bit and the next k hidden bits, one LUT per output channel, shared by all cells. Neighbours cannot read hidden bits.
- **F_shared(k)** (extra, labelled X4): neighbours also read the hidden bits (Moore 3×3 of every channel; 18 inputs for k=1).
- **Boundary:** constant 0 for every channel. **Hidden init:** all 0, declared and stored as 0 seed bits. **Visible seed:** frame 0 of the clip.
- **Observation:** only the visible channel, compared with the target after every S generations. **Sampling cadence:** every source frame (stride 1). Synthetic clips have no timestamps; Bad Apple frames sit at 1/30 s.
- **Conditions** (k = hidden bits per cell, S = generations per displayed frame):

| label | family | k | S | LUT inputs | LUT outputs |
|---|---|---|---|---|---|
| A | memoryless (visible only) | 0 | 1 | 9 | 1 |
| B | F_priv | 1 | 1 | 10 | 2 |
| C | F_priv | 2 | 1 | 11 | 3 |
| D | memoryless | 0 | 2 | 9 | 1 |
| E | F_priv | 1 | 2 | 10 | 2 |
| X1 (extra) | F_priv | 2 | 2 | 11 | 3 |
| X2 (extra) | F_priv | 1 | 3 | 10 | 2 |
| X3 (extra) | memoryless | 0 | 3 | 9 | 1 |
| X4 (extra) | F_shared | 1 | 1 | 18 | 2 |
| X5 (extra) | memoryless | 0 | 4 | 9 | 1 |
| X6 (extra) | F_priv | 1 | 4 | 10 | 2 |

- **Success:** exact reproduction of every frame by an autonomous rollout from the declared seed. Every SAT model is re-simulated by `ca/rule.py` and independently replayed by `player/replay.mjs` (Node, format `ca-universal/rule-v1`). The Node frames are compared with the **target** frames, not only with the Python output.
- **Targets:** `disc12` (12×12 disc, radii 2,3,4,5,4,3,2, 7 frames) and `bar12` (12×12 bar, half-length 5, rotating 15°/frame, 12 frames), both in `ca/synthetic.py`. Bad Apple `ba32_<start>_<len>`: 32×24, threshold 128, **starting at explicitly chosen non-uniform frames. The black opening (frames 0–42) is skipped on purpose; the full-song start condition remains a separate open decision.**
- **SAT:** CaDiCaL 1.9.5 via python-sat, default options, deterministic, so there are no random seeds. Each config is a separate CPU process under `prlimit` (16–64 GB address space, 1800 s CPU) plus a 1800 s wall `timeout -s KILL`. **No SAT job went through gpuq.** Statuses are kept distinct: `SAT`, `UNSAT` (solver proof), `UNKNOWN_timeout`, `UNKNOWN_memout`, `ERROR`. The classifier was tested with a forced 3 s timeout, which came back as `UNKNOWN_timeout`.
- **Encoder validation** (`tests/test_sat.py`): translation SAT; Conway observed every 2nd generation SAT; 3 planted random F_priv(1) rules SAT, each re-simulated exactly; disc12 memoryless UNSAT.

## 2. Infeasibility certificates (cheap, solver-independent)

All three are **necessary conditions** for exact reproduction, implemented in `ca/sat.py`. Each one is a proof only **within its stated scope**. Where a certificate fires, the SAT solver independently returned UNSAT too.

| certificate | scope (the claim holds only under these assumptions) | fires on |
|---|---|---|
| **State repeat** | Any deterministic, autonomous, spatially homogeneous update whose **full state is the visible bits only** (k=0). Any neighbourhood radius, any S, any boundary rule. | disc12 frames 1 and 5 are identical and are followed by radius 4 vs radius 2, so **A, D, X3 (and X5) are infeasible for disc12 at every S.** |
| **History conflict** | F_priv(k) for **any k** (hidden bits read only by their own cell). S=1, every frame observed. Hidden init uniform constant 0. Boundary constant 0. Synchronous, deterministic, homogeneous. A cell's hidden state is then a function of its own observed 3×3 history, so two cells with identical histories but different next values make the clip infeasible. | disc12: 4 conflicting cell-times (from t=2); bar12: 10 (from t=0); every Bad Apple clip in exp05. **B and C (and A) are infeasible for these clips for every k.** It does **not** cover F_shared (X4) or S ≥ 2. |
| **Light cone** | Any Moore radius-1, synchronous, deterministic, homogeneous rule. **Any number and visibility of hidden channels**, hidden init uniform constant 0, boundary constant 0, fixed S. Frame t at a cell is then a function of frame 0 within Chebyshev radius t·S, so cells whose radius-t·S frame-0 patches are identical must agree at frame t. | bar12: conflicts for S ≤ 3 (S=3: 2 cells at frame 1), none from S=4. **So for bar12 every condition with S ≤ 3 is infeasible, including F_shared.** Bad Apple exp05: the minimum S that passes is 2–4 at 32 px and 4 to >6 at 64 px. |

Outside these scopes (e.g. wrap boundary, random or learned hidden init, radius-2 rules), the certificates say nothing. Passing a certificate does **not** imply feasibility.

## 3. SAT results (exact synthesis)

Status meanings:
- **SAT** = an exact rule was found and verified.
- **UNSAT** = the solver proved that no rule in that family/setting reproduces the clip (CaDiCaL proof, not independently checked; the certificates above cross-check most cases).
- **UNKNOWN_timeout** = no answer within the limit. **This is not evidence of infeasibility.**

Full table with vars, clauses and times: `runs/exp03/summary.md`. Brackets give the certificate that independently explains an UNSAT; "solver" means the UNSAT rests on the solver proof alone.

| target | A (k0,S1) | B (k1,S1) | C (k2,S1) | D (k0,S2) | E (k1,S2) | extras |
|---|---|---|---|---|---|---|
| disc12 | UNSAT [repeat] | UNSAT [history] | UNSAT [history] | UNSAT [repeat] | **SAT** 3.3 s | X1 (k2,S2) SAT · X2 (k1,S3) SAT · X3 (k0,S3) UNSAT [repeat] · **X4 (F_shared k1,S1) SAT** |
| bar12 | UNSAT [history, light cone] | UNSAT [history, light cone] | UNSAT [history, light cone] | UNSAT [light cone] | UNSAT [light cone] | X1/X2/X3 UNSAT [light cone] · X4 UNSAT [light cone] · **X6 (k1,S4) SAT** 603 s · X5 (k0,S4) **UNKNOWN_timeout** 1795 s |
| ba32_3000_5 | UNSAT [history, light cone] | UNSAT [history, light cone] | UNSAT [history, light cone] | UNSAT [solver] | **SAT** 2.0 s | |
| ba32_4500_5 | UNSAT [history, light cone] | UNSAT [history, light cone] | UNSAT [history, light cone] | UNSAT [solver] | **SAT** 1.9 s | |

- Every SAT model: Python re-simulation exact, Node replay hash equal to the target hash, and Node frames byte-equal to the target frames.
- X5 is **not** a negative result. The memoryless S=4 bar is unresolved under 30 min.
- Takeaways:
  - **Steps per frame and memory are not interchangeable here.** k bits at S=1 never helped (history certificate). k=0 at S=2 never helped. k=1 with S=2 (S=4 for the bar) did.
  - For the disc, letting neighbours read the hidden bit (X4, S=1) substitutes for the second generation.
  - These are feasibility results for 5–12-frame clips under the stated family, boundary and init. They are not claims about longer sequences or other families.

## 4. Rule size: exact rules found so far are larger than a recording

`runs/exp03/size_vs_recording.md`, `experiments/exp03c_min_exceptions.py` → `runs/exp03c/`.

- **Size measure.** A rule is described as "keep" (each output copies the cell's own current bit) plus a list of **exception** LUT entries, at ⌈log2(C·2ⁿ)⌉ bits each.
- **Baseline.** The change-list recording costs ⌈log2 cells⌉ bits per changed pixel, plus a per-frame count.
- **Search.** Exact SAT plus a totalizer bound gives an anytime descent. Every improvement is saved immediately. Run: 25 min wall per config, CPU, `prlimit` 48 GB.

| config | optimality | first E | best E (at s) | rule bits | change-list bits | visited entries | Node exact |
|---|---|---|---|---|---|---|---|
| disc12 E | upper bound only (no proof) | 317 | 201 (334) | 2211 | 1136 | 312 | ✓ |
| disc12 X1 (k=2,S=2) | upper bound only (no proof) | 613 | 312 (1017) | 4056 | 1136 | 403 | ✓ |
| disc12 X4 (F_shared) | upper bound only (no proof) | 303 | 175 (1481) | 3325 | 1136 | 442 | ✓ |
| bar12 X6 | upper bound only (no proof) | 759 | 667 (1480) | 7337 | 1208 | 762 | ✓ |
| ba32_3000_5 E | upper bound only (no proof) | 237 | 76 (1491) | 836 | 170 | 193 | ✓ |
| ba32_4500_5 E | upper bound only (no proof) | 561 | 217 (1109) | 2387 | 780 | 422 | ✓ |

- **No config reached a proven optimum.** The search never proved "≤ E−1 exceptions is UNSAT". Every E above is an **upper bound** on the minimum exception count. It is not a minimum, and failing to go lower within 25 min is **not** evidence that the minimum is large.
- **Every exact rule found is 2–6× larger than recording the changes.** This holds after a 2–3× reduction from the objective.
- On 5–12-frame clips this is close to unavoidable, because a change list is cheap when there are few changes. A rule can only win by being *reused* across many frames. **So compactness cannot be tested on these clips. It needs rule size measured against clip length** (next experiment).
- Unconstrained SAT and SOP minimization (`runs/exp03/summary.md`) gave 83–258 terms. Phase biasing toward "keep" reduced some rules and enlarged others, so it is not a reliable minimizer.

## 5. DiffLogic: feasibility vs capacity vs optimization, kept separate

GPU work went through `gpuq -G 1` only. 16 seeds, vmapped. Seed s uses wiring permutations `default_rng(1000+s)`; logits use a deterministic pass-through init. Both are recorded in each `runs/exp04/*/result.json`.

**Implementation error found and fixed.** My first DiffLogic (5 layers, random wiring, mild pass-through init) **could not learn Conway's 512-entry LUT exactly**: ≤83% hard accuracy over 12 hyper-parameter settings (`runs/exp04/v1_shallow/distill_calibration_v1.json`). The reference paper learns it, so the fault was in my implementation, not in the CA problem. I replaced it with the reference notebook's architecture (google-research/self-organising-systems@3d5547ca, Apache-2.0; attributed in `ca/difflogic.py`):
- 16 centre-paired perception kernels (8→4→2→1);
- a 16×128 update net with unique wiring, tapering 64…4;
- pass-through logit 10;
- AdamW(b2 0.99, wd 1e-2), value clip 100.

Inputs are restricted so the family equals SAT's F_priv. **Calibration v2: Conway learned exactly (3/8 seeds by step 1500–2000); a random 10-input / 400-entry partial function also exact (3/8).** All v1 DiffLogic results are void and archived in `runs/exp04/v1_shallow/`.

**Capacity** (supervised distillation of the phase-biased SAT rule's visited entries; 5000-step cap):

| reference | care entries | best hard acc | exact seeds |
|---|---|---|---|
| ba32_3000_5 E | 314 | **1.000** | 2/16 |
| disc12 E | 396 | 0.9987 | 0/16 |
| ba32_4500_5 E | 625 | 0.979 | 0/16 |
| bar12 X6 | 819 | 0.957 | 0/16 |

This shows the architecture can represent a known exact solution for ba32_3000. For the other three, capacity is plausible but **not demonstrated**.

**Optimization** (frames only, sum-of-squares loss on the autonomous soft rollout, hardened rule evaluated by exact LUT rollout; best of 16 seeds; 25-min budget for feasible configs, 10 min for infeasible ones):

| config | SAT | steps | hard err | hold err | change-IoU | soft (thresholded) err | live gates | GPU s |
|---|---|---|---|---|---|---|---|---|
| ba32_3000_5 E | SAT | 5500 | 0.0060 | 0.0070 | 0.08 | **0.0010** | 232 | 1550 |
| ba32_4500_5 E | SAT | 5500 | **0.0208** | 0.0393 | **0.31** | **0.0021** | 513 | 1524 |
| disc12 E | SAT | 10000 | 0.194 | 0.187 | 0.10 | 0.127 | 5 | 1509 |
| bar12 X6 | SAT | 5750 | 0.140 | 0.174 | 0.03 | 0.107 | 32 | 1520 |
| disc12 B | UNSAT | 5000 | 0.214 | 0.187 | 0.26 | 0.051 | 36 | 613 |
| bar12 E | UNSAT | 3500 | 0.116 | 0.174 | 0.17 | 0.064 | 125 | 652 |

All Node replays matched the Python rollout of the exported hard rule.

**0 of 64 seed-runs (4 feasible configs × 16 seeds) found an exact rule.** The causes differ by clip and should be reported separately:
- **Bad Apple clips: hardening gap.** The soft rollout is almost exact (0.001 / 0.002), but the hardened rule is 6–10× worse. With soft hidden values the "1 hidden bit" carries a graded number, and hardening removes that. The objective trains a different model from the one exported. *Optimization/relaxation failure, not infeasibility* (SAT says a solution exists; capacity was shown for ba32_3000).
- **disc12 E, bar12 X6: soft training itself stalls** (0.127 / 0.107), and the best hard rule is at or near hold. *Optimization failure*: SAT proves feasibility, capacity is not shown.
- The infeasible configs still beat hold on change-IoU (0.17–0.26), but these are approximations by construction.

## 6. Budget and implementation errors (all counted)

- **GPU:** 11,074 GPU-s (3.08 GPU-h) in total, from gpuq records. 3,457 s was smoke test, pilots, calibration and the **void v1 architecture**; 7,616 s was reference-architecture runs.
- **CPU SAT:** ablation 10 synthetic + 10 Bad Apple configs at ≤ 30 s each, except bar12 X6 (627 s) and X5 (1795 s timeout). Phase-biased reruns took ≤ 7 min. Min-exception search: 6 × 25 min, plus a void first round. Minimization: ≤ 7 min per rule.
- **Implementation errors**, all fixed and kept separate from the findings:
  - v1 DiffLogic architecture (see §5).
  - The first min-exception run relied on `Cadical195.interrupt()` from a timer thread. It was ignored: all runs overran into the 1900 s KILL, and one aborted with rc 134. Nothing was saved, so that round is void (`runs/exp03c/v1_interrupt_failed/`). The fix is to save each improvement immediately and end the run with the outer KILL.
  - An unbounded totalizer over 524k literals (F_shared) gave `std::bad_alloc` under 48 GB, a resource limit and not infeasibility. Fixed by bounding at the first solution's E.
  - Job 21 was killed by me (my own job) because it may have started on a half-edited script.

## 7. Conclusions and the next experiment

1. **State representation is decided by checkable certificates.** At S=1, private hidden memory never helps (history conflicts), and S must exceed the motion light cone. With k=1, S=2 both tested 5-frame Bad Apple clips (32 px; S=2 passes the light cone for these two) became exactly feasible. Two clips is not a general claim.
2. **Exact feasibility ≠ compactness.** Every exact rule found is larger than a recording of the same clip.
3. **DiffLogic's failure is optimization, not representation.** For Bad Apple specifically it is the soft→hard gap.

**Next (proposed, not run):**
- **(a) Compression scaling.** Same Bad Apple region (start frame 4500, 32 px, k=1, S chosen by the light cone), clip length 5/10/20/40 frames. Measure min-exceptions rule bits against change-list bits as length grows. This answers "rule or disguised recording?". Budget 25 min CPU per length. Stop when the rule/recording ratio's trend is clear or the length becomes SAT-intractable.
- **(b) Close the hardening gap.** Hard-forward / straight-through hidden channels (binarize hidden state in the forward pass) or a hard-rollout loss term. Baseline is this checkpoint's soft-trained runs at equal GPU-seconds. Also try SAT repair seeded from the DiffLogic hard rule (phases = DiffLogic LUT).
- **Still open:** the song's uniform black opening; the full-song start condition is undecided.
