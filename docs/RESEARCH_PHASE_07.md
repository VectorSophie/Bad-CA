# Research phase 07: widening the method space (survey, candidates, pilot plan)

Written 2026-11-09, **before any Phase 07 pilot run**. The results go to `docs/CHECKPOINT_07.md`.

**Restated goal:** find a *simple* local CA rule automatically from a video, so that from the first
frame onward the CA reproduces that video autonomously.
- Compression stays an evaluation axis but is no longer the only success condition.
- A short, readable rule that reproduces recognisable motion approximately is a valuable result.

## 1. What is established, and what is not (CHECKPOINT_01–06)

| claim | status | scope |
|---|---|---|
| Exact 5-frame Bad Apple replay exists | **shown** (SAT, Node-verified) | F_priv k=1, S=2, hidden init 0, boundary 0; ba32_3000_5, ba32_4500_5 |
| Exact replay up to 12 frames | shown at L ≤ 12 (ba32_4500); L ≥ 13 **UNKNOWN** (timeouts) | same family, LUT SAT, 30 min – 2 h |
| No radius-1 rule works at S=1 on Bad Apple | **proved** (light cone) | any radius-1 rule with constant init and boundary |
| Memoryless rules fail on clips with repeated frames | proved (state repeat) | k=0, any S |
| Exact rules smaller than recordings | **not shown** on any natural clip | every exact rule found is 1.4–2.8× the best baseline file |
| Small circuits (≤ 12 gates) on ba32_3000_5 | ≤ 4 gates impossible (UNSAT); 5–12 **UNKNOWN**; 12/22 cubes of G=12 UNSAT | 2-input arbitrary gates, same family |
| LUT → ABC circuits | always larger than the LUT encoding (≥ 125 gates) | heuristic ABC mapping, not a minimum |
| DiffLogic (relaxation) | 0/4 exact on ba32_3000_5; hard error 0.0057 vs hold 0.0070; soft/hard gap comes from graded states | 600 s, 4 seeds, 5 frames only |
| Straight-through, binarization penalty | collapsed to the hold rule at that budget | 600 s, 4 seeds |
| Incremental SAT | not faster than cold start at equal cumulative budget | ba32_4500, one run per arm |
| Approximate replay of a **long** clip (≥ 1 s) | **never tested** | – |
| Human-readable rules (rule lists, totalistic rules, trees) | **never tested** | – |
| Motion-aware state (previous frame, direction channels) | **never tested** as a structure; only the F_priv/F_shared ablation | – |

**What the failures are limited to:**
- **Input structure:** the 3×3 Moore neighbourhood, plus k ≤ 2 private or shared hidden bits.
- **Initial state:** hidden init is a uniform 0.
- **S:** S ∈ {1, 2, 4}.
- **Representation:** a full LUT, or a circuit of arbitrary 2-input gates.
- **Learning:** exact SAT/CNF search, DiffLogic relaxation (≤ 600 s, 5 frames) and local hill-climbing.
- None of this is evidence against other rule languages, approximate objectives, longer clips,
  memory of the previous frame, or evolutionary search over small programs.

**Separate questions, not to be confused:**
- **(a)** exact replay of 5 frames;
- **(b)** autonomous replay of a long clip, exact or approximate;
- **(c)** compression;
- **(d)** human-understandable rules.

Phases 01–06 addressed (a) and (c). Phase 07 addresses (b) and (d), with (a) and (c) as constraints that
are reported.

## 2. Survey: papers and code, what each actually did

**Black opening:** the canonical clip opens with 43 black frames, and there are 182 uniform frames in
total. A uniform frame plus a uniform hidden init makes every cell identical, so no homogeneous rule
can break the symmetry. All Phase 07 targets therefore start at **non-uniform frames**. This does
**not** solve the full-song problem, and no claim about it is made.

### 2.1 Papers (what they did ≠ what we infer)

| work | what it actually did | what we take from it / inference | not shown by it |
|---|---|---|---|
| Miotti et al. 2025, *DiffLogic CA* (arXiv 2506.04912) | learns logic-gate CA (relaxed, then hardened) for Game of Life and for generated patterns | our existing gradient path; recurrent training, hardening | arbitrary target video |
| Petersen et al. 2022 (2210.08277); Conv-DLGN 2024 (2411.04732) | differentiable choice among 16 ops; residual init | gate parameterisation | recurrent rollouts |
| Yousefi et al. 2025, *Mind the Gap* (2506.07500, NeurIPS) | Gumbel noise on the gate logits + straight-through: hard forward, soft backward; 98% smaller discretisation gap | a principled hard-forward training alternative to our ST failures | **feed-forward CIFAR only**, not recurrent |
| Richardson et al. 2024, *Learning spatio-temporal patterns with NCA* (PLOS CB, 2310.14809) | continuous NCA trained on image time series and PDE trajectories | from the code (`NCA/trainer/NCA_trainer.py`, `train_step`, `train_sequence`): the sequence is cut into chunks of `iter_n` steps; each chunk starts from the model's *own* previous chunk end state; with `INJECT_TRUE`, one batch gets the true state; per-step range regulariser; gradients normalised per tensor | continuous states only; no discrete rule |
| Pajouheshgar et al. 2023, *DyNCA* (CVPR, 2211.11417) | dynamic **texture** synthesis: appearance loss + optical-flow motion loss; multi-scale perception + positional encoding | motion can be learned as a statistical loss | **not** exact choreography of a target video; uses a positional encoding |
| Gilpin 2019, *CA as CNNs* (PRE; `williamgilpin/convoca`) | a CNN learns a CA rule from image sequences of **CA-generated** data | from `train_ca.py` (`initialize_model`): conv + 1×1 layers, one-step supervised classification, optional totalistic constraint | no hidden state; no arbitrary video |
| Elser 2021 (2012.02179) | divide-and-concur constraint projection recovers rules and unobserved intermediate states of **CA-generated** data | a constraint view of hidden intermediate steps, close to our SAT | arbitrary footage |
| Inoue, Ribeiro, Sakama 2014, LFIT; `pylfit` (GULA, PRIDE) | learns minimal logic rules from **fully observed** state transitions | from `_pride.py`: a covering learner, correct and minimal (no condition can be dropped); seeds from an uncovered positive, adds conditions until no negative matches | latent state; GPL-3.0 |
| Evans et al. 2021, *Apperception Engine* (AIJ; eval 2007.05367) | ASP (clingo) search for causal rules, initial conditions and constraints over symbolic sensory sequences; invents latent objects/predicates | the idea of a short rule language plus invented hidden state; ECA 249/256 correct | inputs under 300 bits, 4 h limit per task: far below a 48×36×60 video |
| Mitchell, Crutchfield, Das 1996; GA/GP on CA rules (Sapin 2003; Öztürkeri & Johnson 2011, CGP for CA self-assembly) | evolutionary search over rule tables or programs | **discrete search with a non-differentiable (rollout) fitness** | video reproduction |
| Zenil et al. 2019, *Causal deconvolution* (Nat. Mach. Intell.) | algorithmic-probability decomposition of patterns into generating programs, tested on CA data | MDL framing of the "simple rule" trade-off | a practical search for our size |
| Second-order CA (Toffoli–Margolus; Alonso-Sanz memory CA) | state t+1 = f(neighbourhood t, own or neighbourhood state t−1) | motion and velocity from a **known** previous state, with no latent search | learning from video |

### 2.2 Implementation inventory (read for this phase)
GitHub page fetches timed out, so the source was read through `gh api` contents. Nothing was executed.

| repo | license | last push | stack | files / functions read | usable element |
|---|---|---|---|---|---|
| AlexDR1998/NCA | none stated | 2024-05 | TensorFlow | `NCA/trainer/NCA_trainer.py`: `train_step`, `train_sequence` | chunked rollout from the model's own states + true injection (idea only) |
| williamgilpin/convoca | none stated | 2020-09 | TF2 (+ TF1 archive) | `train_ca.py`: `initialize_model`, `logit_to_pred`, `augment_data` | one-step supervised baseline (idea only) |
| Tony-sama/pylfit | GPL-3.0 | 2026-11 | Python | `algorithms/_pride.py`: docstring + `_fit_target_value` | covering-rule learning idea (no code copied, because of GPL) |
| xcontcom/evolving-cellular-automata | MIT | 2025-12 | JS | `single.js`: 512-bit rule, test/countpoints loop | GA over full rule tables (idea only) |
| RichardEvans/apperception | none stated | 2021-05 | Haskell + clingo | metadata only | not used: too small a scale |
| google-research/self-organising-systems (local `refs/sos`), maxencefaldor/cax, rplsmn/logicars | Apache-2.0 / MIT / MIT | pinned in CP01 | JAX / Rust | read in CP01–03 | our DiffLogic implementation follows the reference |

Building third-party solver binaries was blocked by the environment's permission policy in Phase 06. In
this phase **only our own code runs**; external code informs design only.

## 3. Candidate rule representations

"Rule count" is never used alone. Every result reports, for its representation:
- the number of rules or conditions;
- operations and constants;
- wiring;
- table entries;
- **the bytes of the actual stored file**.

A "10-rule" result must not hide a large table or the video itself inside one rule.

| # | representation & search | how motion is represented | difference from Phases 01–06 | inductive bias / hard cases | refs, code | pilot here? | what it gives up |
|---|---|---|---|---|---|---|---|
| C1 | Boolean circuit, exact SAT | anything, via hidden bits | – (done) | none; search explodes | ours | no (frontier studied) | – |
| C2 | **Outer-totalistic (Life-like) B/S rules**, exhaustive enumeration of all 2^18, approximate rollout objective | isotropic growth/erosion of regions only | a new rule language: 18 bits, readable as "born with n neighbours / survive with m"; exhaustive, so a global optimum within the language | isotropic, memoryless: no direction, no translation, no repeated-frame logic | Gilpin's totalistic option; Life-like literature | **yes** | direction, memory, exactness on most video |
| C3 | **Second-order CA + rule lists / decision trees** induced supervised (CART-style; DAgger-style re-labelling on its own rollouts) | the previous frame is a known hidden channel (hidden′ = centre), so velocity and direction become local differences | **hidden state no longer latent:** learning is supervised classification; rule = short if-then list | needs motion ≤ 1 px/step at S=1 (the light-cone limit); prior-frame memory only | 2nd-order CA (Toffoli–Margolus); LFIT/PRIDE covering idea; DAgger (Ross et al. 2011) | **yes** | latent counters/phase; exactness where >1 px/frame motion |
| C4 | **Evolutionary discrete program search: CGP** over small circuits with hidden bits, fitness = hard autonomous rollout error | anything the hidden bits and gates can express | no relaxation, so no soft/hard gap; it optimises the real objective; small gate budget from the start | random search; weak on needle-in-haystack exactness | CGP (Miller); GA/GP for CA | **yes** | efficiency; no optimality proof |
| C5 | **DiffLogic, long-horizon approximate** | learned hidden channels | the same method on a **new question** (60-frame approximate replay, not 5-frame exact) | soft/hard gap (CP03) | Miotti et al.; our `ca/difflogic.py` | **yes (GPU)** | simplicity: hundreds of gates |
| C6 | Continuous NCA → extract | continuous hidden fields | not a binary rule unless extracted; extraction is unsolved | smooth fields | Richardson; Growing NCA | no: extraction is the whole problem; deferred | binary simplicity |
| C7 | Multi-scale / hierarchical CA, positional encodings (DyNCA) | coarse-to-fine | changes the "single local rule from the first frame" setting | – | DyNCA | no (separate setting) | the original autonomy criterion |
| C8 | ASP/ILP with invented predicates (Apperception) | invented latent objects | symbolic, with latent state | scale (< 300 bits input) | Evans | no: scale is 5 orders too small | – |

**Chosen for pilots: C2, C3, C4 and C5.** These are four different search paradigms:
- exhaustive enumeration over a tiny language;
- supervised induction with a known memory;
- evolutionary search over programs;
- gradient relaxation.

None is a solver, seed or hyper-parameter variant of another.

## 4. Pilot plan

### 4.1 Common targets and preprocessing
- **Preprocessing:** as before (`ca/video.py`), a box resize to width w, then threshold 128, from the
  canonical mp4 (6572 frames, 30 fps).
- **Synthetic (operation checks):**
  - `translate24` (moving block);
  - `disc12` (grow → shrink; memoryless impossible);
  - `bar12` (rotating bar; S ≥ 4 for radius 1).
- **Bad Apple, short exact case:** `ba32_3000_5`.
- **Bad Apple, long common task:** **`ba48_4500_60`**, frames 4500–4559 (2.0 s), 48×36, a
  morphing silhouette. It is non-uniform from the first frame (§2), so it skips the black opening.
  - Hold-frame error 0.0204 mean change per frame.
  - Chosen from a scan of 21 segments for clear but moderate motion and no uniform frames. The other
    candidate (1800) has thin, fast details. The scan is in `runs/c7/media/contact_ba48_*.png`.
- **Model and boundary:**
  - autonomy as before: one fixed local rule, visible init = first frame, hidden init = a uniform
    constant 0, boundary 0, fixed S;
  - no frame injection at replay.
  - Every extension is marked as such. C3's hidden = "previous frame" is still the autonomous model:
    the hidden channel is part of the rule's state, initialised to 0.

### 4.2 Evaluation (no single combined score)
- **Exact:** byte match on every frame, plus the independent Node player on an exported file.
- **Approximate**, per frame t ≥ 1:
  - pixel error;
  - **hold-frame error** (frame 0 frozen);
  - error ratio = err / hold err;
  - **change-IoU:** predicted vs true changed pixels between t−1 and t;
  - **spatially tolerant error:** a pixel counts only if no pixel of the predicted value lies in the
    target's 3×3 neighbourhood. This separates 1-px offsets from wrong content;
  - **temporally tolerant error:** the minimum over target frames t−1..t+1;
  - **horizon:** the first frame whose error exceeds hold.
  - These are reported as curves (drift) and means. The tolerant metrics can hide a systematic 1-px
    lag, so they never replace the plain error.
- **Media:** a GIF per representative result, with original | replay | diff (red = missing white,
  blue = extra white) at 6× scale, plus contact sheets.
- **Common columns:**
  - rule complexity in the representation's own units;
  - total file bytes (container with rule + first frame + header, where an exporter exists);
  - search CPU/GPU;
  - replay time;
  - autonomous length;
  - whether the procedure applies unchanged to new clips (all four do).
- **Seeds:** when several are used, the best result and the full distribution are both reported.

### 4.3 Per-method pilots, positive controls, budgets, stop rules

| method | positive control (must pass before Bad Apple is interpreted) | Bad Apple runs | budget |
|---|---|---|---|
| C2 Life-like exhaustive (CPU) | clip generated by B3/S23 from a random 24×24 start, 12 frames: the best rule must reproduce it exactly | all 2^18 rules × S ∈ {1, 2} on `ba48_4500_60`, `ba32_3000_5` and the synthetic clips | ≤ 2.5 core-h |
| C3 2nd-order + trees (CPU) | clip generated by a known depth-3 tree over (current, previous) bits, from a random start: recovered tree replays it exactly | depth 1..10 trees, plain and DAgger (3 rounds), on `ba48_4500_60`, `ba32_3000_5` and the synthetic clips | ≤ 1.0 core-h |
| C4 CGP (CPU) | `pcA` (2 gates) and `pcB` (4 gates) known-circuit clips: exact recovery in ≥ 1 of 4 seeds | gate budgets 8, 16, 32; F_priv k=1, S=2; 4 seeds; `ba48_4500_60` and `ba32_3000_5` | ≤ 4.0 core-h |
| C5 DiffLogic long (GPU, gpuq) | `pcA_long` (18 frames, known 2-gate circuit): exact hard rollout in ≥ 1 of 4 seeds | `ba48_4500_60`, full 60-frame rollout from frame 0, 4 seeds, selection by hard rollout | ≤ 1.0 GPU-h (+ ≤ 0.3 core-h CPU) |
| evaluation, media, verification | – | – | ≤ 1.0 core-h |

- **Totals:** ≤ 8.8 CPU core-h (of the 12 ceiling) and ≤ 1.0 GPU-h (of 2). The rest is reserve, used
  only for one small follow-up on a promising method, recorded before it starts.
- **CPU accounting:** `os.wait4` user+sys of each process (threads and children included), in
  `runs/c7/cpu_ledger.jsonl`. Wall time is recorded separately.
- **GPU accounting:** gpuq wall per job. Compile and warm-up are recorded separately in each result.
- **Limits:** 1 thread per CPU job (numpy/BLAS threads = 1), `prlimit --as` 16 GB, a wall cap. The
  code is frozen before each run. Candidates and checkpoints are saved immediately.
- **Positive controls:** if a method's control fails, that is an **implementation failure**. It is
  fixed within the method's budget or the method is stopped. The method itself is not judged.
- **Success**, per method, for this phase:
  - any approximate replay of `ba48_4500_60` **better than hold on mean error and with change-IoU >
    0**, from a rule small enough to state;
  - any exact long replay.
- **Stop rules:**
  - the budget is used up;
  - the control fails twice after a fix;
  - or (C4/C5) no improvement over hold after half the budget.
- **Common task vs diagnostics:** `ba48_4500_60` and `ba32_3000_5` are the common tasks; everything
  else is a method-specific diagnostic.
  - Not every method is forced to an exact solution of the same length.
  - **Comparability caveat:** C2 searches its whole language; C3 is supervised with a known memory;
    C4 and C5 are stochastic, with 4 seeds each. Budgets differ by method; this is stated, not
    normalised.
- **Not in scope:** UI; SAT splitting or small-G proofs; learned hidden init and positional inputs
  (only discussed).

## Sources
- [DiffLogic CA, arXiv 2506.04912](https://arxiv.org/abs/2506.04912) · [Deep DLGN, 2210.08277](https://arxiv.org/abs/2210.08277) · [Conv DLGN, 2411.04732](https://arxiv.org/pdf/2411.04732) · [Mind the Gap, 2506.07500](https://arxiv.org/html/2506.07500v2)
- [Richardson et al., 2310.14809](https://arxiv.org/abs/2310.14809v2) · [DyNCA, 2211.11417](https://arxiv.org/pdf/2211.11417) · [Gilpin, CA as CNNs, 1809.02942](https://arxiv.org/abs/1809.02942) · [Elser, 2012.02179](https://arxiv.org/pdf/2012.02179)
- [Evaluating the Apperception Engine, 2007.05367](https://arxiv.org/pdf/2007.05367) · [pylfit](https://pypi.org/project/pylfit/) · [Causal deconvolution, 1802.09904](https://arxiv.org/abs/1802.09904)
- [Öztürkeri & Johnson 2011 (CGP for CA)](https://gpbib.cs.ucl.ac.uk/gp-html/DBLP_journals_jca_OzturkeriJ11.html) · [Sapin et al. 2003](https://gpbib.cs.ucl.ac.uk/gp-html/sapin03.html) · [Second-order CA](https://en.wikipedia.org/wiki/Second-order_cellular_automaton)
- Code read via the GitHub API: AlexDR1998/NCA, williamgilpin/convoca, Tony-sama/pylfit, xcontcom/evolving-cellular-automata

---
## Follow-up recorded before running (appended 2026-11-09, ledger 3.09 core-h)
**Pilot observation that motivates it:**
- On `ba48_4500_60`, C4 (CGP) circuits with only **3–10 active gates** halve the mean error relative
  to hold (0.089–0.13 vs 0.185).
- They do so by reproducing the coarse silhouette change, not frame-accurate motion: change-IoU is
  0.01–0.04 and the horizon is 1.
- C3 reaches 0.67× hold only with a 1,466-node tree, and C2 0.90× with change-IoU about 0.

**Follow-up question: transfer.** Does the same procedure, unchanged, behave the same way on a
**different** clip?
- **Clip:** `ba48_1800_60`, frames 1800–1859: a figure with thin, fast details. It was the
  second-ranked candidate in the segment scan, and its first frame is non-uniform.
- **Runs:**
  - C2: S=1, all 2^18 rules;
  - C3: depths 1–12, DAgger 3;
  - C4: k1_S2, G=16, 4 seeds × 600 s.
- **Budget:** ≤ 0.8 core-h. No GPU.
- **Not changed:** none of the methods and none of the settings.
