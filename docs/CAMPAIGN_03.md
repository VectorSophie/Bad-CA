# Campaign 03: plan and budget (written before any run)

Date 2026-11-08. Results go to `docs/CHECKPOINT_03.md`.

## Code freezing
Every submitted job runs from `runs/frozen/<sha12>/`. `tools/freeze.sh` makes a read-only copy of `ca/ experiments/ player/ tests/`, named by a content hash. `runs`, `refs` and `cache` are symlinked back to the project. Each result records `code_snapshot`. Editing the working tree after submission cannot change a running or queued job.

## Part A: length scaling (CPU only, never via gpuq)
- **Clips:** Bad Apple, 32×24, threshold 128, start frame 4500 (t = 150.000 s, non-uniform; black opening skipped), lengths 5 / 10 / 20 / 40.
- **Pre-check** (done): the light cone passes at S=2 for every length. The 40-frame clip has a state repeat (frames 28→29), so memoryless rules are infeasible there.
- **Fixed setting:** F_priv k=1, S=2 (condition E), boundary 0, hidden init 0.
- **Method:** min-exception anytime SAT (CaDiCaL), identical for every length. Clauses are streamed into the solver so long clips fit in memory.
- **Budget:** 30 min wall per length (encoding included), 1 CPU thread each, `prlimit` address space 16/24/48/96 GB, outer `timeout -s KILL`. **The same 30 min for every length**, so longer clips get less effective search per frame. Status, time to first solution and best E are reported. Not finding a solution, or not going lower, is reported as UNKNOWN / upper bound and **never as evidence that a large rule is required**.
- **Size accounting:** reproducible bitstream encoders with decode round-trip checks (`ca/costs.py`).
  - **Shared by every method, excluded from both sides and reported separately:** W, H, T, fps, threshold, and the first frame (raw W·H bits).
  - **Rule side:** family id, k, S, boundary, hidden init, default behaviour, exception count and exception indices (Elias-γ gaps). The wiring/taps are fixed by family id and k. A DiffLogic circuit is encoded as its live sub-circuit: ops, wiring and output map.
  - **Recording side:** (i) gap-coded change list, (ii) XOR + alternating run-length coding, (iii) per-frame RLE without temporal prediction, a weak reference. All use the same Elias-γ code.
  - **Exact vs approximate are compared separately.** An approximate rule is compared with the recording cost of *its own output video*, and its error against the target is listed next to it.

## Part B: hard-rule optimization on ba32_3000_5 E only
This is the only clip where the architecture was shown able to represent an exact solution.

| id | track | method |
|---|---|---|
| R0 | video only | relaxation baseline (soft gates, soft states) |
| HG | video only | straight-through gates (hard one-hot forward, softmax backward) |
| HS | video only | straight-through states, all channels (round forward, identity backward) |
| HH | video only | straight-through hidden state only |
| HB | video only | HG + HS |
| D1 | video only | discrete refinement: gate-op hill-climb from R0's best hardened circuit (CPU) |
| D2 | video only | SAT repair: exact rule with minimum Hamming distance to R0's hardened LUT (CPU) |
| T1 | **SAT teacher (diagnostic)** | R0 + loss on the hidden channel against the SAT solution's hidden trajectory |
| T2 | **SAT teacher (diagnostic)** | supervised distillation of the SAT LUT (re-run under frozen code) |

- The teacher track is an optimization diagnostic and is **never counted as end-to-end discovery**.
- Same 4 fixed seeds (wiring seeds 1000–1003) for every GPU method. **No seed scaling.**
- **Budget per method:** 600 s train wall on 1 GPU (GPU methods), 600 s on 1 CPU thread (D1, D2).
- **Checkpoint selection by hard autonomous rollout error.** Soft metrics are recorded separately.
- **Gap decomposition** on each method's selected checkpoint:
  - soft gates + soft states;
  - soft gates + binarized states (all channels / hidden only);
  - hard gates + soft states;
  - hard gates + binary states (= hard rollout);
  - per-frame error curves, to see accumulation;
  - one-step disagreement between the soft and hard net on the hard trajectory's states (local gate-selection gap without accumulation).

## Total budget
| item | estimate |
|---|---|
| GPU pilots (6 methods × 60 s) | 0.15 GPU-h |
| GPU main (6 methods × 600 s + compile/eval) | 1.2 GPU-h |
| CPU length scaling (4 × 30 min) | 2.0 CPU-h |
| CPU D1 + D2 + verification/encoding | 0.5 CPU-h |
| **total** | **≈1.35 GPU-h, ≈2.5 CPU-h** |

The pilot results may lower these numbers but will not raise them without a written note.
