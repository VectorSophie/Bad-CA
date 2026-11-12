# Path D: references and what was actually inspected

Checked on 2026-11-12. "Inspected" means read in this session, not recalled.

| source | availability | license | what was inspected | used here |
|---|---|---|---|---|
| Miotti, Niklasson, Randazzo, Mordvintsev, *Differentiable Logic Cellular Automata: From Game of Life to Pattern Generation*, arXiv:2506.04912 (2025-06-05) | arXiv abstract page; official article (google-research.github.io/self-organising-systems/difflogic-ca/) fetched and read | paper CC BY 4.0 | abstract; the article's method and results text, including circuit sizes: Life 336 active gates, sync checkerboard 22, async checkerboard 577, multicolour G 927 | claims compared in D0 |
| Official implementation: `google-research/self-organising-systems`, `notebooks/diffLogic_CA.ipynb` | local clone `refs/sos` @ 3d5547c; `git fetch` shows no later change to the notebook | Apache-2.0 | all code cells: wiring (`get_unique_connections`, `get_moore_connections`), the 16-gate relaxation, soft/hard decoding, perception (shared wiring, per-kernel logits, per channel), update net, sync/async step, Life and checkerboard training, the robustness tests. The notebook has **no saved outputs**, so the authors' own logs could not be checked. | ported faithfully as `ca/dlca.py` |
| Petersen, Borgelt, Kuehne, Deussen, *Deep Differentiable Logic Gate Networks*, NeurIPS 2022, arXiv:2210.08277 | arXiv; repo `Felix-Petersen/difflogic` cloned to `refs/difflogic` @ 469702c | MIT | `difflogic/difflogic.py` (`LogicLayer`, random/unique connections, `grad_factor` for deep nets), `functional.py` (`bin_op_s`, `get_unique_connections`), `compiled_model.py` (C code export) | the gate relaxation and unique wiring (through the notebook); nothing else |
| Petersen et al., *Convolutional Differentiable Logic Gate Networks*, NeurIPS 2024, arXiv:2411.04732 | arXiv abstract only; **no code for logic-tree convolutions, OR pooling or residual init in `difflogic`** (README and sources checked) | arXiv non-exclusive | abstract: tree convolutions, OR pooling, residual initialisation | not used. The notebook's pass-through init (logit 10 on "a") plays the role of residual init. |
| `rplsmn/logicars` (Rust) | local clone `refs/logicars` @ 1f316a4 (no newer commits) | **none declared** (no LICENSE file, no `license` in Cargo.toml or README; an earlier note in `docs/RESEARCH_PHASE_07.md` said MIT, which is wrong) | README, layout (`src/{gates,perception,update,training,circuit}.rs`, 147 `#[test]`s), `checkerboard_sync_log.csv` | **read-only, no code reused.** Its checkerboard log (3,040-gate model, the same as ours) first reaches hard loss 0 at epoch 511, past the notebook's 500. Its README claims only ">95 %" for small-model Life, not exact. Treated as a third-party reproduction. |
| Lee, Sun, Chung, *Two-Stage Unit Tying for Simplifying Differentiable Logic Gate Networks*, ICML 2026 | ICML poster page `icml.cc/virtual/2026/poster/64145` (links OpenReview `QVt9dgerpS`). **The full text was not accessible:** OpenReview requires a browser challenge for automated access, and no arXiv version or code release was found. | unknown | only the poster listing and search summaries of the abstract: tie units to constants, propagate constants, remove downstream logic; Gauss-Newton screening, then refinement with a few finite-difference evaluations; FPGA area; MNIST/CIFAR-10 | the idea of unit tying only. Our D2 ties a gate to its majority value and scores the tie on behaviour, not on a classification loss. Their screening and refinement steps are **not** reproduced. |
| Khalifa, Nasir, Siper, James, Togelius, *Programmable Cellular Automata*, arXiv:2609.06102 (2026-09-05, submitted to EXAG 2026) | arXiv abstract | arXiv non-exclusive | abstract: local functions + decision function as Python code; optional global functions (some problems need them); recurring functions across experiments | Path E background only. Global functions would break locality and are not used. |

## Faithful parts and differences (D0)
**Identical to the notebook:**
- gate set and relaxation;
- unique wiring algorithm and the first-kernel Moore wiring;
- pass-through init (logit 10 on op 3);
- perception: shared wiring, kernel-specific logits, applied per channel, outputs ordered (channel, output, kernel);
- update input: the centre cell first;
- loss: summed squared error;
- hyper-parameters for Life and the sync checkerboard (lr 0.05, batch 20/2, 3,000/500 epochs, layer widths);
- the PRNG split order at init.

**Different:**
- **Optimizer:** `optax` is not installed and was not added (the JAX environment is left as it is), so `optax.chain(clip(100), adamw(lr, 0.9, 0.99, 1e-2))` is re-implemented in `ca.difflogic.adamw`, with the same formula.
- **Seeds:** a different JAX version may permute wiring differently for the same seed, so "seed 23" is not guaranteed to be the authors' circuit.
- **Logging:** the hard loss is logged after the update, not before. This is cosmetic.
- **Batching:** batches use `jax.vmap` / broadcasting instead of the notebook's patch extraction (`conv_general_dilated_patches`). This is checked: the NumPy executor on explicit patches equals the JAX hard rollout.
- **Not reproduced:** asynchronous training, the lizard and the multicolour G.

## What does not carry over to recurrent CA
- **Classification read-out:** DLGN classification uses `GroupSum` read-out and static accuracy. A CA is iterated, so a near-miss local function compounds. Hence every result here is scored on hard rollouts.
- **`grad_factor`** (difflogic, for depth > 6) is not in the notebook and was not used.
- **Static don't-cares:** static-accuracy pruning, as in unit tying, has no notion of states that the dynamics never visit. In a CA those states make up the real don't-care set (D2).
