# Plan

## Scope
- **Goal:** explicit, readable, executable local rules for Bad Apple motion (see the README for the target).
- **Teacher:** pinned to the whole-song Neural CA `runs/featured/whole_song_s0.ncab`, sha256
  `facb43d9911399eeffa0e04df010a82f713d54b2212d5d2e2e4cf7c72358e4f5`. Every script asserts this hash and never swaps
  the file.
- **What the teacher is used for:** only path B, and only as training data. Replay never calls it.

## Three paths, sharing one executor and one evaluator
| path | idea |
|---|---|
| A, direct synthesis | Search rules directly against the video. There are two starts: the empty program ("blind"), or a decision list fitted to one-step frame pairs ("tree"). No NCA information is used. |
| B, NCA extraction | Abstract the teacher's 16-channel state into v plus k bits (PCA signs or k-means), then fit a decision list to the abstract transitions. |
| C, simple-rule extension | Start from classic rules (hold, Life B3/S23, majority, grow, erode, four shifts) and search for edits. |

- **Phases:** all three can run phased (K programs on a uniform or automatic schedule, state carried, each program stored
  as edits of the previous one). K = 1 is always included.

**Evaluation (`experiments/rr_run.py:evaluate`):**
- Every score is computed by replaying the exported file from its first frame. There is no target injection.
- **Measures:**
  - pixel error;
  - lag-1 and lag-4 change F1 and change mass;
  - worst 16-frame window;
  - thin parts, specks and holes, spurious one-frame flips;
  - complexity and bytes.
- **Baselines:** hold, majority, flicker and the four shifts, all run autonomously the same way.

## Mid-course review (2026-11-11)
This re-scoped the track without starting a new campaign. The changes:
1. **Execution, recovery and synthesis are reported separately.**
   - Replaying a library rule exactly counts as executor validation, not discovery.
   - Recovery means the generator rule was hidden from every method; it is tested from held-out initial states.
2. **Expressiveness vs search failure:**
   - Use SAT or a constructive proof to decide whether a behaviour is expressible.
   - Otherwise write "unconfirmed".
3. **Memory diagnostics:** collisions of identical local inputs with different successors, for k = 0–3 explicit memory
   bits, kept apart from collisions caused only by missing hidden state, substeps or radius.
4. **Fixed-rule CA is the main comparison.**
   - Shared-rule and independent phased CAs are reported with their schedule bytes.
   - They are never called a Conway-style success.
   - A state-driven CA (no external schedule) is a long-term candidate.
5. **Locality limits:** light-cone proofs and heuristics per task.
6. **Small Bad Apple tasks first:**
   - translate (1221:60), contour (4500:56), wipe (1499:32), cut (1678:32);
   - 16-frame clips (1221, 1241, 4500).
7. **Prune paths with weak prospects;** never add seeds under conditions that already failed.

**Order of the tasks:**
1. Positive controls: Life, shift, blink (memory), excitable (memory and refractory).
2. 16-frame clips.
3. 32–60-frame tasks with motion, wipe and cut.
4. Longer tasks only after a fixed rule generalises.

## Budget (separate from Campaign 13)
| | cap | pilot cap | used so far |
|---|---|---|---|
| GPU | 24 GPU-h | 3 GPU-h | **0** (everything ran on CPU) |
| CPU | 64 core-h | 8 core-h | **1.92 core-h** (`runs/readable_rules/cpu_ledger.jsonl`, per-process CPU time) |
