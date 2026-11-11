# Decisions

**D1. Library replays are executor checks, not results** (2026-11-11).
- Exact replays of Life or shift started from the library contained the answer.
- Discovery claims need the generator hidden and held-out initial states (FINDINGS §2).

**D2. Path B (quantised NCA state) is stopped in its current form** (2026-11-11).
- **Why:**
  - 1–3-bit abstractions of the teacher's state leave 66–84 % of samples in ambiguous inputs.
  - The extracted rules replay as hold (0.183 on 4500:56). Balancing toward changes made them worse (0.199 / 0.450).
  - One-step agreement of 0.99 was a "keep" artefact.
- **What could reopen it:** an abstraction chosen for predicting changes, not for variance. No more seeds or k values
  under the same setup.

**D3. Phased explicit rules are a reference, not the target** (2026-11-11).
- **Result:** K = 8 schedules roughly halve the error and are the only explicit models that move on 32–60 frames.
- **Cost:** they need an external timeline (≈ 270 schedule bytes) and 5–7× the conditions.
- **Labelling:** they are reported with that cost and never called a fixed-rule or Conway-style CA.

**D4. Reach per frame must be at least 4 cells for exact replay of these tasks** (proof, FINDINGS §5).
- Runs at S = 1 with `r2` (reach 2) are provably inexact from a zero hidden start on translate, contour and wipe.
- New Bad Apple runs use S ≥ 2.

**D5. Rollout search alone does not find memory rules.**
- On blink and excitable it failed where exact SAT succeeded. On Life it failed where a one-step fit succeeded.
- **Use instead:**
  - a constraint solver (SAT) wherever hidden bits are needed;
  - a one-step fit where the state is fully visible;
  - rollout search only to polish and select.

**D6. Next directions** (1–2, in order).
1. **One fixed rule fitted jointly on several disjoint short clips, selected on held-out clips.**
   - The 16-frame blind rules move, but they memorise their clip (FINDINGS §3a).
   - Fitting several clips at once is the direct test of whether a small local motion law exists for Bad Apple at all.
   - Settings: reach ≥ 4 (D4), k = 0 and 1, CPU only.
2. **SAT or MaxSAT synthesis with 1–2 hidden bits and S = 2 on short clips, then compressed to a decision list.**
   - **Why:** it is the only method here that found memory rules blind (FINDINGS §2), and it gives "unsatisfiable"
     proofs instead of "not found".
   - **Known limit:** exact SAT on Bad Apple timed out beyond about 10 frames in earlier work (C3). Hence the MaxSAT
     variant (minimum wrong pixels) and short clips.

A state-driven CA (mode switches from cell state instead of a timeline) stays a long-term candidate. It is not started.
