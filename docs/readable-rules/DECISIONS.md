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
- **Update (2026-11-12, user decision):** a phased readable-rule export that replays the whole song autonomously
  becomes the featured demo, with the Neural CA moved to a comparison view. Labelled "K rules + schedule" with the
  schedule bytes and its error next to the Neural CA's. The quality bar for that switch is fixed before any such run.

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

**D7. Second round (2026-11-12): what failed, and where to look next** (FINDINGS §8).
- **Rollout search:** its objective is deceptive (near-miss rules score like wrong ones, §8a). Use it only to polish;
  fit with one-step or SAT objectives first (confirms D5).
- **Exact SAT from a zero hidden start:** closed for Bad Apple at radius 1 per substep. Every crop and the full frame
  is UNSAT at 2–6 frames, no timeouts (§8b), as the light cone predicts (§8c). Not repeated with more seeds, k or S.
- **Longer training windows:** one fixed rule loses most of its motion beyond 32 frames (§8d). A mixed short/long
  objective did not help. Not pursued further.
- **Next, in order (CPU only, inside the remaining budget):**
  1. **Free hidden start in SAT, counted as seed bytes.** Every UNSAT so far depends on the hidden bits starting at
     zero; the Neural CA succeeds with a learned start. Letting the solver choose the hidden start (k bits per cell)
     tests that premise directly. Its cost is visible: k × 1,728 bits for the full frame.
  2. **Phased readable rules toward the whole song** (D3 update), one short rule per phase with phase boundaries at
     cuts and wipes, fitted with one-step/SAT objectives and polished by rollout.
- **Remaining hypotheses (open, not tested):**
  - H1: exact short replay needs a non-constant hidden start (tests with direction 1).
  - H2: with R ≥ 4 (radius 2, S = 2) a fixed rule exists for translate/contour that the searches missed; the light cone
    no longer rules it out, and no SAT encoding for radius 2 exists yet.
  - H3: Bad Apple's motion is locally lawful only within scenes, so a per-scene rule list is the smallest honest
    readable model.

