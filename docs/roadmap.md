# Roadmap

Each item starts from a documented failure or gap. No dates or outcomes are promised; each step needs its own plan,
budget and stopping rule, fixed before its runs (as in the CAMPAIGN files).

The final goal stays as stated in `claude_ca_build_prompt.md` and the story handoff: the whole video replayed
autonomously from its start, faithful to the binary original at the chosen resolution, with automatically
optimised phases and human-readable rules, an interface to inspect cells, rules and phases, and full storage
accounting. It has not been reached.

## Open now

1. **Whole-song quality.** The whole-song run (110 phases, no noise) failed as a replay after ~1,080 frames: the
   chained rules amplify float-order differences. Training-only state noise (σ = 0.02) made replay equal training
   for 9 chunks (frames 0–2115). The next step is the whole song with that recipe (~28 chunks, est. ~7 GPU-h per
   seed), plus more phases around fast wipes (1507–1522 is the first failure) and a bound on hidden-state growth.
2. **Phases vs capacity.** Settled for 1760–1999: 4 phases (0.027–0.033) beat a single rule with the same parameter
   count (0.045–0.047). Not shown on cut-free material, where 2 phases gave no clear gain.
3. **Automatic segmentation.** Never better than equal spacing (K = 2, 4, 6). A boundary criterion that predicts
   where a radius-4, hidden-state CA needs a new rule is still missing.

## Next, grounded in failures

- **Transitions and cuts.** The 240-frame single rule failed mostly in the middle third (wipe at 1880, shot change
  near 1911). Greedy training was much worse than joint on those phases, which suggests that hidden state prepared
  across boundaries matters. Measure what the carried hidden state holds just before a cut; test shorter phases
  around transitions.
- **Longer horizons without blow-up.** Rules run past their trained horizon diverged (C11 warm start; C12 greedy
  warm start). A stability term or training slightly past the phase end could address this. It is untested.
- **Smaller models.** 3,884 and 1,304 parameters failed the C11 gates (specks, filled gaps, under-motion at 120
  frames). Distillation was worse than direct training. Low-rank per-phase deltas (`delta`, rank 4) were worse than
  independent phases. Directions: higher rank, sharing across neighbouring phases only, quantisation beyond f16.
- **Explicit rules on real motion.** Decision lists replay Life and a hidden-bit rule exactly, but reach only
  0.0995 on a 56-frame Bad Apple clip (NCA: 0.004), and extraction from the NCA reaches 0.153. Candidates:
  richer integer features (directional counts at radius 2, hidden-neighbour directions), learning against the
  autonomous rollout instead of teacher-forced pairs, more hidden bits.
- **Storage.** Every model file is larger than a lossless recording of its frames. A compression claim needs a model
  that is both smaller than the recording and within the gates. No model so far is both.
- **Higher resolution.** Every result is 48×36. The pipeline accepts widths up to 256. Cost and quality above 48 px
  have not been measured.
- **Internal-state phase switching (type 3).** The cells' own state would select the rule, removing the external
  schedule. Not implemented; a pilot only after type 2 works on long intervals.
- **Other videos.** The pipeline runs on any decodable video. Quality has only been studied on Bad Apple. A small
  set of non-Bad-Apple clips with the same gates would test generality.

## Engineering

- A full lock file (only the direct dependencies are pinned in `requirements.txt`).
- `ca.pipeline` support for phased (`ncab-v2`) training, export and replay. Today this lives in experiment scripts.
- Pre-deployment hardening for the UI (content-based MIME checks, isolated decoding, authentication, request and
  storage limits) before it is ever exposed beyond 127.0.0.1.
