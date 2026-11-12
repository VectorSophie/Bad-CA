# Readable Rules

**Goal:** reproduce video motion with a few human-readable local transition rules, in the spirit of Conway's Game of Life:
- every cell sees only itself and a small neighbourhood;
- every cell follows the same law;
- the state is small and discrete;
- motion emerges from repeated local updates, not from injected frames.

The whole-song Neural CA stays a separate, successful result. Here it is used as a quality reference and, optionally, as a
teacher. It is not reinterpreted as a failure.

| page | content |
|---|---|
| [PLAN.md](PLAN.md) | scope, paths, task ladder, budget, the mid-course review that re-scoped the work |
| [RULE_LANGUAGE.md](RULE_LANGUAGE.md) | the rule format (`dlrule-v2`), its exact semantics and how complexity is counted |
| [FINDINGS.md](FINDINGS.md) | results: execution, recovery and synthesis kept apart; locality limits; costs |
| [DECISIONS.md](DECISIONS.md) | what was stopped, what continues, and why |

## Status
**State as of 2026-11-12, round 4 (generalisation): NO-GO.** Rules fitted on one 16-frame window do not transfer to
the next window of the same scene. The best patch table of radius 1–4 doesn't either, even one step ahead. Large-scale
search is stopped and Readable Rules stays an experimental Rule Lab feature (FINDINGS §10, DECISIONS D9). Rounds 2–3:
FINDINGS §8–9.

The pilot status below is kept as recorded.

**State as of 2026-11-11: the pilot and the mid-course review are complete.**

- **Execution** (the right rule given): exact in Python, Node and the browser, on 4 positive controls and on random programs.
- **Blind recovery** (the generator rule hidden):
  - The movement and Life controls were recovered exactly, including from unseen initial states.
  - The two memory controls were recovered by exact SAT plus a decision list. They are exact on the training run and
    near-exact on unseen starts. Plain rollout search failed on them.
- **Synthesis on Bad Apple:**
  - On 16-frame clips, a single fixed rule of 3–8 conditions, found by search from the empty program, beats every
    autonomous baseline and produces real motion (lag-4 change F1 0.5–0.77).
  - These rules are clip-specific: on later clips they never saw, they mostly do no better than holding the first frame.
  - On 32–60-frame tasks, only scheduled (phased) rule sets move. Those use an external timeline and are not reported as
    Conway-style results.

**Code:**
- `ca/rules.py` and `player/rule_core.mjs` (engines);
- `ca/rr_search.py` (search);
- `experiments/rr_run.py`, `rr_extract.py`, `rr_diag.py`, `rr_short_check.py`.

**Results:** `runs/readable_rules/` (`diag/*.json`, `pilot/*/result.json`, `cpu_ledger.jsonl`).

**UI:** the rule lab, group "Bad Apple · explicit rules", shows "one rule found blind frames 4500-4515" and the others.
