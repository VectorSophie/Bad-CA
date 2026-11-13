# Research record index

> Start with [How we got here](journey.md) (the decisions, in order) or the [timeline](timeline.md); this page lists every original record.

> **Founding brief** (the original goal and claim boundaries): [brief/](brief/README.md).

> **Readable Rules track** (explicit local rules, after C13): [docs/readable-rules/](../readable-rules/README.md).

> **Path D: Differentiable Logic CA** (learned Boolean circuits, then simplified; experimental): [docs/path-d/](../path-d/README.md).

> **Later experiments** (after the full-song model, summarised in [How we got here §7](journey.md#7-after-the-full-song-smaller-rules-other-videos-other-mechanisms)):
> [context codebook](../context-s0/README.md) (shared local rules + a mode switch),
> [velocity-carrying CA and difficulty atlas](../path-v/README.md),
> [another film, data and phase ablations](../generalization/README.md),
> [shape-aware CA](../shape/README.md).

Every campaign has a plan (`CAMPAIGN_NN.md`, written before its runs; later decisions are appended, never rewritten)
and a result (`CHECKPOINT_NN.md`). Machine-readable results live under `runs/`. The records are kept unchanged in
`docs/`; this page only indexes them. One line per file. Where a line gives a result, it comes from that file.

## Phase 1: exact Boolean rules (Campaigns 01–06)

| file | one line |
|---|---|
| [CHECKPOINT_01](../CHECKPOINT_01.md) | Exact lookup-table baselines, independent replay, data validation of the Bad Apple source (6,572 frames, 182 uniform); a uniform frame cannot evolve detail under a homogeneous rule. |
| [CHECKPOINT_02](../CHECKPOINT_02.md) | Corrections to 01 with exact definitions; state-representation ablation with SAT and DiffLogic on short clips. |
| [CAMPAIGN_03](../CAMPAIGN_03.md) | Plan: length scaling of exact SAT rules and hard-rule optimisation; frozen code snapshots and size accounting introduced. |
| [CHECKPOINT_03](../CHECKPOINT_03.md) | Exact rules found up to 10 frames, 20/40 frames timed out; exact rules larger than every recording baseline. |
| [CAMPAIGN_04](../CAMPAIGN_04.md) | Plan: SAT-centred campaign (incremental vs cold SAT, repair from learned starts, shrinking rules into circuits). |
| [CHECKPOINT_04](../CHECKPOINT_04.md) | Incremental SAT not faster than cold start at equal budget; a learned start did not shorten time to exact; compression still not shown. |
| [CAMPAIGN_05](../CAMPAIGN_05.md) | Plan: direct SAT synthesis of small circuits, compared by total decodable file size. |
| [CHECKPOINT_05](../CHECKPOINT_05.md) | Small circuits win only on synthetic positive controls; no compression on Bad Apple; break-even gate counts mostly unknown (timeouts). |
| [CAMPAIGN_06](../CAMPAIGN_06.md) | Plan: better search for a compressing circuit on the 5-frame clip `ba32_3000_5`. |
| [CHECKPOINT_06](../CHECKPOINT_06.md) | No compressing exact circuit found (UNKNOWN, not impossible); ≤ 4 gates proved impossible in that family. |

## Phase 2: approximate rules and motion (Phase 07, Campaigns 08–09)

| file | one line |
|---|---|
| [RESEARCH_PHASE_07](../RESEARCH_PHASE_07.md) | Survey and table of what was established by 01–06; four method pilots planned (Life-like, second-order trees, CGP, DiffLogic long); readable rules and long clips named as untested. |
| [CHECKPOINT_07](../CHECKPOINT_07.md) | CGP best of the four for simple rules; 3–10-gate circuits reach about 0.5–0.7× hold error on 60 frames but follow coarse shape, not motion. |
| [CAMPAIGN_08](../CAMPAIGN_08.md) | Plan: motion-aware CGP objective vs extended structure; loophole baselines fixed before search. |
| [CHECKPOINT_08](../CHECKPOINT_08.md) | 0 of 32 runs followed the motion to the end; late "motion" scores were edge flicker, a metric loophole. |
| [CAMPAIGN_09](../CAMPAIGN_09.md) | Plan: A/B/C diagnosis (teacher-forced, autonomous) of short real actions, CGP vs Neural CA. |
| [CHECKPOINT_09](../CHECKPOINT_09.md) | The 10,320-param Neural CA passes stage C on both tasks in 3/3 seeds; CGP passes no stage; the NCA is not a small rule and not compression. |

## Phase 3: Neural CA replay, pipeline, phases (Campaigns 10–12)

| file | one line |
|---|---|
| [CAMPAIGN_10](../CAMPAIGN_10.md) | Plan: full-span diagnosis and same-size fine-tuning (more training, contour hinge, change matching). |
| [CHECKPOINT_10](../CHECKPOINT_10.md) | Errors were contour bites and under-motion, not delay; same-size model reaches 0.0040 (56 frames) and 0.0134 (60 frames). |
| [CAMPAIGN_11](../CAMPAIGN_11.md) | Plan: smaller models, longer playback, ncab export, video→CA pipeline, local UI. |
| [CHECKPOINT_11](../CHECKPOINT_11.md) | 1,304 and 3,884 params fail the gates; 120 frames pass at 10,320 params (≈ 0.031); 240 frames fail; every file is larger than a lossless recording; pipeline and UI built. |
| [CAMPAIGN_12](../CAMPAIGN_12.md) | Plan, pilot results and allocation: phased NCAs on an external schedule, explicit dlrule-v1, rule-inspection UI, whole-song run. |
| CHECKPOINT_12 | Phases with carried state, automatic vs uniform segmentation, explicit dlrule-v1 rules, the whole-song attempt and the noise repair (longest passing replay 0–1467), UI inspector, docs. |

### Campaign 12 survey (`CAMPAIGN_12.md` §2)

The sources cited there, and the design choice each one led to:

| source | used for |
|---|---|
| Time-varying / non-uniform CA (Sipper 1997); switching dynamical systems (Fox et al. 2009; Linderman et al. 2017, rSLDS) | external phase schedule (type 2); state-dependent switching (type 3) not pursued |
| Change-point detection and MDL: PELT (Killick et al. 2012), Bayesian online CPD (Adams & MacKay 2007), MDL (Rissanen 1978) | `ca/segment.py`: exact DP over K segments under a per-segment 3×3 transition-table code length |
| Growing / texture / dynamic-texture NCAs (Mordvintsev et al. 2020; Niklasson et al. 2021; Pajouheshgar et al. 2023, DyNCA) | the Neural CA update and training practice used since C9 |
| LoRA (Hu et al. 2021) | `mode=delta`: shared base + per-phase rank-r deltas |
| Rule extraction (Craven & Shavlik 1996, TREPAN); symbolic regression and program synthesis (Cranmer 2023, PySR; Ellis et al. 2021, DreamCoder) | dlrule-v1 decision lists learned by a small CART; symbolic regression and program synthesis not used yet |

No external code was reused in Campaign 12.

## Where this documentation's layout comes from

The README and doc split follow the structure of the user's own repository VectorSophie/hgit (read-only reference;
no text was copied). Borrowed elements:
- **README order:** one-paragraph description → status callout stating exactly what exists today → "Try it" with
  real commands and their output → "What it actually does" → differences and limits → "Experimental" → data and
  licence notes.
- **Status callout** as a blockquote near the top.
- **Doc split** into separate architecture, status/results and format documents (`docs/architecture.md`,
  `docs/format.md`, `docs/evaluation.md`, `docs/rule-models.md`; a `docs/roadmap.md` existed until the 2026-11-13
  cleanup and is in the Git history).
- **Research index:** this page, modelled on hgit's `docs/research/` index.

Not borrowed: badges, release and install sections. This project has no CI, releases or packages.
