# Checkpoint 10: cleaner autonomous replay at the same size (10,320-parameter Neural CA)

**Plan, diagnosis and frozen settings:** `docs/CAMPAIGN_10.md`.
**Results table:** `runs/c10/summary10.md`. Machine-readable versions:
- `runs/c10/eval/c10_main10_{main,confirm}_best.json`, with every metric per seed, the medians, the
  pre-registered criteria and the Node checks;
- `runs/c10/pre_diag.json`, the pre-training diagnosis.

**How results are measured:**
- **Full span** = the whole autonomous replay, main src 4500–4555 (56 frames) and confirm 1800–1859 (60 frames).
- **Window** = 4540–4555 / 1844–1859, cut from that same rollout.
- **Medians:** all numbers are medians over seeds 0–2 unless a seed is named.
- **Replays:** all replays start from the first frame and hidden 0, with no reset or injection, and are made from
  the **exports**.

## Videos to open first
1. `runs/c10/media/c10_main_4_ZOOM_FULL_4500-4555_rows4-31_cols0-27_orig_c9_BC_diffs_rep_s2_0.25x.mp4`
   - Arm/shoulder ×12: original / C9 / B+C / both diffs, representative seed.
2. `runs/c10/media/c10_main_1_FULL_4500-4555_orig_c9_A_B_C_rep_s2_1x.mp4`
   - Full span: original / C9 / A / B / C / B+C, same seed.
3. `runs/c10/media/c10_confirm_2_FULL_1800-1859_orig_c9_BC_diffs_rep_s0_0.25x.mp4`
   - Confirm task: original / C9 / B+C / diffs.

**Other video sets:**
- **Representative / best / worst seeds:** `c10_{task}_1_*` and `c10_{task}_2_*`.
- **Continuous vs binary:** `c10_{task}_3_*`.
- **All-arm zoom:** `c10_{task}_4b_*`.
- **Pre-training:** `pre10_*`. `FULL_` is the whole span; `WINDOW_` is cut from the full rollout; `ZOOM_FULL_`.

**Formats and checks:**
- Every video exists as 1x and 0.25x MP4 plus GIFs.
- 26 videos (52 MP4 + 52 GIF) were decoded back: 18 Campaign 10 comparison videos (`video_index10.json`) and 8
  pre-training videos. Frame count = 56 / 60, fps 30 / 7.5, all OK.

## 1. Full replay check (before training)
- **Exports match C9:** the 6 C9 exports (2 tasks × 3 seeds), replayed from their first frame alone, equal the
  C9 evaluation.
  - Full rollout identical.
  - Window identical.
  - Continuous difference 0.0.
- **Training span:** C9 trained on and replayed the **whole** span. Its videos showed only the last 16 frames.

## 2. What the "noise" was (observed facts, C9 models, `pre_diag.json`)
- **Contour errors:** 91–98 % of wrong pixels lie within the 2-px band of the original's contour.
  - Interior and background errors are negligible (≤ 0.1 % of pixels).
  - These errors are the holes and bumps seen in the videos:
    - 1–2 px bites into the arm (FN);
    - bumps and narrowing of the true arm–body gap (FP);
    - on main, small true holes filled in: 2–6 small holes vs 19 in the original.
- **Errors arise during the action:**
  - Error triples in the main clip's last third (0.017–0.039 vs ≤ 0.011).
  - Fewer than 6 % of window errors began before the window.
  - Resetting the visible state to the truth at the window start does not reduce them.
- **Under-motion, not delay:**
  - The replay changes 27–54 % fewer pixels per frame than the original.
  - The best post-hoc time alignment is d = 0 for every seed.
  - The perceived "delay" is smaller and late changes along the contour, not a time offset.
- **Not a threshold problem:**
  - The continuous output is a raw state value regressed toward 0/1. It is not a probability or logit; range −0.33 to 1.62.
  - About ⅓ of wrong pixels lie within 0.1 of the threshold and ⅓ are wrong by ≥ 0.25.
  - No global threshold from 0.3 to 0.7 beats 0.5.

## 3. Arms (identical model, start checkpoint, lr 2e-4, fresh Adam, 900 s GPU)

**Comparison set-up:**
- A, B and C ran in one vmapped job per task, so they had **identical steps**: 10,688 on main, 10,297 on confirm.
- B+C ran as a follow-up job with ~2× the steps (see the caveat below).

**Full-span medians** (err / contour-band err / lag-1 change F1 / lag-4 change F1):

| arm | main | confirm |
|---|---|---|
| C9 (start) | 0.0129 / 0.0121 / 0.726 / 0.910 | 0.0266 / 0.0261 / 0.580 / 0.870 |
| A: more training | 0.0068 / 0.0064 / 0.861 / 0.949 | 0.0174 / 0.0170 / 0.687 / 0.901 |
| B: + contour hinge (λB = 2) | **0.0048 / 0.0041 / 0.900 / 0.969** | **0.0143 / 0.0135 / 0.749 / 0.923** |
| C: + change matching (λC = 0.05) | 0.0071 / 0.0067 / 0.856 / 0.946 | 0.0172 / 0.0168 / 0.693 / 0.905 |
| B+C (~2× steps) | 0.0040 / 0.0035 / 0.911 / 0.976 | 0.0134 / 0.0127 / 0.771 / 0.928 |

**Window error** (main / confirm):

| arm | window error |
|---|---|
| C9 | 0.0247 / 0.0313 |
| A | 0.0106 / 0.0128 |
| B | 0.0075 / 0.0099 |
| B+C | 0.0067 / 0.0091 |

**Criteria:**
- Every arm meets all four pre-registered criteria on both tasks:
  - shape;
  - motion: the C9 stage-C rule passes 3/3 for every arm;
  - no local trade: every third improves;
  - no new flicker.
- Representative candidate by the rule (lowest band error): **B+C**.

**Facts:**
- **Extra training alone (A)** removes about half of C9's error.
  - Main 0.0129 → 0.0068, confirm 0.0266 → 0.0174.
  - It also raises change F1.
  - At the original lr 2e-3 it did the opposite (pilot 1): restarting Adam knocked every model off its optimum.
    The gain needs a smaller continuation lr.
- **The contour term (B) adds on top of A at equal steps.**
  - Band error is lower than A in 3/3 seeds on both tasks: medians −36 % main, −21 % confirm.
  - Lag-1 F1 rises (+0.04 / +0.06), and so does change mass: 0.79 vs 0.71 main, 0.62 vs 0.58 confirm.
  - Main small holes partly come back: 10–16 vs 5–23 for A, 19 true.
  - **What it made worse:** on main, B creates small specks the original does not have.
    - s1: 20, s2: 11, against 0 in the original and 0–5 for C9 and A.
    - Spurious blips rise on main s2: 19 vs 11 for C9.
- **The change term (C) has no measurable effect beyond A.**
  - Medians are within noise of A, with mixed sign per seed.
  - It did not cause stalling or flicker; blips are 7 / 34, like A.
- **B+C vs B:** at matched step counts, from the training logs, B+C ≈ B on all 6 seed–task pairs (within 0.0003).
  - B+C's lower final error comes from its ~2× steps (21k vs 10.5k), not from the change term. This is a design
    flaw of the follow-up (wall time was matched, steps were not).
  - B+C also has the most specks on main s1 (33).

**Inference (not proven):**
- L2 averaged over all pixels spends most of its weight on easy interior and background pixels. The hinge
  re-weights the 14 % / 28 % of pixels where the errors are, which is why B helps.
- The remaining confirm error is mostly FN (thin staff and hand parts missing; FN 0.009 vs FP 0.004 for B+C) and
  under-motion (mass 0.63). It is unchanged in kind, only smaller.

## 4. Did we get a cleaner replay at the same size?
**Yes, by every pre-registered metric, on both tasks and in every seed.**
- Same 10,320 parameters and architecture.
- Representative B+C vs C9, full-span error:
  - main 0.0129 → 0.0040 (−69 %);
  - confirm 0.0266 → 0.0134 (−50 %).
- In the main zoom at 4552:
  - the true arm–body gap is now kept;
  - a 3–4 px hole inside the elbow remains.
- Not solved:
  - the confirm task still under-moves (lag-1 change mass 0.63);
  - confirm still loses thin parts;
  - single specks appear on main.

## 5. Teacher candidates for the later reduction
- **Main:** `runs/c10/exports/c10_combo10_main_best_BC_s0.nca.json`, full-span error 0.0016, window 0.0024.
- **Confirm:** `runs/c10/exports/c10_combo10_confirm_best_BC_s1.nca.json`, error 0.0130.
- **Equal-step alternatives:** the B models (`c10_main10_*_best_B_s*`).
- **Caveat carried from C9:** these models are still trained on one trajectory each. Campaign 10 did not test
  stage A/B generality.

## 6. Verification
- **Implementations agree (numpy vs Node):** 30 exports checked (C9 ×6, A/B/C ×18, B+C ×6).
  - Binary mismatch is 0 px in 29 of them.
  - B+C main s1 has **1 px**: frame 51 (19, 19), where v = 0.5000007 in numpy and 0.4999923 in Node. That is
    float accumulation order at the threshold. The binary output is never fed back, so it cannot spread.
  - Max |Δv| ≤ 2.5e-4.
- **Agreement is not accuracy:** this agreement is separate from how closely each replay matches the original
  (the error columns above).
- **No clip access:** every replay reads only the export (first frame + weights).
- **Checkpoint choice does not matter:** `last_*` checkpoints give the same picture as `best_*` (`eval/*_last.json`).

## 7. Cost

**GPU:** 70.5 min of 120, 6 gpuq jobs, no failures.

| job(s) | what | time |
|---|---|---|
| 72 | pilot 1 | 5.1 min |
| 73 | pilot 2 | 5.1 min |
| 74–75 | main | 15.1 min each |
| 76–77 | B+C | 15.1 min each |

**CPU:** 0.37 core-h of 8 (`runs/c10/cpu_ledger.jsonl`).
- Includes a conservative 600 s estimate for one unmetered CPU smoke test.
- A few small interactive analysis one-liners were not metered; they are under a minute each.

## 8. Reproduce

```sh
export CA_LEDGER=runs/c10/cpu_ledger.jsonl
python experiments/exp36_full10.py                 # full replay check, diagnosis, pre10_* videos
D=$(experiments/freeze.sh)                          # read-only code snapshot
gpuq -L c10-main-main -G 1 python $D/experiments/exp37_ft10.py main 900 c10/main10_main runs/c10/cfg/main.json --lr 2e-4    # same for confirm
gpuq -L c10-combo-main -G 1 python $D/experiments/exp37_ft10.py main 900 c10/combo10_main runs/c10/cfg/combo.json --lr 2e-4
python experiments/exp38_eval10.py c10/main10_main,c10/combo10_main main --node   # exports, metrics, criteria, Node check
python experiments/exp39_videos10.py                # comparison videos + decode-back checks
```

- **Pilots:** `exp37_ft10.py main 300 c10/pilot{,2}_main runs/c10/cfg/pilot.json [--lr 2e-4]`.
- **Not committed:** `runs/c10/replays/` is regenerated by exp38.
