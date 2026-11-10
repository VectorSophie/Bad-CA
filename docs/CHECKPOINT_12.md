# Checkpoint 12: phases, explicit rules, rule-inspection UI, documentation

**Plan, gates and every decision made during the runs (with timestamps of intent):** `docs/CAMPAIGN_12.md`.
**Machine-readable results:** `runs/c12/results12.json` (table form: `runs/c12/summary12.md`); per-run details in
`runs/c12/eval/*.json`.

**How results are measured:**
- Every number is an **autonomous replay of an export file**: the first frame (+ stored initial hidden where stated),
  then the exported rules under the exported schedule, with **state carried across phase boundaries**. There is no
  reset or injection.
- Scored against the preprocessed binary original (48×36, gray ≥ 128).
- Preprocessing loss is shown separately (`c12_preprocessing_loss_*`).

## 1. What to open / run
- **UI:** `.venv/bin/python ui/server.py --port 8766` → http://127.0.0.1:8766/ (local only, Ctrl-C to stop).
  - **New:** phase timelines (A/B compare), `+1 substep` / `+1 frame`, a rule view, a cell inspector (NCA:
    neighbourhood, pre/post-ReLU features, Δs, next state; rules: features, every condition ✓/✗, the firing rule),
    rule editing with replay from the start, and a `runs/c12` shelf.
- **Videos** (`runs/c12/media/`; 1x and 0.25x MP4; the full-song files are not committed, regenerate with the command
  in §9):
  1. `c12_L240_FULL_1760-1999_orig_fixed_uniform4_auto4_1x.mp4`: fixed rule vs 4 phases on the interval C11 failed.
  2. `c12_SONG_PREFIX_0-1467_orig_nonoise_vs_noise_trained_1x.mp4`: the longest gate-passing replay (48.9 s,
     including the black opening), next to the same model trained without noise.
  3. `c12_SONG_FULL_0-6571_orig_phasedNCA_diff_1x.mp4`: the whole song with 110 phases. **Fails** after ~1,080 frames.
  4. `c12_main_FULL_4500-4555_orig_explicit_rules_vs_C10_1x.mp4`: executable explicit rules vs the C10 NCA.
  5. `c12_ui_screen_recording_cell_inspector_phase_switch.mp4`: a real UI session with the cell inspector, a phase
     switch, and an explicit-rule cell explanation.
- **Also:** `c12_L240_ZOOM_cane_*` (thin cane, zoom ×12), `c12_SONG_PREFIX_0-1079_*`, `c12_preprocessing_loss_1760-1999_*`.

## 2. What phases improved (L240 = 1760–1999: a wipe at 1880, a shot change ~1911)
Joint training: same steps (12k), lr 5e-4 (the wide fixed rule needed 1.25e-4), 2 seeds. Full-interval error s0 / s1:

| arm | params / seed | error | worst 16-frame window | near cuts/wipes | thin parts / gaps kept | gates |
|---|---|---|---|---|---|---|
| fixed rule | 10,320 | 0.072 / 0.065 | 0.22 / 0.20 | 0.137 / 0.134 | 0.49–0.55 / 0.60–0.61 | fail |
| fixed rule, capacity-matched (Fh 256) | 41,232 | 0.047 / 0.045 | 0.12 / 0.12 | 0.092 / 0.091 | 0.61–0.64 / 0.63–0.64 | fail |
| 4 uniform phases | 4 × 10,320 | 0.027 / 0.028 | 0.069 / 0.067 | 0.045 / 0.050 | 0.74–0.75 / 0.76–0.77 | pass / pass |
| 4 automatic phases | 4 × 10,320 | 0.028 / 0.033 | 0.069 / 0.089 | 0.045 / 0.049 | 0.71–0.75 / 0.75–0.77 | pass / fail (motion) |
| 4 auto phases, shared base + rank-4 deltas | 10,320 + 4 × 1,232 | 0.049 / 0.054 | 0.11 / 0.14 | 0.095 / 0.107 | 0.59–0.61 / 0.63–0.64 | fail |
| 6 uniform phases | 6 × 10,320 | 0.019 / 0.021 | 0.054 / 0.061 | 0.033 / 0.036 | 0.79–0.81 / 0.83 | pass |
| 6 automatic phases | 6 × 10,320 | 0.021 / 0.022 | 0.062 / 0.067 | 0.030 / 0.029 | 0.78–0.79 / 0.83 | pass |

**Facts:**
- Per-phase rules with carried state fix most of what one rule could not do on this interval.
  - Error drops ~2.5× vs one rule, and ~1.7× vs a fixed rule with the same parameter count.
  - Errors around cuts and wipes drop from 0.13 to 0.03–0.05.
  - Thin parts are kept better.
- More, shorter phases help further (6 > 4).
- **Shared base + low-rank deltas:** with the deltas costing 12 % of a full rule per phase, it gets about half of the
  gain.
- **On the cut-free L120 (1760–1879), phases bring no clear gain:** 2 phases give 0.029–0.031, vs 0.030 / 0.032 for C11's
  fixed rule. The benefit is tied to transitions, not to length.
- **Costs that come with phases:** spurious one-frame flicker rises from ~10 (fixed) to 83–140 (phases), and there are
  more specks and holes than in the original. All of it is reported, none of it is hidden by the gates.

**Caveats:**
- The capacity control needed a 4× lower lr and was still improving at 12k steps.
- Two seeds per arm on one interval: a statement about this interval and these settings only.

## 3. Automatic vs uniform segmentation
**Not better.** At K = 4 (L240), K = 6 (L240) and K = 2 (L120), automatic stage-1 MDL boundaries never beat equal
spacing by the pre-registered rule (lower in both seeds and ≤ 10 % worse worst window). Uniform was equal or slightly
better in most cases.

- **Song scale:** the same chunked recipe with uniform phases is reported in §6.
- **Stage 2** (residual-driven boundary moves) was reduced to the K = 4 vs K = 6 comparison. The finding was "more
  phases", not "better boundaries".
- **Why the boundaries do not help:** the MDL proxy (3×3 → centre statistics) does not predict where a radius-4,
  hidden-state CA needs a new rule.

## 4. Executable explicit rules (dlrule-v1)
- **Engine:** readable decision lists over integer neighbourhood features (own state, `n8`, `r2`, `N/S/E/W`, hidden
  bits). First firing rule wins, the default is keep, synchronous update, external program schedule.
- **Exactness:** Python and Node match **bit for bit**. The UI cell inspector reproduces the engine at 2,407 checked
  cell-substeps.
- **Positive controls:** Conway's Life was recovered **exactly** as 3 rules, and a known hidden-bit rule was recovered
  exactly.

**On Bad Apple** (executed autonomously; teacher-forced one-step match reported separately):

| interval | best result | rules / conditions / bytes | hold baseline | NCA |
|---|---|---|---|---|
| main 4500–4555 | **0.0995** (14 programs) | 117 / 560 / 1,252 B | 0.183 | C10 0.004 |
| L240 | 0.293 (30 programs) | 282 / 1,370 / 2,462 B | 0.401 | phases 0.019 |

- **One-step match is high but meaningless here:** it is 96–98 % (most pixels just keep their value), yet a single
  learned program replays as hold.
- **Change-weighted training** swings the rules to flicker: change mass 33–43×.
- **Extraction from the C10 NCA** (quantised hidden bits, k = 1–3) gives at best 0.153.
- **Conclusion:** readable rules reach **~0.10 error on a 56-frame clip, and only by switching programs every 4
  frames**. Their files are then about as large as the lossless recording (1,252 B vs 1,903 B). They do not
  reproduce the motion.
- **No hybrid:** no hybrid (NCA + explicit) model was built. None of the NCA results is presented as readable rules.

## 5. Where the spatial and temporal information lives, and what it costs
| model | weights | init / anchor | schedule | file |
|---|---|---|---|---|
| L240 4-phase | 4 rules × 41,280 B | first frame 216 B; no init | 4 entries in the header | 154,187 B |
| Song, longest passing replay (0–1467) | 26 phases × 41,280 B | first frame 216 B + init_g 1,620 floats (6,480 B, f32) | 26 entries | 1,004,632 B |
| Whole-song export (110 phases) | 110 × 41,280 B | first frame + init_g | 110 entries | 4,228,462 B |

- **The lossless recording is smaller:** the same binary frames as 1 bit/px + zlib-9 take far less
  (`runs/c12/eval/recording_baselines.json`):

  | frames | zlib-9 | model file |
  |---|---|---|
  | L240 | 13,429 B | 154,187 B |
  | 0–1467 | 43,315 B | 1,004,632 B |
  | whole song | 231,053 B | 4,228,462 B |
  | main 56 frames | 1,903 B | – |

  **These models are not compression.**
- **Spatial information** for a new shot is stored in **the per-phase rules and the carried hidden state**; the
  schedule is a list of frame numbers.
- **Joint training lets earlier phases prepare hidden state for later ones.** Greedy (phase-by-phase) training cannot,
  and it fails exactly at the wipe and shot change (0.099 / 0.085 vs 0.03 jointly).
- **The opening** (frames 0–42 are black) uses the learned low-res initial hidden.
  - The earlier claim that a black start cannot break symmetry was **wrong**: the zero boundary breaks it after one
    update (`opening_symmetry.json`).
  - Training without any stored initial information was not tested.

## 6. Longest success and what is left for the whole song
**Longest replay passing every C12 gate: frames 0–1467, 1,468 frames, 48.9 s, including the black opening.**
- Export: `runs/c12/song/noise_export_0-1467.ncab`; error 0.0104; 0/43 fades missed.
- Node vs numpy: 5 binary pixels differ, max |Δv| 1.8e-4.

**Whole song (110 automatic phases, chunked-joint, one seed, 6.69 GPU-h): FAILED as a replay.**
- Training reported low errors per chunk. The replay from the export (numpy, and JAX on CPU, identical) tracks
  training only to frame ~1,080; gates pass up to the 1,080-frame prefix.
- The rest degrades: full-song error 0.339, 113/191 fades missed, hidden values reach ~5e18, and Node vs numpy
  diverge.
- **Cause (measured):** the chained rules amplify tiny float-order differences between the GPU state used in training
  and the CPU replay. They fit one exact trajectory and are not robust.

**Repair (one bounded attempt): training-only state noise σ = 0.02.**
- Replay then equals training **in every chunk** (9 chunks, frames 0–2115).
- 0–2115: error 0.0157. It fails only the worst-window gate, at one fast wipe at frames 1507–1522 (0.159).

**Uniform-phase song run** (same chunked recipe without noise, 110 equal phases, 7.15 GPU-h): **failed the same way**.
- Full-song error 0.344; 114/191 fades missed.
- Longest gate-passing prefix: **480** frames, vs 1,080 for the automatic phases.
- Both runs are dominated by the same divergence and are single-seed, so this is **not** evidence that automatic
  segmentation is better at song scale.

**Remaining for the whole song:**
- Train the whole song with the noise recipe: ~28 chunks, est. ~7 GPU-h per seed.
- Fast wipes and cuts: more or better-placed phases around them.
- Hidden-state growth: bound it.
- Seeds: more than one.
- Storage: total cost is ~4 MB of float32 rules for the whole song, far above a recording.

## 7. UI and documentation
- **UI** (`ui/`): everything listed in §1.
  - The inspector is generated by the same code paths as the engines and checked bit-exactly (`node ui/test_inspector.mjs`).
  - **The C11 open item is closed:** UI upload → GPU training through gpuq (job 16, 10 s) → export → re-import, with
    browser = numpy = Node hashes.
- **Docs:**
  - `README.md` (layout borrowed from the README/doc structure of the user's VectorSophie/hgit; recorded in
    `docs/research/README.md`);
  - `docs/architecture.md`, `rule-models.md`, `format.md`, `evaluation.md`, `roadmap.md`;
  - `docs/research/README.md` (an index of every campaign record).
  - The campaign records were not moved.
- **Format rename:** the decision-list format is now `ca-universal/dlrule-v1`, to end a name collision with the
  legacy C1–C4 LUT exports. Readers accept the old name only with a `programs` field.
- **Formats:** `ncab-v2` (phased; v1 files still load everywhere).
- **Test data:** `tests/data/make_test_videos.py` regenerates the synthetic test clips (identical decoded frames).

## 8. Failures, untested items, budget, reproduce
**Failures:**
- The whole-song replay.
- Explicit rules on real motion.
- Automatic segmentation was not better than uniform.
- The greedy warm start exploded.
- The first wide fixed rule went NaN (at lr 5e-4; rerun at 1.25e-4).
- My cancel of a queued job was refused by the permission policy, so the pilot ran ~4.05 GPU-h against its 4 h cap.

**Loophole check:** a 3×3 majority blur of the TRUE original passes every pre-registered C12 gate (L240: error 0.023)
while erasing thin parts (0.59 / 0.62). The gates are kept as fixed, so every result is reported with
thin-part/gap recall, specks/holes and flicker.

**Not done:**
- type-3 (internal-state) phase switching;
- stage-2 boundary moves beyond the K comparison;
- training without stored initial information;
- multi-seed song runs;
- `ca.pipeline` support for v2 (v2 lives in `experiments/exp45_c12.py`).

**Budget:**
- **GPU (assigned GPUs × runtime per job, the user's definition): 23.03 of 24 GPU-h** (17 jobs).
  - The pilot was 4.05 GPU-h against its 4 h cap (the over-run is explained in §8).
  - Whole-song runs: 6.69 (auto) + 7.15 (uniform).
  - Noise repair: 1.56 + 0.68.
  - The UI agent's GPU check: 0.003.
- **CPU:** `runs/c12/cpu_ledger.jsonl`.
- **Queue waits** are recorded per job in `runs/c12/gpu_ledger.jsonl` and are not counted.

## 9. Reproduce
```sh
D=$(experiments/freeze.sh)                      # frozen snapshots used: runs/c12/frozen.txt
# L240 arms (pilot):  python $D/experiments/exp45_c12.py 1760:240 c12/pilot/uniform4 uniform --K 4 --steps 12000
# whole song:         python $D/experiments/exp45_c12.py 0:6572 c12/song/chunked_auto110 chunked --phases runs/c12/song_phases_auto110.json --seeds 0 --steps 8000 --init-factor 4
# noise repair:       ... 0:1468 c12/song/noise_prefix1468 chunked --phases runs/c12/song_phases_auto110_prefix1468.json ... --noise 0.02
python experiments/exp46_eval12.py runs/c12/eval/x.json <export.ncab> --node        # metrics + gates + Node check
python experiments/exp46_eval12.py runs/c12/eval/baselines.json --baselines 1760:240  # loophole baselines
python experiments/exp50_prefix.py out.json <export.ncab> 0 120                       # longest gate-passing prefix
python experiments/exp47_rules.py out.json 4500:56 direct --leaves 32 --K 14         # explicit rules
python experiments/exp49_videos12.py NAME 0:6572 "title=<export>" --diff 1 --no-gif   # videos (incl. full song)
python experiments/summarize12.py                                                     # results12.json / summary12.md
node ui/test_inspector.mjs; node player/nca_replay.mjs <export.ncab>; node player/rule_replay.mjs <file.rule.json>
```
