# Bad CA!! · local web UI

```
.venv/bin/python ui/server.py            # http://127.0.0.1:8765/
.venv/bin/python ui/server.py --port 8790 --host 127.0.0.1
```
Stop with Ctrl-C. CPU training jobs run in their own session and keep running after the server stops; cancel them from
the UI first (or `kill -TERM -<pgid>` with the pgid in `runs/pipe/<run>/ui_job.json`).

Python stdlib + the project's `ca.pipeline` (numpy, PyAV). No build step, no CDN. The CA frames are computed in the
browser from the export alone (`player/nca_core.mjs` in a Web Worker); the original frames are fetched only for the
comparison and error panels.

Open a given model directly: `http://127.0.0.1:8765/?run=example_c10_main`. Views: `#demo` (default), `#lab`, `#research`.

**Default screens (demo, rule lab) use plain names only**: the featured model's `display_name` verbatim, no campaign
numbers, file names or `runs/` paths. Provenance (parameter count, precision, file size, sha256 and whether it matched,
source path, the manifest's `model` / `status` / `reason` / `source` / `metrics`) sits in the collapsed **Model details**.
Problems that affect playback stay visible outside it: sha256 mismatch, "computing ahead at N frames/s" while slower than
real time, "no original for comparison". `[ Open your own video ]` (`#new`) jumps to `run.new`; `[ save model file ]`
downloads the export being played. **How it was built** (`#research`) is the decision flow A–F with small record links;
the campaign log, shelves and model details are inside the collapsed "All saved models (research archive)". The header
shows `/assets/logo/bad-ca-mark.png` (server: read-only `/assets/{logo,figures}/*.png`), favicon `static/favicon.png`.
Light theme: same CSS variables under `prefers-color-scheme: light` (text contrast ≥ 4.8:1); the CA video stays black/white.

## Views (C13)
- **demo** (default): one model. By default the featured model named in `runs/featured/manifest.json` (`featured.export`,
  `display_name`, `reason`, `sha256`, `fps`; nothing is hard-coded, switch the manifest to switch the demo). The browser
  hashes the file it plays and shows a warning if it differs from `sha256`. Left: the CA video; right: the rule in effect
  (phase i of K, frames, seconds = frame / fps, rule id and the phases that share it, the update written from
  `substep()` with structure in blue and parameters as red underlined badges, labelled in words; click a parameter for a
  heatmap + paged values, also as change vs the previous phase; per phase: parameter count, changed values |Δ| > 1e-6,
  L2 and max |Δ|; at a boundary frame the switch is highlighted, never interpolated). Bottom: timeline with phase
  boundaries, the computed range and the playhead (click/drag to seek). `[ tools ]` (off by default): original, error
  view, continuous s₀ (paused), zoom, speed, frame/substep stepping, compare with B. Below: cell inspector (`traceCell`,
  checked against the real substep). A research export opened from the shelf plays here with a "not the featured model"
  banner and a way back.
- **rule lab**: dlrule-v1 explicit rules only (synthetic samples + `runs/c12/rules/*` marked experimental), with the rule
  list, fire counts, per-boundary diff, edit + replay from start, and the rule cell inspector.
- **research**: campaign timeline C1–C13, the shelves (examples, runs, imported, runs/c12), model details (architecture,
  curve, checkpoints, export download) and `run.new` (upload / preview / train / cancel / resume).

## Whole-song playback
`static/worker.js` computes frames strictly in order from the export (frame-index based, no clock decides what is
computed), posts packed binary frames (1 bit/pixel) about every 100 ms and a full-state checkpoint every `every` frames
(`every` = max(4, ceil(T·C·H·W·4 B / 48 MB))). The page plays only computed frames (it waits at the buffer edge), shows
"computed up to frame N", and regenerates any other state (substeps, inspector) from the nearest checkpoint with
`substep()`. Original frames are never used as state.

Speed (10,320-param featured model, 48×36, S=4): `nca_core.substep()` was reordered (weights outer, contiguous pixel loop
over a zero-padded float64 copy, 4 features / 4 channels per pass). It adds the same float64 terms in the same order per
value, so results are bit-identical (`substepRef()` keeps the old loops; the test compares them). Node: 5.4 → 24.4
frames/s (full 1,467-frame replay 199.8 s → 60.0 s); headless Chromium: 4.9 → 21.0 frames/s. Below real time (30 fps),
so playback of a freshly loaded model may wait at the buffer edge.

## Endpoints
| method | path | what |
|---|---|---|
| GET | `/`, `/static/*`, `/player/<name>.mjs`, `/assets/{logo,figures}/*.png` | UI; `player/` served read-only, `.mjs` only; assets read-only, `.png` only |
| GET | `/api/config` | presets (with notes), limits |
| GET | `/api/videos` | built-in (Bad Apple, `tests/data/color_disc_96x64.mp4`) + uploads |
| GET | `/api/video/<id>/probe` | decoder-validated metadata |
| POST | `/api/upload` | raw body, `X-Filename` header (display only); ≤ 200 MB; stored as `runs/ui/uploads/<sha256[:16]><ext>`, probed, deleted on failure (400 + message) |
| POST | `/api/preview` | `{video,start,frames,width,threshold,fps,invert}` → gray + binary frames (base64 uint8 + shape) + prep manifest |
| POST | `/api/train` | `{…preview fields, preset, steps?, device:"cpu"\|"gpu"}` → `runs/pipe/ui_<time>_<rand>`, `python -m ca.pipeline run …` |
| GET | `/api/runs` | runs/pipe/* (example_* first) + imported exports |
| GET | `/api/runs/<name>` | manifest, status, progress, hist (live from `train/state.npz` while training), log tail (40 lines) |
| POST | `/api/runs/<name>/cancel` | own jobs only: SIGTERM to the job's process group, writes `ui_status.json` |
| POST | `/api/runs/<name>/resume` | own jobs only: relaunch with `--resume` |
| GET | `/api/runs/<name>/export` | the `.ncab` (or nca-v1 JSON) download |
| GET | `/api/runs/<name>/original` | original binary frames (comparison only) |
| GET | `/api/featured`, `/api/featured/export`, `/api/featured/original` | featured manifest + server-side sha check + header; the file; original frames from its meta (comparison only) |
| GET | `/api/runs/c12~<path>` , `…/export`, `…/original` | read-only listing of `runs/c12/**/*.ncab` and `**/*.rule.json` (`~` = `/`); original frames from the export meta (`src_start`, or `spec` "start:T") + T frames of Bad Apple at `meta.width`, threshold 128, via `ca.pipeline.prep`, with `first_frame_mismatch_px` vs the export (comparison only) |
| POST | `/api/import-export` | raw `.ncab` / nca-v1 `.json`, ≤ 5 MB, validated with `ca.pipeline.load_export`, stored in `runs/ui/imported/<sha16>/`, listed as `imp-<sha16>` (CA-only playback) |

All POSTs need header `X-CA-UI: 1` (blocks cross-site form posts); requests with a Host other than localhost are refused.
Run names are checked against `^[A-Za-z0-9_][A-Za-z0-9_.-]{0,79}$` / `^imp-[0-9a-f]{16}$` and every path is resolved under
its allowed root. Status `died` = the pipeline wrote `running` but its process is gone.

## Limits
Upload 200 MB (pipeline accepts ≤ 2 GB, ≤ 4096 px side), interval ≤ 600 frames, width 8..256, exports ≤ 5 MB.
`demo` preset = 1,304 params, meant for ≤ 32 px × ≤ 24 frames on CPU; `quality` (10,320 params) needs the GPU.
CPU jobs use `JAX_PLATFORMS=cpu` and 4 threads. Device `gpu` submits `gpuq -L ui-<run> -G 1 env -u JAX_PLATFORMS
XLA_PYTHON_CLIENT_PREALLOCATE=false sh -c 'exec "$@" >>log.txt 2>&1' …` (output goes to the run's `log.txt`); status is
`queued` while gpuq reports queued/allocating, `died` if gpuq says finished without a final status; cancel uses `gpuq -r`
(waiting) or `gpuq -k` (running), only for the run's own job id.
Browser replay of a 10,320-param 48×36 model: ≈ 21 frames/s (streamed; playable while it computes).

Screenshots: `ui/screenshots/`. Design notes: `ui/DESIGN.md`.

## Phases, rules, inspector, editing (C12)
- **Formats**: ncab-v1 (one NCA rule), ncab-v2 (phased NCA, external schedule, `ca/phase_nca.py`), dlrule-v1 explicit rules
  (`ca/rules.py`). NCA frames: `player/nca_core.mjs` `rolloutStates` in the worker (bit-identical to `rollout`, checked by
  the test); dlrule-v1: `player/rule_core.mjs` `step` in the page (checked against `run()` on load).
- **Phase timeline**: segments = schedule entries (frame 0 is the given first frame), boundary lines, current phase, rule /
  program id and reason; click a segment, boundary or table row to seek. Seconds = frame / fps; 1 video frame = S substeps.
  "compare with B" loads a second export; panels original / A / B (+ error A, error B), synchronised by frame index.
- **Stepping**: `[ +1 substep ]`, `[ +1 frame ]`. Substep states come from frame states computed in the browser from the
  export (NCA: `substep()` from the nearest cached frame state; rules: the stored substep sequence), never from the original.
- **model.rules**: dlrule-v1 programs as text in priority order (default keep), JSON bytes per program, per-phase diff vs the
  previous phase (rules matched by their condition list), fire counts over a frame range (re-runs `step()` with a fired
  array and counts per cell per substep). NCA: C, Fh, S, K, mode, rank, per-phase effective params, delta params, changed
  params (|Δ| > 1e-6) and L2 vs the previous phase, raw stored bytes. Weights are never shown as rules.
- **Editing (dlrule-v1)**: programs JSON in a textarea, validated (features, ops, integer constants, v/h ranges, programs used
  by the schedule). `[ replay from start ]` (default) recomputes every substep from frame 0; `[ experimental: continue from
  current state ]` keeps substeps 0..current of the previous version and is labelled as a hybrid; every apply bumps the
  version counter and replaces all cached frames, fire counts and inspector state.
- **cell.inspector**: click a CA cell. dlrule-v1: `explainCell()` (built on `features()` / `fire()`): features, every
  condition evaluated, first failing condition of each earlier rule, firing rule, next state, and a check against the
  stepped state. NCA: `traceCell()` (same arithmetic as the substep): 3×3 neighbourhood of all channels, pre/post-ReLU of
  the Fh features, Wu·p, bias, next state, and a bit-exact check against the stepped state.
- Test: `node ui/test_inspector.mjs [extra exports…]` → PASS/FAIL lines (≈4 min; the two samples, example_c10_main, and
  `substep()` == `substepRef()` bit-exact, including the featured model across its first two phase boundaries).
