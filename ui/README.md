# CA-universal · replay bench (local web UI)

```
.venv/bin/python ui/server.py            # http://127.0.0.1:8765/
.venv/bin/python ui/server.py --port 8790 --host 127.0.0.1
```
Stop with Ctrl-C. CPU training jobs run in their own session and keep running after the server stops; cancel them from
the UI first (or `kill -TERM -<pgid>` with the pgid in `runs/pipe/<run>/ui_job.json`).

Python stdlib + the project's `ca.pipeline` (numpy, PyAV). No build step, no CDN. The CA frames are computed in the
browser from the export alone (`player/nca_core.mjs` in a Web Worker); the original frames are fetched only for the
comparison and error panels.

Open a given model directly: `http://127.0.0.1:8765/?run=example_c10_main`.

## Endpoints
| method | path | what |
|---|---|---|
| GET | `/`, `/static/*`, `/player/<name>.mjs` | UI; `player/` served read-only, `.mjs` only |
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
Browser playback of a 10,320-param, 56-frame 48×36 model takes about 8 s once, then it is cached for the session.

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
- Test: `node ui/test_inspector.mjs [extra exports…]` → PASS/FAIL lines (≈45 s; runs the two samples and example_c10_main).
