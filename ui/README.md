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
| POST | `/api/import-export` | raw `.ncab` / nca-v1 `.json`, ≤ 5 MB, validated with `ca.pipeline.load_export`, stored in `runs/ui/imported/<sha16>/`, listed as `imp-<sha16>` (CA-only playback) |

All POSTs need header `X-CA-UI: 1` (blocks cross-site form posts); requests with a Host other than localhost are refused.
Run names are checked against `^[A-Za-z0-9_][A-Za-z0-9_.-]{0,79}$` / `^imp-[0-9a-f]{16}$` and every path is resolved under
its allowed root. Status `died` = the pipeline wrote `running` but its process is gone.

## Limits
Upload 200 MB (pipeline accepts ≤ 2 GB, ≤ 4096 px side), interval ≤ 600 frames, width 8..256, exports ≤ 5 MB.
`demo` preset = 1,304 params, meant for ≤ 32 px × ≤ 24 frames on CPU; `quality` (10,320 params) needs the GPU.
CPU jobs use `JAX_PLATFORMS=cpu` and 4 threads. Device `gpu` submits through `gpuq -G 1` (implemented, untested).
Browser playback of a 10,320-param, 56-frame 48×36 model takes about 8 s once, then it is cached for the session.

Screenshots: `ui/screenshots/`. Design notes: `ui/DESIGN.md`.
