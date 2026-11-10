# Architecture

CA-universal has four parts that share nothing but files on disk: preprocessing, training, export, and replay.
Training writes an export. Every replay, evaluation and UI view reads only that export plus, for scoring, the
preprocessed original. Nothing downstream of training touches the source video.

```
video ──prep──> binary grid (T,H,W) ──train──> checkpoint ──export──> .ncab / .nca.json / .rule.json
                      │                                                    │
                      │                    ┌───────────────────────────────┼──────────────────────────┐
                      │                    v                               v                          v
                      │            numpy reference replay        Node player (player/)      browser (ui/, Web Worker)
                      │                    │                               │                          │
                      └──── original ──> evaluation (ca/diag10.py, experiments/exp46_eval12.py) <──┘  │
                                           (Python vs Node agreement is checked separately)            │
                                                                       original shown only in comparison/error panels
```

## Preprocessing (`ca/pipeline.py prep`, `ca/video.py`)

- PyAV decodes only the requested interval. The decoder validates the file; the extension is not trusted.
- Colour → luma with PIL `convert("L")` (ITU-R 601-2: 0.299 R + 0.587 G + 0.114 B).
- Box resize to the chosen width; height = round(width × h / w), so the aspect ratio is kept.
- Binary = gray ≥ threshold (default 128), or < with `--invert`.
- Output fps: frame k is the source frame nearest to start + k / fps. No blending.
- Cache: `cache/pipe/<key>/` with `gray.npy`, `bin.npy`, `manifest.json`. The key is the sha256 of the source
  sha256 and the preprocessing settings, so width, threshold and fps each get their own entry.
- Limits (local tool, not a service): ≤ 2 GB file, ≤ 4096 px side, ≤ 600 frames per interval, width 8–256.
- The campaign experiments use `ca/video.py` (`gray_frames`, `binarize`) at width 48, threshold 128.
  `tests/test_pipeline.py` checks that `prep` produces the same Bad Apple target bit for bit.

## Models and training

| module | role |
|---|---|
| `ca/nca.py` | single-rule Neural CA: numpy float32 reference step/rollout, JAX step/rollout, `nca-v1` JSON export |
| `ca/nca_train.py` | single-rule trainer used by the pipeline and C11 (vmapped seeds, resumable) |
| `ca/phase_nca.py` | phased Neural CA: modes fixed / indep / delta, low-res initial hidden, `ncab-v2` export and loader |
| `ca/phase_train.py` | phased training: `joint` (full BPTT from frame 0 across boundaries), `greedy` (phase by phase on the carried state) |
| `ca/segment.py` | automatic phase boundaries: exact DP over K segments minimising an MDL code length, plus cut/fade/wipe detection |
| `ca/rules.py` | explicit dlrule-v1 programs: features, exact synchronous step, CART learner, JSON export, text rendering |
| `ca/ncab.py` | `ncab-v1` binary writer/reader |

Model details are in `docs/rule-models.md`.

Training is GPU work and goes through the shared `gpuq` queue on this machine. Each job runs from a read-only,
content-addressed snapshot of the code (`tools/freeze.sh`, `experiments/freeze.sh`); the snapshot id is recorded with
the results. Trainers checkpoint to `state*.npz` and resume. The best checkpoint is chosen by the binary error of an
autonomous rollout on the trained span; the initial weights are a candidate too.

## Replay engines

Each format has two independent implementations that share no code.

| format | Python reference | Node / browser |
|---|---|---|
| `nca-v1`, `ncab-v1` | `ca/nca.py` `rollout_np` | `player/nca_core.mjs` (CLI: `player/nca_replay.mjs`) |
| `ncab-v2` | `ca/phase_nca.py` `rollout_np` | `player/nca_core.mjs` (same CLI) |
| `dlrule-v1` (decision lists) | `ca/rules.py` `run` | `player/rule_core.mjs` (CLI: `player/rule_replay.mjs`) |
| legacy LUT `dlrule-v1`, `circuit-v1`, `.cab` (C1–C8) | `ca/rule.py`, `ca/export.py`, `ca/container.py` | `player/replay.mjs`, `player/decode_file.mjs` |

- Python and JS compute the Neural CA step in float32 storage; the JS player accumulates sums in float64 and rounds
  each stored value with `Math.fround`. Binary output usually matches exactly. Where it does not, the mismatch is
  reported per export.
- The explicit rule engines are integer-only and match bit for bit.
- The CLIs print a SHA-256 over the T×H×W binary frames, so two replays can be compared by hash.

## Pipeline CLI (`python -m ca.pipeline`)

`probe`, `prep`, `train`, `export`, `replay` (numpy vs Node), `video` (comparison MP4/GIF, decoded back), `run`
(the whole chain with `RUN/status.json`), `import` (make a run directory from an existing export). Presets: `demo`
(CPU, 1,304 params) and `quality` (GPU, 10,320 params); explicit flags override them and the effective values go
into the manifest.

The CLI handles single-rule v1 models only. Phased models (`ncab-v2`) and explicit rules come from the C12
experiment scripts (`experiments/exp45_c12.py`, `experiments/exp47_rules.py`).

## Local UI (`ui/`)

- `ui/server.py`: Python standard library HTTP server on 127.0.0.1. It refuses non-local Host headers, needs a
  custom header on every POST, validates run names, and resolves every path under an allowed root. Uploads are
  stored under hashed names, decoder-checked, and deleted on failure. Training runs `ca.pipeline` as a subprocess
  with an argument list (no shell), on CPU or through `gpuq`.
- `ui/static/`: plain HTML/CSS/JS, no build step, no CDN. `worker.js` runs `player/nca_core.mjs` in a Web Worker
  on the export alone. Rule-v1 programs run through `player/rule_core.mjs` in the page.
- Views: synchronised original / CA / error panels, a phase timeline, a `model.rules` panel (dlrule-v1 programs as
  text with per-phase diffs and fire counts; NCA phases as numbers only), a cell inspector that explains one cell's
  update with the same code path as the step, and dlrule-v1 editing with replay from frame 0.
- `ui/test_inspector.mjs` checks that the browser-side rollout and the cell traces equal the reference step.

Endpoints and limits: `ui/README.md`. Design sources: `ui/DESIGN.md`.

## Run artefacts

- `runs/c<N>/`: per-campaign configs, ledgers (`gpu_ledger.jsonl`, `cpu_ledger.jsonl`), exports, evaluations
  (`eval/*.json`), media.
- `runs/pipe/<run>/`: pipeline and UI runs. `example_*` directories are ready-to-play imports of campaign exports;
  `ui/make_examples.sh` rebuilds their comparison frames after a fresh clone (needs the Bad Apple file).
- `runs/c12/samples/`: small format samples (a v2 file holding the C10 rule as two phases; a Life-then-grow
  dlrule-v1 program).
- `cache/` and `refs/` are git-ignored.
