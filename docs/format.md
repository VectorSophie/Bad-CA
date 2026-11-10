# Export formats

Every export is self-contained: architecture, weights or rules, schedule, first frame and any stored initial state.
A replay needs the file and nothing else. All bit-packed frames use `numpy.packbits(..., bitorder="little")` over
the row-major H×W grid (bit i of the stream is cell i; byte i holds cells 8i..8i+7, least significant bit first).

| format | kind | writer | readers |
|---|---|---|---|
| `nca-v1` | JSON, one Neural CA rule | `ca/nca.py export` | `ca/nca.py load`, `ca/pipeline.py load_export`, `player/nca_core.mjs parseNcaJson` |
| `ncab-v1` | binary, one Neural CA rule | `ca/ncab.py write` | `ca/ncab.py read`, `ca/pipeline.py load_export`, `player/nca_core.mjs parseNcab` |
| `ncab-v2` | binary, phased Neural CA | `ca/phase_nca.py export` | `ca/phase_nca.py load`, `player/nca_core.mjs parseNcab` |
| `dlrule-v1` | JSON, explicit rule programs + schedule | `ca/rules.py export` | `ca/rules.py load`, `player/rule_core.mjs parseRules` |

**Compatibility.** v1 files still load in every reader listed for them, including the Node player and the UI. v2
is read by `ca/phase_nca.py` and `player/nca_core.mjs`; `parseNcab` dispatches on the version byte, so
`node player/nca_replay.mjs` replays both. A v2 file with `mode=indep` that holds the C10 rule in two phases
(`runs/c12/samples/c10_main_as_v2_2phase.ncab`) replays to the same SHA-256 as the v1 file (`f2401adb…`). The
`ca.pipeline` CLI reads v1 only.

## nca-v1 (JSON)

| field | value |
|---|---|
| `format` | `"ca-universal/nca-v1"` |
| `dtype` | `"float32 little-endian"` |
| `C`, `Fh`, `S` | channels, perception features, substeps per frame |
| `n_frames` | T, frames in the replay, including the first |
| `height`, `width` | grid size |
| `boundary` | `"zero padding, all channels"` |
| `hidden_init` | `0.0` |
| `threshold` | `0.5` (visible > threshold → white) |
| `first_frame_b64` | base64 of the packed first frame |
| `params` | `{Wp, bp, Wu, bu}`, each `{shape, b64}` (base64 float32 LE, C order) |
| `step` | the update in words, including the kernel index convention `(f, c, dy, dx)`, dy, dx = 0..2 for offsets −1..1 |
| `n_params` | total parameters |
| `meta` | free-form provenance (source interval, training run) |

Readable but large: the C10 model is 56,234 bytes as nca-v1.

## ncab-v1 (binary)

```
"NCAB" | u8 version = 1 | u8 flags (bit 0: the rest is zlib-deflated) | rest
rest = u32le header_len | header JSON (UTF-8) | weights | packed first frame
```

- Writers always set flags = 1 and deflate at level 9. Readers accept both.
- Weights are `Wp, bp, Wu, bu` in `header.params` order, little-endian, as `weight_dtype`.
- `f16` is storage only: weights are widened to float32 on load, and state and compute stay float32.

Header fields: `C`, `Fh`, `S`, `T`, `H`, `W`, `threshold` (0.5), `hidden_init` (0.0), `boundary` (`"zero"`),
`weight_dtype` (`"f32"` or `"f16"`), `params` (`[[name, shape], ...]`), `meta`.

## ncab-v2 (binary, phased)

Same container as v1 with version byte 2:

```
"NCAB" | u8 version = 2 | u8 flags | deflate( u32le header_len | header JSON | float blobs | packed first frame )
```

| header field | meaning |
|---|---|
| `C`, `Fh`, `S`, `T`, `H`, `W` | as in v1 |
| `threshold` | 0.5 |
| `boundary` | `"zero"` |
| `update` | `"synchronous, S substeps per frame, residual"` |
| `weight_dtype` | `"f32"` or `"f16"` |
| `mode` | `fixed` (K = 1), `indep` (K stacked rules), `delta` (base + per-phase low-rank deltas) |
| `K` | number of rules |
| `schedule` | `[{start, end, rule, reason}]`: frames t in [start, end) are produced from frame t−1 by rule `rule`. `reason` is informational: from `ca/segment.py` it is `start`, `cut`, `wipe`, `mdl`, or `uniform` (equal-length phases); samples may use other strings |
| `schedule_semantics` | the sentence above, stored in the file |
| `init` | `{kind: "zero"}` or `{kind: "lowres", factor: f, hidden_channels: C−1}` |
| `blobs` | `[[name, shape], ...]`: the order of the float arrays after the header |
| `meta` | provenance, including `src_start` (first source frame) |

Blobs by mode:
- `fixed` / `indep`: `Wp (K,Fh,C,3,3)`, `bp (K,Fh)`, `Wu (K,C,Fh)`, `bu (K,C)`.
- `delta`: base `Wp (Fh,C,3,3)`, `bp`, `Wu`, `bu`, then `Ap (K,Fh,r)`, `Bp (K,r,C·9)`, `dbp (K,Fh)`, `Au (K,C,r)`,
  `Bu (K,r,Fh)`, `dbu (K,C)`. Effective rule k: `Wp + (Ap_k @ Bp_k).reshape(Fh,C,3,3)`, `bp + dbp_k`,
  `Wu + Au_k @ Bu_k`, `bu + dbu_k`, combined in float64 and rounded to float32 once.
- Optional last blob `init_g (C−1, ⌈H/f⌉, ⌈W/f⌉)`: learned low-res initial hidden, nearest-upsampled by f and
  cropped to H×W.

Frame 0 is the stored first frame. The schedule covers frames 1..T−1.

## dlrule-v1 (JSON, explicit rules)

| field | value |
|---|---|
| `format` | `"ca-universal/dlrule-v1"` (files written before the rename say `"ca-universal/rule-v1"`; readers accept that only together with a `programs` field) |
| `k` | hidden bits per cell (0–3) |
| `S` | substeps per frame |
| `T`, `H`, `W` | frames, grid size |
| `features` | feature names for this k (`v, h0.., n8, r2, N, S, E, W, m0..`) |
| `semantics` | `"first firing rule wins; default keep; synchronous; zero boundary; S substeps per frame; schedule = external timeline"` |
| `programs` | `[{rules: [{if: [[feature, op, const], ...], then: {v: 0/1, h: int}}]}]`; op in `<=`, `>=`, `==`; `h` packs the hidden bits, bit i = h_i |
| `schedule` | `[{start, end, program}]`, same frame semantics as v2 (note the key is `program`, not `rule`) |
| `first` | hex of the packed first frame |
| `h0` | hex of the packed initial hidden bits (k×H×W), or `null` for all zero |
| `meta` | provenance |

**Former name collision (resolved in C12).** The Campaign 01–04 LUT exports written by `ca/export.py` (read by
`player/replay.mjs`) carry `format: "ca-universal/rule-v1"` with different fields (`taps`, `luts_b64`, `channels`, ...).
The C12 decision-list format was renamed to `ca-universal/dlrule-v1`; every C12 file under `runs/c12` was rewritten
with the new name. Readers (`ca/rules.py`, `player/rule_core.mjs`, the UI) also accept the old name only when the
file has `programs`, so the legacy LUT files are never mistaken for decision lists.

Older Boolean formats (`circuit-v1` JSON, Campaign 05 `.cab` containers) are described in the module docstrings of
`ca/export.py`, `ca/container.py`, `player/replay.mjs` and `player/decode_file.mjs`.

## Byte accounting

`ca/ncab.py write` and `ca/phase_nca.py export` return a breakdown measured before compression, plus the real file
size:

| key | what it counts |
|---|---|
| `weights` | all float blobs, at the stored dtype (v2: includes deltas and `init_g`) |
| `first_frame` | packed first frame, ⌈H·W/8⌉ bytes (216 at 48×36) |
| `header` | JSON header + 4-byte length + 6-byte prefix |
| `raw_total` | uncompressed file size |
| `file` | actual bytes on disk after deflate |
| `n_params` | parameter count (excludes `init_g`) |
| `init_values` | v2 only: number of stored initial-hidden values |

`dlrule-v1` complexity (`ca/rules.py complexity`) reports programs, rules, conditions, constants, hidden bits,
`rules_json_bytes` (compact JSON of k, S, programs, schedule), `file_bytes` (zlib-9 of that JSON + packed first
frame + packed h0), `first_frame_bytes` and `h0_bytes`.

**Recording baseline** (`experiments/exp43_storage.py`): the same binary target frames, packed at 1 bit/px, then
zlib level 9 (lzma-9e for reference). It is a lossless recording; the models are approximate.

Measured in C11 (`docs/CHECKPOINT_11.md` §3):

| | weights | first frame | header | file f32 | file f16 | nca-v1 JSON |
|---|---|---|---|---|---|---|
| 10,320 params (C10) | 41,280 | 216 | 217 | 38,839 | 19,630 | 56,234 |
| 3,884 params | 15,536 | 216 | ~440 | 14,833 | 7,599 | – |
| 1,304 params | 5,216 | 216 | ~430 | 5,225 | 2,819 | – |

| interval | recording, zlib-9 | lzma |
|---|---|---|
| 4500–4555 (56 frames) | 1,903 B | 1,580 B |
| 1800–1859 (60 frames) | 4,053 B | 3,060 B |
| 1760–1879 (120 frames) | 7,532 B | 5,500 B |

Every model file is larger than the recording of the frames it approximates. There is no compression advantage so
far. f16 weights change the error by ≤ 0.0002 and flip 3–88 pixels per replay; float32 stays the default.
