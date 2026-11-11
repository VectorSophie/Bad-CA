# Rule language: dlrule-v2

`dlrule-v2` extends `dlrule-v1` (`ca/rules.py`, `player/rule_core.mjs`). A v1 file is valid v2.

## State
- Each cell holds a visible bit `v` and `k` hidden bits `h0..h(k-1)`. Everything is discrete; there are no numeric rules
  in this track.
- **Initial state:** `v` comes from the file's first frame. Hidden bits are 0 unless the file stores `h0`, which is then
  counted in the size.

## Features a condition can read
| feature | meaning | range |
|---|---|---|
| `v`, `h<i>` | own visible and hidden bits | 0–1 |
| `n8` | number of the 8 Moore neighbours that are on | 0–8 |
| `r2` | number of on cells on the radius-2 ring (16 cells) | 0–16 |
| `N` `S` `E` `W` | on cells in the 3-cell row or column on that side | 0–3 |
| `u` `d` `l` `r` | the single neighbour above, below, left and right (**new in v2**) | 0–1 |
| `m<i>` | number of Moore neighbours with hidden bit i set | 0–8 |

- **Reach:** `r2` makes the neighbourhood radius 2. One substep therefore moves information at most 2 cells (Chebyshev
  distance), and one frame moves it at most 2·S cells.
- There are no coordinate, time or random inputs.

## Semantics
1. **Rule order:** a program is an ordered list `if c1 and c2 ... -> v=a, h=b`, where each condition is
   `feature (<=|>=|==) constant`. The first rule whose conditions all hold fires. If none fires, the cell keeps its state.
2. **Updates:** synchronous. Every cell reads the state from before the substep.
3. **Boundary:** zero; cells outside the grid are off and have hidden 0.
4. **Substeps:** `S` substeps make one frame. Only frame times are compared with video.
5. **Schedule:** `[{start, end, program}]` says which program runs on frames `[start, end)`. This is an external
   timeline. A file with a single program and a single entry is a **fixed-rule CA**.
6. **Derived programs (new in v2):** `{"base": j, "edits": [...]}` means program j (an earlier one) with edits applied
   in order:
   - `set` replaces rule i;
   - `insert` adds a rule at i;
   - `delete` removes rule i.

   Sharing and per-phase change are thus stored explicitly and counted.

**Engine agreement:** Python and Node produce bit-identical results. This was checked on random v2 programs with
k = 0, 1, 2 and derived programs, and on every example exported for the UI. The UI cell inspector
(`ui/test_inspector.mjs`) checks the fired rule and the next state against the engine.

## Complexity (what "readable" is measured by)
`rules.complexity()` reports the following:
- **Program counts:** programs; full and derived programs; edits; phases; distinct programs used; reused programs.
- **Rule size:** rules and conditions, counting only the stored text, so derived programs count only their edit rules;
  constants (conditions + 2 per rule); hidden bits.
- **Bytes:**
  - `rules_json_bytes`, which includes the schedule;
  - `file_bytes`, the zlib-compressed rules + first frame + h0;
  - the first-frame and h0 bytes separately.

The reports in FINDINGS also give schedule bytes separately.

Rule counts are never compared with Neural CA parameter counts; the comparison is by error, motion, and total bytes.
