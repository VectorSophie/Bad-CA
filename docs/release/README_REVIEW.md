# README and repository presentation: review and recommendations

**Scope:** the README on `master` (`eed06a6`), reviewed on 2026-11-13. Nothing on `master` was changed; the
recommendations below are for the maintainers to accept or reject.

## What already works well
- The demo comes first and is honest:
  - "time-varying CA with an *external* schedule";
  - no frames injected after the first;
  - "not compression" (4.2 MB vs 231 kB).
- Limits sit next to the results: fast wipes, speckle, seeds.
- Replay is independent (numpy, Node, browser), and the instructions say what works without the source video.
- The data and licence section says the video is not included.

## Recommended changes, in priority order

1. **Add a one-paragraph "what this is / is not" box under the title.**
   - Suggested text: *"A Neural CA with 110 scheduled rules replays one specific video from its first frame. It is not
     a single Conway-style rule, not a general video generator, and not compression."*
   - Why: readers arriving from a short post will look for exactly this distinction.

2. **Add a results gallery** that links the actual-output assets in `assets/release/`:
   - `nca_vs_original.gif`, `nca_state.gif`;
   - `logic_gh_9gates.gif`, the synthetic success;
   - `logic_realvideo_failure.gif`, the real-video failure;
   - `shape_ring.gif`;
   - the Fantasmagorie replay, if it is accepted.

   Every asset needs its caption from `assets/release/captions.json`. Failures belong in the gallery too.

3. **Rewrite "Why this approach" to point to one history page.** Per stage it would give the genuine result and the
   limitation, using the draft `docs/release/PROJECT_HISTORY.md`. Today the later work (Readable Rules NO-GO, Path D,
   context codebook, Path V, external video) lives only on unmerged branches and in `docs/` sub-folders.

4. **Reproducibility table** (one row each for the featured export, Path D examples and Path V): file, SHA-256, the
   command to replay, the expected output hash.
   - Featured export: file `facb43d9…`.
   - Node replay frames: `277fff0f…`, recorded 2026-11-13 in this campaign. No previous record of the output hash
     existed; adding one makes "the replay is unchanged" checkable.

5. **Data registry note.** `data/manifests/*.json` and `ca/datasets.py` (this branch) describe each video:
   - source, hash, timing, rights, preprocessing;
   - which derived files are committed: Fantasmagorie (public domain) yes, Bad Apple no.

6. **Attribution section.**
   - Fantasmagorie: "Fantasmagorie (1908) by Émile Cohl (Gaumont), public domain; file from Wikimedia Commons (PAL DVD
     edition)."
   - Bad Apple!!: keep the current wording, and state that the derived 48×36 frames in GIFs and videos are shown for
     comparison only.
   - Whether derived Bad Apple frames may be shown publicly is a rights question for the maintainers; this review does
     not settle it.

7. **Installation.**
   - Pin Node ≥ 24 and Python 3.12 (both already stated); add one `make check` style command that runs the existing
     tests.
   - `tests/test_path_d.py` needs Node, and the GPU is optional for replay.

8. **Remove internal jargon from the top level.**
   - "C13", "Path D", "chunked-joint" belong in `docs/`.
   - The README can say "trained in chunks of ~8 s with the state carried over" (it mostly already does).

## Not recommended
- Do not move the featured demo to a logic-rule or velocity-CA result. None of them reproduces more than short
  selected segments.
- Do not present Path V or Track S animations without "oracle-controlled" in the caption.
