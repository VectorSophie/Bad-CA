# Roadmap: colour, resolution, frame rate, portability (proposal, nothing started)

Nothing here has been run. The costs below are scaled from measured numbers:
- C13: ~7 GPU-h for 6,572 frames at 48×36 (8,000 steps per ~235-frame chunk, ~15 min per chunk);
- this campaign's pilots: 0.056 s per step for a 120-frame chunk.

## 1. What a cell would hold

| option | state | pros | cons |
|---|---|---|---|
| binary (now) | 1 visible + 15 hidden floats | proven | black/white only |
| palette index | K visible logits (K = 4–16), argmax shown | crisp flat colours, natural for anime | needs palette extraction per video; softmax output |
| quantised RGB | 3 channels at 2–3 bits each | no palette step | banding; more visible channels |
| continuous RGB | 3 visible floats | simplest loss | a blur prior; flicker in flat areas |

**Recommendation:** start with a **palette of 8 colours** on a short clip.

**Measure:**
- colour consistency per region (do identical source colours stay identical);
- temporal flicker (one-frame flips per pixel, as for Fantasmagorie);
- model size.

**Expected first problem:** colour flicker at boundaries. This is the same mechanism as the binary specks
(README "Speckle"), with more classes.

## 2. Resolution

| | 48×36 | 64×48 | 96×72 | 192×144 |
|---|---|---|---|---|
| cells | 1,728 | 3,072 | 6,912 | 27,648 |
| cost per step vs 48×36 (convolutions, linear in cells) | 1× | 1.8× | 4× | 16× |
| cells a 3×3 rule with S = 4 substeps can reach per frame, as a fraction of the width | 8/48 | 8/64 | 8/96 | 8/192 |

- **Reach shrinks at higher resolution.** Fast motion and cuts need more substeps or a hierarchy (a coarse grid that
  coordinates a fine one). Otherwise the same 1-cell rule must pass information farther in the same number of steps.
- **Memory is not the limit.** Peak memory in the pilots was 0.3 GB; BPTT memory grows linearly in cells × frames ×
  substeps.
- **This campaign's evidence** is the G4 96×72 window runs (see the report): they test whether thin parts and slow
  boundaries become *learnable*, not just *visible*.
- **Recommendation:** 96×72 for one short clip of each video before any longer run. Decide on substeps from measured
  reach failures (cuts, fast wipes) rather than raising them blindly.

## 3. Frame rate
- **More frames are not more information.**
  - Bad Apple: 6,572 decoded frames, 4.2 % identical and 9.8 % near-duplicate consecutive frames.
  - Fantasmagorie: a 25 fps PAL transfer of a slower film (the decoded animation has 1,687 frames).
- **Measure distinct and near-distinct frames, not decoded frames** (`data/manifests/*.json`). Use source timestamps
  (`pts`); never assume 30 fps.
- **Frame-rate independence is possible but not free.** A CA with a fixed number of substeps per *source* frame is
  tied to that frame rate. Making it independent would need "substeps per second", with the schedule in seconds.
  That changes the training problem (the same rule must work at a different number of steps per change).
- **Interpolated frames are not observations.** Generating intermediate frames for smooth playback is display
  interpolation: report it separately and never count it as data.

## 4. Portability vs generalisation
These are different claims:
1. **Same code, retrained per video** (portability). This campaign tests it on Fantasmagorie.
2. **One trained rule set that animates an unseen video** (generalisation). The G5 zero-shot numbers in this
   campaign's report bear on it; nothing here suggests it works yet.
3. **Shared initialisation plus fine-tuning** (transfer). Not tested.

**Next steps, in order:**
- a third short black-and-white clip with the unchanged recipe;
- fine-tuning from the Bad Apple rule set versus from scratch, at equal steps;
- only then colour.

## 5. Suggested order and budget (to be approved separately)
1. 96×72 binary, one 240-frame clip per video, phased: ~1 GPU-h.
2. Palette-8 colour on one 240-frame public-domain colour clip: ~1–2 GPU-h, after a CPU check of palette extraction
   and flicker.
3. Fine-tuning versus scratch on a third video: ~1 GPU-h.
