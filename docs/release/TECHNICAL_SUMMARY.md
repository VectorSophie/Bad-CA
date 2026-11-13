# Bad CA!!: technical summary (draft for public presentation)

## What runs as a cellular automaton
- **The grid.** 48×36 cells for Bad Apple. Each cell holds 16 floating-point channels: channel 0 is shown
  (thresholded at 0.5) and channels 1–15 are hidden memory.
- **One update.** Every cell reads its own 3×3 neighbourhood (all 16 channels, zero outside the frame) and applies the
  same small neural network to it:
  - a 3×3 convolution to 64 features, ReLU, then a linear map back to 16 channels;
  - the result is added to the cell's state (a residual update).
- **Timing.** All cells update at once, 4 times per video frame.
- **Parameters.** 10,320 numbers per rule (`ca/nca.py`).
- **Phases.** The featured model has **110 such rules**, and an external timeline says which rule is active at each
  frame (`ca/phase_nca.py`, format ncab-v2). This is a *time-varying* CA with a schedule, not one fixed law.
- **Start.** The first frame plus a learned low-resolution hidden array of 1,620 numbers.
- **Nothing else is read during playback.** No target frame is injected after the first. Python, Node and browser
  players replay the file independently; Node and browser are bit-identical, and numpy differs in 92 threshold pixels
  over the whole song.

## What is learned and what is designed

| part | learned | designed by hand |
|---|---|---|
| per-phase rule weights | yes (gradient descent through 4-substep rollouts, BPTT) | architecture, S = 4, channel count |
| phase boundaries | automatic segmentation (minimum-description cost); equal spacing was not worse | the number of phases |
| initial hidden array | yes | its resolution |
| training-only state noise (σ = 0.02) | — | yes: makes replay robust to float-order differences between devices |

## Is it compression? No.

| item | size |
|---|---|
| featured model | 4.2 MB (110 × 10,320 float32) |
| the same 6,572 binary frames, 1 bit/pixel + zlib | 231 kB |
| raw packed bits | 1.42 MB |

The model is a **generator that replays one video**, not a compressor.

## What generalises (measured in this campaign, `runs/gen/summary.json`)
**Same recipe, retrained on another video** (*Fantasmagorie*, 1908, public domain):
- replays all 1,687 animation frames from the first frame with 28 phases;
- mean error 0.066 vs 0.015 for Bad Apple; boundary F1 0.93; no learned film flicker;
- the C13 learning rate diverged once and was lowered to 2e-4;
- this is portability of the method, not of a learned model.

**Phase count** (Bad Apple 1760–1999, 240 frames, 4,000 steps):

| phases | reconstruction error | motion F1 | error on 32 later, unseen frames (hold 0.113) |
|---|---|---|---|
| 1 | 0.065 | 0.11 | 0.082 |
| 4 | 0.032 | 0.34 | 0.181 |
| 16 | 0.011 | 0.72 | 0.435 |

- More phases fit better and extrapolate worse.
- Restarting a phase from the true frame with empty hidden state is 3–20× worse than continuing with the carried state.
  The rules depend on accumulated hidden memory, not only on the visible picture.

**One shared rule trained on many short windows of a video, tested on held-out windows** (2 observed frames, then 15
generated):
- Bad Apple at 48×36: about as good as holding the last frame (0.151 vs 0.155), motion F1 about 0.05.
- More training data helps a little. 96×72 helps a little more (one-step error 21 % below hold).
- Across videos, zero-shot fails in both directions.

## Not claimed
- That the 110-phase model has discovered a law of animation.
- That any small readable rule reproduces Bad Apple. The logic-circuit and readable-rule tracks did not get there; see
  the history page.
- That oracle-controlled velocity or shape CAs generate motion by themselves. Their controls were chosen by looking at
  the target.
