# Reddit post draft (NOT posted; for review)

## Before posting: rules check (could not be verified automatically)
Reddit refused unauthenticated requests to the rules endpoints on 2026-11-13, and web search returned only stale
mirrors. Before posting, check by hand:
- [ ] the destination's current sidebar rules: self-promotion / own-content ratio, AI-generated-content policy, flair
  requirements, video vs. GIF vs. link posts;
- [ ] whether r/touhou wants Bad Apple derivative works under a specific flair (its wiki lists Bad Apple!! under
  viral/memetic videos; the "Video" flair existed in an older copy);
- [ ] for r/cellular_automata: whether a learned (neural) CA is on topic. Lead with the CA mechanism, not the song.
- [ ] Rights: the post shows derived 48×36 frames of Bad Apple!! (Alstroemeria Records / Anira PV, based on ZUN's
  Touhou). Decide whether to show the original next to the CA or the CA alone, and credit the original either way. The
  Fantasmagorie frames are public domain.

**Suggested destinations, in order:**
1. r/cellular_automata (mechanism-first version below);
2. r/touhou or a Bad Apple community (lighter version: lead with the video).

**Media:** one short video or GIF (≤ 20 s) of CA output next to the original, plus a link to the repository. The
full-song MP4 in `assets/demo/` is 3 min 39 s and 9.1 MB.

---

## Title options
1. "I tried to make Bad Apple!! with cellular automata. The full song works, but only with 110 learned rules, not one"
2. "Bad Apple!! played by a cellular automaton: every cell runs the same tiny neural rule on its 3×3 neighbourhood"
3. "What I learned trying to find a Game-of-Life-style rule for Bad Apple!!"

## Body

I wanted to know whether Bad Apple!! could come out of a cellular automaton: a grid where every cell looks only at its
neighbours and applies the same rule, like Conway's Game of Life.

**What works.** The whole song (6,572 frames at 48×36) plays from the first frame alone.
- Every cell has 16 numbers of state: 1 is shown, 15 are hidden memory.
- Four times per frame, each cell runs the same small network on its 3×3 neighbourhood.
- No video frames are fed in after the first one. You can replay the file in Python, Node or the browser and get the
  same result.
- Average error: about 1.5 % of pixels per frame.

**The catch, up front:**
- It is not one rule. The song is split into 110 phases, each with its own learned rule, switched on a fixed timeline.
  The hidden state carries over between phases.
- The model (4.2 MB) is much bigger than the video itself (231 kB zipped). It is a generator for this one video, not
  compression and not a general video model.

**What didn't work (most of the project):**
- **Exact rules found by SAT solvers:** they reproduce a few frames, then grow bigger than the frames themselves.
- **Small human-readable rules and learned logic-gate circuits:** they follow a shape for ~8 frames, then collapse
  into blobs and wedges. The same logic-gate method *did* recover a known rule exactly from scratch: 1,598 learned
  gates simplified to 9 gates that equal the Greenberg–Hastings excitable-wave rule on all 262,144 inputs. It just
  doesn't find one for Bad Apple.
- **Giving cells explicit velocity**, and nudging regions: with hindsight-chosen nudges it tracks motion, but no better
  than moving blocks of the image around.

**The lesson:** in this video, the same local picture is followed by different futures. The next frame isn't
determined by what a cell can see, so the information has to live somewhere: in hidden memory, in the schedule, or in
outside control. I ended up measuring how much of each is needed instead of hunting for one magic rule.

**Bonus:** the same training recipe, retrained from scratch on *Fantasmagorie* (Émile Cohl, 1908, public domain).
- *[insert the measured result from the campaign report]*
- One model does **not** transfer between the two videos.

Code, replay players and all the negative results: *[repository link]*

---

## Lighter version (Touhou / Bad Apple communities)
"Bad Apple!!, but every pixel is a cell that only looks at its 8 neighbours. The same tiny neural rule runs everywhere,
the hidden state carries the memory, and a timeline of 110 rules gets through the whole song. Original on the left, CA
on the right. Full write-up (including everything that failed) in the repo."
