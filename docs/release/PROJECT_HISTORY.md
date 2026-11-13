# Bad CA!!: the project history in one page (draft for public presentation)

**The question:** can a cellular automaton (CA), a grid of cells that each update from their neighbours by a local
rule, play *Bad Apple!!*?

Below is every major attempt, with **the genuine result** and **the limitation**. Detailed records stay in `docs/`, and
none of them were rewritten. Several of these lines of work live on unmerged research branches
(`context-s0`, `research/velocity-ca-v0`, `research/generalization-shape`).

| # | attempt | genuine result | limitation |
|---|---|---|---|
| 1 | **Conway-style fixed rules**: lookup tables, Life-like rules, SAT-synthesised circuits with memory | exact replay of a few frames; proofs of *where* a radius-limited rule cannot be exact | rules grow larger than the frames they replay; nothing transfers to the next clip ([journey](../research/journey.md) §1–3) |
| 2 | **Readable rules**: decision lists, short windows, phases | fitted programs replay their own 16-frame window | they transfer with real motion to 8 of 153 adjacent windows and 0 of 117 distant ones. Even the best possible local table does not carry over one step (NO-GO, `docs/readable-rules/FINDINGS.md` §10) |
| 3 | **Neural CA** with hidden state and 110 scheduled phases | **the whole song**, 6,572 frames at 48×36, from the first frame alone; mean error 0.0152; three trained variants pass all acceptance criteria; independent Python / Node / browser replays | 110 scheduled rules (not one law); 4.2 MB (bigger than the video, 231 kB zlib); fast wipes and cuts are the weak spots |
| 4 | **Differentiable Logic CA**: learned Boolean gates | a learned 1,598-gate circuit simplified to **9 gates** that equal the Greenberg–Hastings rule on all 262,144 inputs; excitable waves learned from the visible channel alone | synthetic, known rules. On real Bad Apple, circuits keep motion for ~8 frames inside one motion phase, then collapse into blobs (`docs/path-d/`) |
| 5 | **Context codebook**: "maybe a few shared rules plus a mode switch" | a clean negative: a global codebook of local laws recovers 18 % of the per-window gap on new scenes and matches a plain 49-way shift | selection from the past fails; rollouts make no motion (`docs/context-s0/`) |
| 6 | **Velocity-carrying CA**: cells carry direction and speed; regional nudges | with oracle nudges every 4 frames it follows real motion far better than standing still | about as good as moving image blocks with the same control information; one nudge persists usefully only for translation (`docs/path-v/`) |
| 7 | **External video**: *Fantasmagorie* (1908, public domain) through the same pipeline | see the Fantasmagorie section of this campaign's report (filled from `runs/gen/summary.json`) | one model per video; no zero-shot transfer between the two videos |
| 8 | **Shape-aware CA**: a coarse lattice of control nodes steering local transport | on synthetic deformations it keeps thin rings and bending arms connected where pixel shifts tear them | needs oracle controls; on real windows it is only slightly better than the velocity CA and well behind keyframes |

**What we learned, in one sentence:** local rules can *replay* this animation when they are given hidden memory and a
schedule, but we found no small local law that *predicts* it. Every measurement says the next frame is not determined
by the local picture, so the information has to come from somewhere (hidden state, a schedule, or control signals),
and we now count exactly how much.
