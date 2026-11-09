# Video-to-CA research and prototype build prompt

Build a local research tool that automatically searches for compact cellular-automaton programs reproducing uploaded videos. Bad Apple is the main demonstration, but the pipeline must accept other videos, including colour footage.

Start with repository research, small reproducible experiments, and a working vertical slice. Proceed through the stages below without stopping after a plan. Do not deploy or expose the service publicly.

## Goal and claims

Given a video, produce:
- A starting cellular state.
- One fixed, spatially shared local update rule.
- Explicit boundary and playback settings.
- An independently executable export.

Playback must evolve autonomously from the starting state. No target-frame access, frame injection, per-scene resets, external timestamps supplied to cells, or rule switching during playback.

Primary ambition: compact, explicit discrete rules. Training may use numerical relaxations, but evaluate the hardened discrete artifact separately.

Distinguish:
- Reconstructing a known CA from its observations.
- Synthesizing a CA for an arbitrary video.
- Memorizing video through coordinates/time.
- Encoding video into a large seed and running a generic player.

Do not claim that success on one establishes the others. Report “smallest found under this search budget,” not global minimality.

Arbitrary-seed convergence is a separate possible future track. It is not required for this prototype.

## Research sources

Read these papers and inspect their implementations:

1. Differentiable Logic Cellular Automata: From Game of Life to Pattern Generation — Miotti et al., 2025; arXiv 2506.04912.
2. Deep Differentiable Logic Gate Networks — Petersen et al., 2022; arXiv 2210.08277.
3. Learning spatio-temporal patterns with Neural Cellular Automata — Richardson et al., 2024; DOI 10.1371/journal.pcbi.1011589.
4. Reconstructing cellular automata rules from observations at nonconsecutive times — Elser, 2021; arXiv 2012.02179.

Secondary references as needed: Growing Neural Cellular Automata; DyNCA; CAX; Convolutional Differentiable Logic Gate Networks.

Relevant repositories:
- google-research/self-organising-systems — notebooks/diffLogic_CA.ipynb
- maxencefaldor/cax — Conway, DiffLogic, evolution examples
- Felix-Petersen/difflogic — underlying logic-gate learning
- rplsmn/logicars — Rust reproduction and hard-circuit export
- AlexDR1998/NCA — image-sequence learning
- AlexDR1998/Differentiable-Patterning — newer training infrastructure
- IVRL/DyNCA — motion objectives and video training
- xcontcom/evolving-cellular-automata — discrete-rule evolution and visualization
- Pratik-3706/BadApple_nn — preprocessing, progress previews, browser streaming
- mspitzna/NCAtorch — modular NCA and interactive application
- williamgilpin/convoca — rule inference baseline
- cnlohr/badderapple and Timendus/chip-8-bad-apple — compactness/compression comparisons

Inspect actual code, dependencies, licenses, and reproducibility. README claims are not verified results. Record commits used. Reuse suitable implementations instead of rebuilding everything; avoid combining several incompatible research frameworks.

## Stage 1: feasibility and architecture

Inspect the machine and available accelerators. Run a small JAX/CAX compatibility check. Prefer Python/JAX for search, FastAPI for local job control, and TypeScript/React for the interface. PyTorch is an alternative if evidence favours it. Keep one primary training framework.

Define a shared search/evaluation contract before implementing several methods:
- Input target and preprocessing identity.
- Rule family, initialization constraints, and compute budget.
- Candidate circuit and seed.
- Autonomous rollout evaluator.
- Complexity and quality metrics.
- Checkpoint/resume and export format.

Measure rule bytes/gates, seed bytes, cell-state bits, neighbourhood radius, simulation steps, wall time, and peak memory separately. Include model wiring and constants in complexity accounting.

Do not spend substantial compute before timing short pilot runs and establishing an explicit configurable budget.

## Stage 2: data pipeline

Accept ordinary colour video. Inspect metadata and timestamps, handle orientation, preserve aspect ratio, and support:
- Grayscale conversion.
- Adjustable black-and-white threshold.
- Resolution selection.
- Temporal sampling and clip selection.
- Immediate first-frame preview.

Use existing media libraries. Evaluate FFmpeg with an appropriate Python interface, TorchCodec where relevant, and suitable chunked-array storage such as Zarr or memory-mapped NumPy. Choose based on a small working comparison, not dependency accumulation.

Generate resolution/settings variants on demand and cache them. Cache keys must include source identity, crop/aspect policy, resolution, sampling timestamps, threshold, and preprocessing version. Separate grayscale caches from derived binary targets when useful.

Read and generate time chunks lazily. Do not decode or duplicate every variant eagerly. Preserve exact frame-to-time mappings.

Bad Apple sources to inspect:
- C-mykk/Bad-Apple-Starter: BAframedata.zip and BAframesAsText.txt.
- Felixoofed/badapple-frames: frames.zip, video, extraction script.
- bleov/bad-apple-frames: higher-resolution frame collection.
- SirPigari/bad-apple-api: supplementary source; documented coverage is only 42 seconds.

Validate downloaded data: orientation, polarity, dimensions, ordering, frame count, timestamps, missing/duplicate frames, and coverage. Do not assume different collections are temporally identical.

## Stage 3: experiments before scaling

Establish:
1. Hand-authored Conway correctness against a trusted simulator.
2. Rule recovery on a known CA trajectory.
3. A tiny moving-shape sequence.
4. A short nontrivial animation with changes beyond translation.
5. Several short Bad Apple clips with different motion/scene characteristics.

First analyze conflicting local transitions. Implement neighbour-count and full-neighbourhood diagnostics. Clearly limit any impossibility claim to the stated family, state representation, and generations-per-frame assumption.

Start with visible state from the first processed frame. Hidden state, if introduced, should initially use a declared constant initialization. Any learned seed is a separate labelled experiment, with its information cost counted.

A uniform first frame on a translation-invariant deterministic domain cannot spontaneously form spatial detail. Detect this condition. State any boundary, anchor, or randomness allowance explicitly.

Try a small portfolio:
- Birth/survival rules and directional lookup-table baselines.
- DiffLogic circuits, including a verified Conway-initialized variant if practical.
- Discrete mutation/refinement of hardened circuits.
- Circuit pruning and simplification.

Do not presume first-frame-only, low-memory exact reconstruction is feasible for arbitrary clips.

## Experimental autonomy

After reproducing baselines, propose and run bounded creative experiments. Each needs:
- Hypothesis and mechanism.
- Comparable baseline.
- Compute budget and stopping rule.
- Result, including negative results.
- Decision to retain, modify, or discard.

Candidate ideas:
- Compiler techniques: constant propagation, dead gates, common subexpressions, circuit rewrites.
- Program synthesis: counterexample-guided expansion and short-expression search.
- Coarse-to-fine spatial training.
- Progressive temporal horizons and motion/difficulty-based curricula.
- Gradient search followed by hardening, mutation, and simplification.
- Diverse populations and random restarts.
- Successive halving or adaptive allocation across runs.
- Checkpointing and memory-efficient differentiation through rollouts.

Training windows may be sampled or reordered. Final evaluation must replay the original sequence uninterrupted from its seed. No concatenating independently initialized clip reconstructions.

Compare methods at equal total accelerator budget. Multiple simultaneous runs are orchestration, not automatically an algorithmic improvement. Log total compute, including unsuccessful runs.

Keep a quality–complexity frontier. Track the soft/hard performance gap. Prevent background-heavy pixel accuracy from concealing failed silhouettes: include foreground/edge and temporal measurements alongside overall reconstruction.

## Stage 4: explicit design extraction from larperCord

Reference repository: RshaCuDeVidro/larperCord.

Inspect:
- site/landing.html
- docs/assets/site/site.css
- docs/assets/site/site.js
- docs/assets/site/morph.js
- docs/assets/site/playground.js
- docs/assets/fonts/

Do not merely write “inspired by larperCord” and improvise a generic dashboard.

Extract and document concrete source mechanisms:
- Departure Mono pixel display typography.
- Syntax-like section tags and framing.
- Pixel/cell background treatment.
- Pointer-responsive depth and motion.
- Wipe/glitch transitions.
- Morphing demonstration panels.
- Editor/live-preview relationship.
- Theme variables, spacing, borders, and reduced-motion behaviour.

Create a design-transfer table: source file/selector or function → mechanism → adaptation in our tool → screenshot evidence.

Implement at least four recognizable adapted mechanisms, including typography, framing, one transition system, and one live demonstration interaction. Adjust composition, colour system, wording, and application behaviour for our project. Do not reproduce Discord branding, mascots, page copy, screenshots, or the landing-page layout wholesale.

Inspect licenses before copying code or assets. Prefer original implementations of observed design mechanisms where direct reuse is unsuitable; preserve required attribution and notices for actual reuse.

The experiment workspace should include:
- Video upload and preprocessing preview.
- Synchronized reference and CA canvases.
- Inspectable rule list/circuit and highlighted candidate changes.
- Search progress and competing-run inspection.
- Quality-versus-complexity history.
- Final replay, seed restart, and export.

Each previewed candidate runs with fixed rules. Candidate updates restart or clearly switch the preview; never imply that changing rules mid-rollout demonstrates an autonomous solution.

Use real computation. Sample data and prerecorded progress must be visibly labelled. Browser playback should independently execute the exported hard circuit.

Validate the UI in a browser using actual screenshots and interactions. Compare the result against the design-transfer table.

## Security staging

Local-only for now. Use generated filenames, isolated job directories, safe subprocess argument lists, bounded media/job resources, cancellation, and cleanup. Export rules as validated declarative circuits, never executable uploaded source.

Record predeployment requirements without letting them dominate the prototype: content-based MIME detection, extension/decoder cross-checks, isolated decoding workers, upload/storage/concurrency limits, authentication, and request limits.

## Deliverables and execution

Create concise repository documentation for:
- Research findings and source/commit inventory.
- Architecture and artifact schema.
- Experiment configurations, results, and budget.
- Design extraction and browser evidence.
- Known limitations and deferred security work.

Proceed through research, baseline reproduction, a small real search experiment, and a functional vertical slice. Do not stop after producing documentation or a mock interface.

The first reviewable milestone is:
upload or select a short clip → preprocess at a chosen resolution → run a real bounded rule search → inspect an actual candidate → replay its exported discrete rules independently → show measured quality and complexity.

Report what actually worked, what failed, what was reused, and which experiment should run next. A poor reconstruction is an acceptable early result; a polished simulation of nonexistent progress is not.
