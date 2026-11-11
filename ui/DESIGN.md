# Design notes

Reference: the larperCord site (github.com/RshaCuDeVidro/larperCord). Its code and CSS are GPL-3.0 and were **not**
copied; only its fonts (SIL OFL 1.1, licence files in `static/fonts/`) are reused. The ideas below were reimplemented.

| mechanism there | what we took | how it changed here |
|---|---|---|
| Pixel-font display heading (Departure Mono, line-height ~1.16) with a tiny mono "code tag" kicker | the scale contrast: huge pixel heading, small DM Mono kicker, keyword in the accent | the kicker is the real update equation `update { s += Wu·relu(Wp ∗ s + bp) + bu } // ×S per frame, from the first frame`; heading "Bad CA!!" next to the pixel mark |
| Box-drawing section tags, `[ label ]` buttons in the accent | `┌─ ` prefix and bracketed buttons | panels are named after what they hold (`┌─ player.compare`, `┌─ model.architecture`, `┌─ run.new`, `┌─ job <run>`); disabled buttons fade to the rule colour |
| Header canvas of sparse 4 px squares on a 28 px grid, lit near the pointer, alpha ×0.88 every 48 ms, stops when hidden | the same grid/decay numbers | amber squares, header only, the timer stops as soon as nothing is lit, when the tab is hidden, and **whenever the video starts playing**; off entirely under `prefers-reduced-motion` |
| Chromatic split heading (2 px opposite text-shadows) | the split | the two colours are our diff colours, FN red left and FP blue right, the same ones the error panel uses |
| Panel entry "rise", toast with a wipe | rise (opacity 0, translateY 8 px) on panel entry; toast wiped in with `clip-path` in 8 steps for job started / finished / failed / died | nothing animates inside or near the video canvases after entry |

Own identity: near-black ink `#0c0c0b`, paper `#ece8df`, one accent (phosphor amber `#ffb43a`), square corners, 1 px rules.
Pixels are drawn at native size and scaled by integer CSS zoom with `image-rendering: pixelated`.

Honesty rules baked into the UI: the "model" panel shows the architecture, parameter count formula, an SVG of the local
update and the training curve / evaluation points; it never lists "rules". The continuous view is labelled as a raw state
value thresholded at 0.5, not a probability. The player states "autonomous playback from the first frame (hidden 0), no
target injection", and the original frames are only drawn in the comparison and error panels.

C12 additions reuse the same tokens and components (box-drawing panel tags `┌─ model.rules`, `┌─ cell.inspector`, bracket
buttons, 1 px rules, DM Mono tables). The phase timeline is a flat bar of 1 px-ruled segments; the current phase and the
"now" line use the accent, nothing animates. The selected cell is drawn in the accent on the CA panel. Diff colours keep
their meaning (blue = added, red = removed, amber = changed). NCA phases are summarised as numbers only, never as rules.

C13 restructure (same tokens, not a redesign): a compact masthead with three bracket tabs (`[ demo ]`, `[ rule lab ]`,
`[ research ]`; the selected one is filled amber). The demo is a two-column screen (CA video left, `┌─ rule.in_effect`
right, single column under 960 px) with the timeline full-width below. The rule panel is the one place that uses the diff
blue/red for another meaning: blue = fixed structure of the update, red = phase parameters. Colour is never the only cue:
the legend says it in words, and parameters are underlined monospace badges (buttons that open the values). Matrix
heatmaps use amber (positive) / paper (negative) on ink, brightness sqrt(|v|/max), so they do not reuse blue/red. The
timeline is ink segments with thin boundaries, a dim "computed" bar and the amber playhead; labels sit in HTML below the
SVG so they cannot collide. Comparison panels live behind `[ tools ]`, off by default.

Polish pass (same tokens): the pixel mark (80×60, `image-rendering: pixelated`, 1 px rule border) sits left of the heading.
A light theme reuses every role under `prefers-color-scheme: light` (ink `#f3f0e8`, paper `#141412`, accent darkened to
`#8a5300`, diff red/blue to `#b8261a` / `#1d5fb8` so text stays ≥ 4.8:1); dark stays the default. Timeline fills are
variables (`--seg-cur`, `--seg-alt`). Disclosures (`Model details`, the research archive) use the amber summary style.
