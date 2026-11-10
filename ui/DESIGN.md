# Design notes

Reference: the larperCord site (github.com/RshaCuDeVidro/larperCord). Its code and CSS are GPL-3.0 and were **not**
copied; only its fonts (SIL OFL 1.1, licence files in `static/fonts/`) are reused. The ideas below were reimplemented.

| mechanism there | what we took | how it changed here |
|---|---|---|
| Pixel-font display heading (Departure Mono, line-height ~1.16) with a tiny mono "code tag" kicker | the scale contrast: huge pixel heading, small DM Mono kicker, keyword in the accent | the kicker is the real update equation `update { s += Wu·relu(Wp ∗ s) } // ×S per frame, from frame 0 with hidden = 0`; heading "replay bench" |
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
