"""Bad CA!! documentation figures, drawn by code (PIL) from the actual implementation and results; English labels.
Every text box is registered and checked: no two labels may overlap and none may leave the canvas (AssertionError).
Outputs (assets/figures/):
  pipeline.png          video -> preprocess -> phases -> training -> export -> independent replay (+ original only for scoring)
  cell_update.png       one cell: 3x3 neighbourhood of 16 channels -> update equation -> next state (from ca/nca.py step_np)
  phase_switch.png      rules switch at a phase boundary, cell state carries on
  approach_tree.png     approaches tried, grouped by representation, with search methods / time structure as annotations
  approach_tree_full.png  the same tree with result notes and record pointers
  l240_fixed_vs_phased.png  matched comparison on frames 1760-1999 (same steps, 2 seeds, export replay)
  featured_error.png    error per frame of the featured export (numpy replay vs original) with phase boundaries
usage: .venv/bin/python tools/make_figures.py"""
import json, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); OUT = os.path.join(ROOT, "assets", "figures"); os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "experiments"))
REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"; BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"; MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
INK, MUTE, BLUE, RED, GREEN, AMB, BG, BOX = (25, 25, 30), (110, 110, 120), (40, 100, 200), (205, 55, 45), (30, 130, 70), (200, 130, 0), (255, 255, 255), (244, 245, 248)


class Fig:
    def __init__(self, w, h, title=None):
        self.im = Image.new("RGB", (w, h), BG); self.d = ImageDraw.Draw(self.im); self.boxes = []; self.w, self.h = w, h
        if title: self.text((w // 2, 24), title, 26, bold=True, anchor="mm")

    def f(self, sz, bold=False, mono=False): return ImageFont.truetype(MONO if mono else (BOLD if bold else REG), sz)

    def text(self, xy, s, sz=18, col=INK, bold=False, mono=False, anchor="la", check=True):
        f = self.f(sz, bold, mono); bb = self.d.textbbox(xy, s, font=f, anchor=anchor)
        assert bb[0] >= 2 and bb[1] >= 2 and bb[2] <= self.w - 2 and bb[3] <= self.h - 2, f"text leaves canvas: {s!r} {bb}"
        if check:
            for o in self.boxes:
                assert bb[2] <= o[0] or bb[0] >= o[2] or bb[3] <= o[1] or bb[1] >= o[3], f"text overlap: {s!r} vs {o[4]!r}"
            self.boxes.append((*bb, s))
        self.d.text(xy, s, font=f, fill=col, anchor=anchor); return bb

    def lines(self, cx, y, rows, sz=17, gap=6, anchor="ma"):
        for r in rows:
            s, col, bold = (r + (INK, False))[:3] if isinstance(r, tuple) else (r, INK, False)
            bb = self.text((cx, y), s, sz, col, bold, anchor=anchor); y = bb[3] + gap
        return y

    def box(self, x0, y0, x1, y1, fill=BOX, outline=INK, w=2): self.d.rectangle((x0, y0, x1, y1), fill=fill, outline=outline, width=w)

    def arrow(self, x0, y0, x1, y1, col=INK, w=3):
        self.d.line((x0, y0, x1, y1), fill=col, width=w); v = np.array([x1 - x0, y1 - y0], float); v /= np.linalg.norm(v) + 1e-9; n = np.array([-v[1], v[0]])
        p = np.array([x1, y1]); self.d.polygon([tuple(p), tuple(p - 12 * v + 6 * n), tuple(p - 12 * v - 6 * n)], fill=col)

    def save(self, name): self.im.save(os.path.join(OUT, name)); print("wrote", name, self.im.size)


def pipeline():
    F = Fig(1500, 330, "From a video to an exported, independently replayed CA")
    steps = [("Video", ["any local file", "(Bad Apple!! here)"]), ("Preprocess", ["luma, box resize", "48x36, threshold 128"]),
             ("Phases", ["external schedule", "of frame ranges"]), ("Train rules", ["one rule per phase", "state carried, noise"]),
             ("Export", ["one .ncab file:", "rules, schedule, init"]), ("Replay", ["numpy, Node, browser", "from the file only"])]
    bw, gap, y0 = 200, 50, 80; x = 30
    for i, (t, sub) in enumerate(steps):
        F.box(x, y0, x + bw, y0 + 120, fill=(235, 242, 255) if i in (3, 4, 5) else BOX)
        F.text((x + bw // 2, y0 + 16), t, 20, bold=True, anchor="ma"); F.lines(x + bw // 2, y0 + 52, [(sub[0], MUTE), (sub[1], MUTE)], 15)
        if i < len(steps) - 1: F.arrow(x + bw + 4, y0 + 60, x + bw + gap - 4, y0 + 60)
        x += bw + gap
    F.d.line((30 + bw // 2, y0 + 124, 30 + bw // 2, 270), fill=MUTE, width=2); F.arrow(30 + bw // 2, 270, 5 * (bw + gap) + 30 - 10, 270, MUTE, 2)
    F.text((30 + 5 * (bw + gap) + bw // 2, 262), "score only", 15, MUTE, anchor="ma")
    F.text((330, 278), "the original frames are used for training and for scoring, never fed into the replay", 15, MUTE)
    F.save("pipeline.png")


def cell_update():
    F = Fig(1500, 470, "One cell, one CA substep (4 substeps per video frame)")
    x0, y0, c = 40, 90, 34
    for i in range(3):
        for j in range(3): F.box(x0 + j * c, y0 + i * c, x0 + (j + 1) * c, y0 + (i + 1) * c, fill=(225, 232, 245) if (i, j) != (1, 1) else (255, 215, 140), w=1)
    F.lines(x0 + 51, y0 + 112, [("3x3 neighbourhood", INK, True), ("16 channels each:", MUTE), ("1 visible + 15 hidden", MUTE), ("zero outside grid", MUTE)], 15)
    F.arrow(160, 140, 215, 140)
    F.box(220, 80, 690, 200, fill=(240, 245, 255))
    F.text((235, 92), "features (64)", 16, MUTE)
    F.text((235, 122), "p = ReLU(", 20, BLUE, mono=True); F.text((365, 122), "Wp", 20, RED, mono=True, bold=True); F.text((395, 122), " * s +", 20, BLUE, mono=True)
    F.text((475, 122), "bp", 20, RED, mono=True, bold=True); F.text((505, 122), " )", 20, BLUE, mono=True)
    F.text((235, 160), "Wp: 64x16x3x3, bp: 64   (* = 3x3 weighted sum)", 14, MUTE)
    F.arrow(695, 140, 750, 140)
    F.box(755, 80, 1195, 200, fill=(240, 245, 255))
    F.text((770, 92), "residual update (16 channels)", 16, MUTE)
    F.text((770, 122), "s <- s +", 20, BLUE, mono=True); F.text((880, 122), "Wu", 20, RED, mono=True, bold=True); F.text((910, 122), " . p +", 20, BLUE, mono=True)
    F.text((990, 122), "bu", 20, RED, mono=True, bold=True)
    F.text((770, 160), "Wu: 16x64, bu: 16", 14, MUTE)
    F.arrow(1200, 140, 1255, 140)
    F.box(1260, 80, 1470, 200, fill=(235, 250, 238))
    F.lines(1365, 92, [("next state", INK, True), ("pixel = 1 if s0 > 0.5", INK), ("(s0: a state value,", MUTE), ("not a probability)", MUTE)], 15)
    F.text((40, 300), "blue: fixed structure, the same in every phase", 17, BLUE)
    F.text((40, 330), "red: learned numbers of the rule of the current phase (10,320 float32 values per rule)", 17, RED)
    F.text((40, 360), "every cell applies the same rule at the same time (synchronous); the shown frame is after 4 substeps", 17, MUTE)
    F.text((40, 390), "source: ca/nca.py step_np and player/nca_core.mjs (same arithmetic; Node / browser sum in float64)", 15, MUTE)
    F.save("cell_update.png")


def phase_switch():
    F = Fig(1500, 380, "Rules switch at a phase boundary; the cell state carries on")
    y = 110; F.d.line((60, y, 1440, y), fill=INK, width=3)
    for x, lab in ((60, "frame a"), (720, "boundary b"), (1440, "frame c")):
        F.d.line((x, y - 12, x, y + 12), fill=INK, width=3); F.text((x, y + 20), lab, 15, MUTE, anchor="ma")
    F.box(90, 160, 690, 240, fill=(255, 238, 236), outline=RED); F.lines(390, 172, [("rule k-1 (its own 10,320 values)", RED, True), ("used for frames a .. b-1", MUTE)], 16)
    F.box(750, 160, 1410, 240, fill=(255, 238, 236), outline=RED); F.lines(1080, 172, [("rule k (its own 10,320 values)", RED, True), ("used for frames b .. c-1", MUTE)], 16)
    F.box(90, 270, 1410, 320, fill=(236, 245, 255), outline=BLUE)
    F.text((750, 282), "state s (visible + 15 hidden channels per cell) continues unchanged across the boundary: no reset, no injection", 16, BLUE, anchor="ma")
    F.text((60, 62), "the schedule (frame ranges and rule ids) is stored in the export; the switch is instant (no interpolation)", 16, MUTE)
    F.save("phase_switch.png")


TREE = [  # (label, result note, record, status: chosen | tried | partial)
    ("Discrete states, one rule", [
        ("Memoryless / Life-like / LUT", "exact only where 3x3 context is unambiguous", "CHECKPOINT_01", "tried"),
        ("Hidden bits + several updates per frame", "exact short replays become possible", "CHECKPOINT_02", "partial")]),
    ("Searching small discrete rules", [
        ("SAT for exact rules", "exact <= 10 frames; rules larger than recordings", "CHECKPOINT_03-06", "partial"),
        ("Circuit synthesis + shrinking", "small circuits only on synthetic tests", "CHECKPOINT_05", "tried"),
        ("CGP evolution", "coarse shape, no real motion (flicker loophole)", "CHECKPOINT_07-09", "tried"),
        ("DiffLogic (soft logic)", "soft fits; hardened rules much worse", "CHECKPOINT_02, 07", "tried"),
        ("Decision lists (explicit rules)", "exact on known rules; ~0.10 error on Bad Apple", "CHECKPOINT_12", "partial")]),
    ("Continuous state (Neural CA)", [
        ("One NCA rule", "recognisable motion on 56-120 frames", "CHECKPOINT_09-11", "partial"),
        ("Phased NCA (external schedule)", "transitions fixed on 240 frames", "CHECKPOINT_12", "chosen"),
        ("Shared base + low-rank deltas", "about half of the phase gain", "CHECKPOINT_12", "tried"),
        ("Training-only state noise", "long replays reproducible from the file", "CHECKPOINT_12-13", "chosen")]),
]


def tree(full):
    """Top-down, centred: question -> three representation groups -> leaves stacked under each group (connectors never cross text)."""
    W = 1650; cxs = (290, 825, 1360); bw = 500; lh = 100 if full else 74; gap = 16; top = 250
    H = top + max(len(g[1]) for g in TREE) * (lh + gap) + 110
    F = Fig(W, H, "Approaches tried in Bad CA!!" + (" (detailed)" if full else ""))
    F.box(W // 2 - 270, 56, W // 2 + 270, 100, fill=(250, 250, 250)); F.text((W // 2, 78), "Can local update rules replay a video?", 18, bold=True, anchor="mm")
    F.d.line((W // 2, 100, W // 2, 120), fill=INK, width=2); F.d.line((cxs[0], 120, cxs[-1], 120), fill=INK, width=2)
    for cx, (gname, leaves) in zip(cxs, TREE):
        chosen = any(l[3] == "chosen" for l in leaves); F.arrow(cx, 120, cx, 148, INK, 2)
        F.box(cx - bw // 2, 150, cx + bw // 2, 196, fill=(235, 242, 255) if chosen else BOX, outline=BLUE if chosen else INK, w=3 if chosen else 2)
        F.text((cx, 173), gname, 18, bold=True, anchor="mm")
        spine = cx - bw // 2 + 14; F.d.line((spine, 196, spine, top + (len(leaves) - 1) * (lh + gap) + lh // 2), fill=INK, width=2)
        for i, (lab, note, rec, st) in enumerate(leaves):
            y = top + i * (lh + gap); col = {"chosen": GREEN, "partial": AMB, "tried": MUTE}[st]; x0 = spine + 26
            F.arrow(spine, y + lh // 2, x0 - 2, y + lh // 2, INK, 2)
            F.box(x0, y, cx + bw // 2, y + lh, fill=(236, 248, 240) if st == "chosen" else BG, outline=col, w=3 if st == "chosen" else 2)
            F.text((x0 + 12, y + 10), lab, 16, INK, bold=st == "chosen")
            F.text((x0 + 12, y + 38), note, 14, col)
            if full: F.text((x0 + 12, y + 66), f"{ {'chosen': 'used in the demo', 'partial': 'worked in part', 'tried': 'not adopted'}[st] }  ·  record: {rec}", 13, MUTE)
    F.text((40, H - 62), "Representations (columns) and search methods are different axes: SAT, circuit synthesis, CGP and DiffLogic searched discrete rules;", 14, MUTE)
    F.text((40, H - 40), "NCAs are trained by gradient descent. 'not adopted' means not used now, not 'impossible'. Green = path used by the demo; amber = worked in part.", 14, MUTE)
    F.save("approach_tree_full.png" if full else "approach_tree.png")


def bar_l240():
    E = {}
    for f in ("pilot.json", "main.json"): E.update(json.load(open(os.path.join(ROOT, "runs", "c12", "eval", f))))
    arms = [("one rule, 10,320", "pilot/fixed64"), ("one rule, 41,232 (same size as 4 phases)", "main/fixed256_lowlr"),
            ("4 equal phases", "pilot/uniform4"), ("4 automatic phases", "pilot/auto4"), ("6 equal phases", "main/L240_uniform6"), ("6 automatic phases", "main/L240_auto6")]
    F = Fig(1500, 560, "Frames 1760-1999 (one wipe, one cut): error of the exported replay")
    x0, x1, y0 = 560, 1440, 90; vmax = 0.08
    for v in np.arange(0, vmax + 1e-9, 0.02):
        x = x0 + (x1 - x0) * v / vmax; F.d.line((x, y0 - 6, x, y0 + 6 * 58), fill=(225, 225, 230)); F.text((x, y0 + 6 * 58 + 6), f"{v:.2f}", 14, MUTE, anchor="ma")
    for i, (lab, key) in enumerate(arms):
        yy = y0 + 16 + i * 58; vals = [E[k]["all"]["err"] for k in E if f"/{key}/" in k]
        F.text((x0 - 14, yy), lab, 16, anchor="rm")
        for v in vals: F.d.ellipse((x0 + (x1 - x0) * v / vmax - 7, yy - 7, x0 + (x1 - x0) * v / vmax + 7, yy + 7), fill=RED if lab.startswith("one rule") else BLUE)
    F.text((x0, y0 + 6 * 58 + 34), "full-interval pixel error (lower is better), one dot per seed (2 seeds), all trained 12,000 steps, replayed from the export", 14, MUTE)
    F.text((x0, y0 + 6 * 58 + 56), "the 41,232-param single rule needed a 4x lower learning rate (it diverged at the common one); same interval, same scoring", 14, MUTE)
    F.save("l240_fixed_vs_phased.png")


def featured_error():
    m = json.load(open(os.path.join(ROOT, "runs", "featured", "manifest.json")))["featured"]; path = os.path.join(ROOT, m["export"])
    from ca import phase_nca
    from exp45_c12 import target
    hd, E, sched, h0, f0 = phase_nca.load(path); B, _ = phase_nca.rollout_np(E, f0, sched, hd["S"], h0); F_ = target(f"{hd['meta']['src_start']}:{hd['T']}")
    e = (B != F_).mean((1, 2)); T = len(e); w = np.convolve(e, np.ones(15) / 15, mode="same")
    F = Fig(1500, 420, f"{m['display_name']}: error per frame of the exported replay"); x0, x1, y0, y1 = 80, 1460, 70, 320; vmax = max(0.1, float(np.ceil(e.max() * 20) / 20))
    for p in hd["schedule"][1:]: x = x0 + (x1 - x0) * p["start"] / T; F.d.line((x, y0, x, y1), fill=(220, 225, 240))
    for v in np.linspace(0, vmax, 5):
        y = y1 - (y1 - y0) * v / vmax; F.d.line((x0, y, x1, y), fill=(238, 238, 240)); F.text((x0 - 8, y), f"{v:.2f}", 13, MUTE, anchor="rm")
    pts = [(x0 + (x1 - x0) * t / T, y1 - (y1 - y0) * min(e[t], vmax) / vmax) for t in range(T)]; F.d.line(pts, fill=(150, 170, 210), width=1)
    pts = [(x0 + (x1 - x0) * t / T, y1 - (y1 - y0) * min(w[t], vmax) / vmax) for t in range(T)]; F.d.line(pts, fill=BLUE, width=3)
    for s in (0, T // 4, T // 2, 3 * T // 4, T - 1): F.text((x0 + (x1 - x0) * s / T, y1 + 8), f"{s / 30:.0f} s", 13, MUTE, anchor="ma")
    F.text((x0, y1 + 34), f"pixel error per frame vs the binary original (light) and 15-frame mean (blue); vertical lines = {len(hd['schedule']) - 1} phase boundaries; 1 seed;", 14, MUTE)
    F.text((x0, y1 + 56), f"numpy replay of {os.path.basename(path)} ({T} frames, sha {m['sha256'][:12]}), not training-time numbers", 14, MUTE)
    F.save("featured_error.png")


if __name__ == "__main__":
    pipeline(); cell_update(); phase_switch(); tree(False); tree(True); bar_l240(); featured_error()
