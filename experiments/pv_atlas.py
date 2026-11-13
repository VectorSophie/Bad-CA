"""Path V difficulty atlas: descriptive metrics and a metric-based tag for every 16-frame window of the 48x36 binary
video (windows [16 i, 16 i + 16), i = 0..409; the video itself is unchanged). Tags are measured, not semantic: a tag
"translation" means a single integer shift explains most one-step change, not that the scene shows a walking figure.

Per window (15 transitions): change mass, max change, cut / wipe / fade flags (ca.segment.signals, wipe rule of
FINDINGS §10), foreground (= not the border colour) area change and monotonicity, centroid displacement, component
births / deaths, IoU first -> last, and how much of the one-step change is explained (1 - err / hold) by
  g: the best global integer shift per transition (dy, dx in [-2, 2]),
  r: the best shift per region of a 3x3 grid, per transition,
  m: the better of one 4-neighbour dilation / erosion per transition.
g, r, m are the best over lags k in {1, 2, 4} (frame t -> t+k, shifts dy, dx in [-4, 4]), so that sub-pixel motion
(a boundary that moves 1 px every 3 frames) is visible to an integer shift.
Tag rules, applied in order (v2; fixed before any method is scored against the atlas):
  cut_wipe          a cut, a fade edge or a wipe run (>= 3 frames with change >= 0.08) inside the window
  static            mean change < 0.003
  appear_disappear  >= 2 component births or >= 2 deaths (components >= 4 px, no overlap with the previous frame)
  translation       g >= 0.5 and the modal non-zero shift (at the best lag) holds in >= 50 % of moving transitions
  multi_motion      r >= 0.5 and r - g >= 0.2
  growth            |area change| >= 5 %, the area changes monotonically in >= 70 % of moving transitions, centroid
                    moves < 3 px (m, the dilate/erode share, is reported but not used: whole-silhouette dilation
                    almost never matches local growth, so v2 with m >= 0.3 tagged 0 windows)
  major_transform   foreground IoU first -> last < 0.4
  deformation       everything else that moves
Revision note: v1 (one-step, lag 1 only, shifts in [-2, 2], IoU tested before translation) tagged 1 window of 410 as
translation and put fast rigid moves (e.g. the bar at 2336) under major_transform; seen on atlas_examples.png and the
metric distributions, before any method was scored. v2 replaces it (and v2's growth rule was replaced once, see above).
usage: pv_atlas.py  -> runs/path_v/atlas.json, runs/path_v/atlas_examples.png"""
import json, os, sys
import numpy as np
from scipy import ndimage
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path[:0] = [ROOT, os.path.dirname(os.path.abspath(__file__))]
OUT = os.path.join(ROOT, "runs", "path_v"); L = 16
TAGS = ("cut_wipe", "static", "appear_disappear", "translation", "multi_motion", "growth", "major_transform", "deformation")
ST8 = ndimage.generate_binary_structure(2, 2); CROSS = ndimage.generate_binary_structure(2, 1)


def video():
    from exp45_c12 import target
    return target("0:6572").astype(np.uint8)


def shift(v, dy, dx):
    H, W = v.shape; o = np.zeros_like(v)
    o[max(dy, 0):H + min(dy, 0), max(dx, 0):W + min(dx, 0)] = v[max(-dy, 0):H + min(-dy, 0), max(-dx, 0):W + min(-dx, 0)]
    return o


SH = [(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3)]


def bg_of(f):
    b = np.concatenate([f[0], f[-1], f[:, 0], f[:, -1]]); return int(b.mean() >= 0.5)


def comps(m):
    lab, n = ndimage.label(m, ST8); sz = np.bincount(lab.ravel())
    return lab, [i for i in range(1, n + 1) if sz[i] >= 4]


SH4 = [(dy, dx) for dy in range(-4, 5) for dx in range(-4, 5)]


def explained(F, k, bg):
    """Fraction of lag-k change explained by: best global shift, best shift per 3x3 region, best of dilate/erode;
    plus the modal non-zero global shift share over moving transitions."""
    ys, xs = np.linspace(0, 36, 4).astype(int), np.linspace(0, 48, 4).astype(int); H = G = R = M = 0; best = []
    for t in range(len(F) - k):
        a, b = F[t], F[t + k]; h = (a != b).sum()
        if h == 0: continue
        e = [(shift(a, *d) != b) for d in SH4]; es = [x.sum() for x in e]; j = int(np.argmin(es)); best.append(SH4[j])
        H += h; G += es[j]
        R += sum(min(x[ys[i]:ys[i + 1], xs[q]:xs[q + 1]].sum() for x in e) for i in range(3) for q in range(3))
        fa, fb = a != bg, b != bg; M += min(h, (ndimage.binary_dilation(fa, CROSS, k) != fb).sum(), (ndimage.binary_erosion(fa, CROSS, k, border_value=0) != fb).sum())
    if not H: return 0.0, 0.0, 0.0, 0.0
    nz = [d for d in best if d != (0, 0)]; modal = max(nz.count(d) for d in set(nz)) / len(best) if nz else 0.0
    return 1 - G / H, 1 - R / H, 1 - M / H, modal


def window_metrics(F):
    T = len(F) - 1; bg = bg_of(F[0]); fg = F != bg; ch = (F[1:] != F[:-1]).mean((1, 2)); mv = ch > 0
    ex = {k: explained(F, k, bg) for k in (1, 2, 4)}; kg = max(ex, key=lambda k: ex[k][0])
    area = fg.sum((1, 2)).astype(float); da = np.diff(area); s = np.sign(da[mv]) if mv.any() else np.zeros(0)
    mono = float(max((s > 0).mean(), (s < 0).mean())) if len(s) else 0.0
    cen = [np.array(np.nonzero(m)).mean(1) if m.any() else None for m in fg]
    disp = float(np.linalg.norm(cen[-1] - cen[0])) if cen[0] is not None and cen[-1] is not None else None
    births = deaths = 0
    for t in range(T):
        la, ca = comps(fg[t]); lb, cb = comps(fg[t + 1])
        births += sum(1 for c in cb if not (fg[t] & (lb == c)).any()); deaths += sum(1 for c in ca if not (fg[t + 1] & (la == c)).any())
    inter, uni = (fg[0] & fg[-1]).sum(), (fg[0] | fg[-1]).sum()
    return dict(change=float(ch.mean()), max_change=float(ch.max()), g=float(ex[kg][0]), g_lag=kg, r=float(max(e[1] for e in ex.values())),
                m=float(max(e[2] for e in ex.values())), modal_shift=float(ex[kg][3]),
                area0=int(area[0]), area_change=float((area[-1] - area[0]) / max(area[0], 1)), area_monotone=mono, centroid_disp=disp,
                births=births, deaths=deaths, iou_first_last=float(inter / uni) if uni else 1.0, bg=bg)


def tag(m, cut, wipe, fade):
    if cut or wipe or fade: return "cut_wipe"
    if m["change"] < 0.003: return "static"
    if m["births"] >= 2 or m["deaths"] >= 2: return "appear_disappear"
    if m["g"] >= 0.5 and m["modal_shift"] >= 0.5: return "translation"
    if m["r"] >= 0.5 and m["r"] - m["g"] >= 0.2: return "multi_motion"
    if abs(m["area_change"]) >= 0.05 and m["area_monotone"] >= 0.7 and (m["centroid_disp"] or 0) < 3: return "growth"
    if m["iou_first_last"] < 0.4: return "major_transform"
    return "deformation"


def build(V):
    from ca import segment
    sig = segment.signals(V); ch = sig["change"]; wipe = np.zeros(len(V), bool)
    hi = ch >= 0.08
    for t in range(len(V)):
        if hi[t] and ((t + 2 < len(V) and hi[t + 1] and hi[t + 2]) or (t >= 1 and t + 1 < len(V) and hi[t - 1] and hi[t + 1]) or (t >= 2 and hi[t - 1] and hi[t - 2])): wipe[t] = True
    fade = np.zeros(len(V), bool); fade[1:] = sig["uniform"][1:] != sig["uniform"][:-1]
    rows = []
    for i in range(len(V) // L):
        s = i * L; F = V[s:s + L]; m = window_metrics(F)
        c, w, fd = bool(sig["cut"][s + 1:s + L].any()), bool(wipe[s + 1:s + L].any()), bool(fade[s + 1:s + L].any())
        rows.append(dict(start=s, end=s + L, time_s=round(s / 30, 2), cut=c, wipe=w, fade=fd, uniform_frames=int(sig["uniform"][s:s + L].sum()), tag=tag(m, c, w, fd), **m))
    return rows


def tag_of(rows, a, b):
    """Atlas tags overlapping frames [a, b) (for re-stratifying windows that do not align with the atlas grid)."""
    return sorted({r["tag"] for r in rows if r["start"] < b and r["end"] > a})


def frame_tags(rows, n=6572):
    out = ["static"] * n
    for r in rows:
        for t in range(r["start"], r["end"]): out[t] = r["tag"]
    return out


def examples(V, rows, path, per=4, every=3):
    from PIL import Image, ImageDraw
    rng = np.random.default_rng(0); sc = 2; W_, H_ = 48 * sc + 2, 36 * sc + 2; cols = len(range(0, L, every))
    img = Image.new("RGB", (120 + per * (cols * W_ + 12), len(TAGS) * (H_ + 14) + 4), (246, 245, 242)); d = ImageDraw.Draw(img)
    for ti, t in enumerate(TAGS):
        cand = [r for r in rows if r["tag"] == t]; pick = [cand[j] for j in sorted(rng.choice(len(cand), min(per, len(cand)), replace=False))] if cand else []
        y0 = 4 + ti * (H_ + 14); d.text((4, y0 + 30), f"{t} ({len(cand)})", fill=(20, 20, 20))
        for pi, r in enumerate(pick):
            x0 = 120 + pi * (cols * W_ + 12); d.text((x0, y0 - 2), f'{r["start"]}', fill=(90, 90, 90))
            for ci, k in enumerate(range(0, L, every)):
                f = V[r["start"] + k] * 255; im = Image.fromarray(np.repeat(np.repeat(f, sc, 0), sc, 1).astype(np.uint8)).convert("RGB")
                img.paste(im, (x0 + ci * W_, y0 + 10))
    img.save(path)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); V = video(); rows = build(V)
    from collections import Counter
    cnt = Counter(r["tag"] for r in rows)
    json.dump(dict(window_frames=L, tags=TAGS, rules=__doc__, counts=cnt, windows=rows), open(os.path.join(OUT, "atlas.json"), "w"), indent=0)
    examples(V, rows, os.path.join(OUT, "atlas_examples.png"))
    print(dict(cnt))
