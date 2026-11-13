"""Track S experiments (CPU). ORACLE control for every method: events every E frames, sequential segments from the
method's own last frame, coordinate descent per control unit (2 sweeps) minimising the segment's pixel mismatches;
the runtime gets the start frame and the schedule only.
Methods and control cost per event:
  block    3x3 regions, integer whole-image shift (dy, dx) in [-2, 2]^2 px/frame          9 x log2 25 = 41.8 bits
  vcaL     Path V velocity CA, 3x3 regions, (vy, vx, g) in [-4, 4]^3                       9 x log2 729 = 85.6 bits
  morph    3x3 regions, {hold, dilate, erode}                                              9 x log2 3 = 14.3 bits
  lsca     S1 level-set CA, 3x3 regions, (g in [-4, 4], smooth on/off)                    9 x log2 18 = 37.5 bits
  ffdca    S2 lattice transport, 3x3 nodes (2x2 cells), (vy, vx) in [-4, 4]^2 per node   9 x log2 81 = 57.1 bits
  ffdwarp  geometric reference with the same lattice and alphabet                          57.1 bits
Synthetic tasks (48x36, white shape on black, 16 frames, rasterised from analytic shapes; the intended deformation is
known): arm (a 2-px arm rotates 60 deg about a joint on a body), ring (an elliptic 1.5-px ring squashes: rx 12->7,
ry 7->12), grow (a disk grows r 5->9 while carrying a 1-px antenna), line (a 1-px segment bends into an arc),
rigid (a 14x6 bar rotates 30 deg while translating 6 px).
usage: shape_s.py synth | real   -> runs/shape/synth.json, real.json, figures"""
import json, math, os, sys, time
from multiprocessing import Pool
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path[:0] = [ROOT, os.path.dirname(os.path.abspath(__file__))]
import pv_vca as V, pv_v0, shape_ca as S
OUT = os.path.join(ROOT, "runs", "shape"); H, W = 36, 48
yy, xx = np.mgrid[0:H, 0:W] + 0.5


def seg(p, q, r):
    d = np.array(q) - np.array(p); t = np.clip(((yy - p[0]) * d[0] + (xx - p[1]) * d[1]) / max(d @ d, 1e-9), 0, 1)
    return np.hypot(yy - p[0] - t * d[0], xx - p[1] - t * d[1]) <= r


def task(name, n=16):
    fr = []
    for k in range(n):
        a = k / (n - 1)
        if name == "arm":
            th = math.radians(60 * a); j = (18, 24); e = (j[0] - 14 * math.sin(th), j[1] + 14 * math.cos(th))
            f = (np.hypot(yy - 22, xx - 18) <= 6.5) | seg(j, e, 1.1)
        elif name == "ring":
            rx, ry = 12 - 5 * a, 7 + 5 * a; r = np.hypot((xx - 24) / rx, (yy - 18) / ry); f = np.abs(r - 1) * min(rx, ry) <= 0.8
        elif name == "grow":
            rr = 5 + 4 * a; f = (np.hypot(yy - 20, xx - 24) <= rr) | seg((20 - rr, 24), (20 - rr - 8, 27), 0.55)
        elif name == "line":
            cur = 0.0001 + 0.09 * a; L = 26; ts = np.linspace(-L / 2, L / 2, 120); px = 24 + np.sin(cur * ts) / cur; py = 26 - (1 - np.cos(cur * ts)) / cur
            f = np.zeros((H, W), bool); f[np.clip(np.round(py - 0.5).astype(int), 0, H - 1), np.clip(np.round(px - 0.5).astype(int), 0, W - 1)] = True
        elif name == "rigid":
            th = math.radians(30 * a); cx, cy = 18 + 6 * a, 18; u = (xx - cx) * math.cos(th) + (yy - cy) * math.sin(th); v = -(xx - cx) * math.sin(th) + (yy - cy) * math.cos(th)
            f = (np.abs(u) <= 7) & (np.abs(v) <= 3)
        fr.append(f.astype(np.uint8))
    return np.stack(fr)


TASKS = ("arm", "ring", "grow", "line", "rigid")
R33 = V.region_map((3, 3)); VEL = [(a, b) for a in range(-4, 5) for b in range(-4, 5)]
METHODS = {
    "block": dict(units=9, opts=[(a, b) for a in range(-2, 3) for b in range(-2, 3)], run=lambda f, n, C: V.block_shift(f, n, {0: C}, R33)),
    "morph": dict(units=9, opts=[(0,), (1,), (2,)], run=lambda f, n, C: S.morph_run(f, n, {0: C}, R33, B=len(C))),
    "lsca": dict(units=9, opts=[(g, s) for g in range(-4, 5) for s in (0, 1)], run=lambda f, n, C: S.lsca_run(f, n, {0: C}, R33, B=len(C))),
    "ffdca": dict(units=9, opts=VEL, run=lambda f, n, C: S.ffdca_run(f, n, {0: C.reshape(len(C), 3, 3, 2)}, (2, 2), B=len(C))),
    "ffdwarp": dict(units=9, opts=VEL, run=lambda f, n, C: S.ffdwarp_run(f, n, {0: C.reshape(len(C), 3, 3, 2)}, (2, 2), B=len(C))),
}


def fit(method, F, every):
    """Oracle sequential greedy for frame-state methods. Returns (pred (n, H, W), schedule, bits)."""
    M = METHODS[method]; n = len(F) - 1; ev = list(range(0, n, every)); cur = F[0]; d = len(M["opts"][0]); codes = np.zeros((M["units"], d), int)
    codes[:] = M["opts"][len(M["opts"]) // 2] if method in ("block", "ffdca", "ffdwarp") else M["opts"][0]
    preds, sched = [], {}
    for k, t0 in enumerate(ev):
        T = (ev[k + 1] if k + 1 < len(ev) else n) - t0; tgt = F[t0 + 1:t0 + 1 + T]
        for _ in range(2):
            for u in range(M["units"]):
                C = np.repeat(codes[None], len(M["opts"]), 0); C[:, u] = M["opts"]
                e = (M["run"](cur, T, C) != tgt[None]).sum((1, 2, 3)); codes = C[int(np.argmin(e))]
        fr = M["run"](cur, T, codes[None])[0]; preds.append(fr); cur = fr[-1]; sched[t0] = codes.tolist()
    return np.concatenate(preds), sched, len(ev) * M["units"] * math.log2(len(M["opts"]))


def score(P, F):
    import pd_vis
    m = pv_v0.metrics(P, F); m.pop("err_t"); m.pop("iou_t")
    th = [pd_vis.frame_shape(p, f)["thin_recall"] for p, f in zip(P, F[1:])]; th = [x for x in th if x is not None]
    m["thin_recall"] = float(np.mean(th)) if th else None; return m


def job(a):
    name, method, F, every = a; t0 = time.process_time()
    if method == "hold": P, bits = np.repeat(F[:1], len(F) - 1, 0), 0.0
    elif method == "vcaL": P, sc = pv_v0.fit_vca(F, (3, 3), every); bits = len(sc) * 9 * math.log2(729)
    elif method.startswith("keyframe"): P, bits = pv_v0.keyframe(F, int(method[8:]))
    else: P, _, bits = fit(method, F, every)
    return dict(task=name, method=method, every=every, bits=bits, cpu_s=time.process_time() - t0, **score(P, F)), P


def synth():
    os.makedirs(OUT, exist_ok=True); J = []
    for t in TASKS:
        F = task(t)
        J += [(t, "hold", F, None)] + [(t, m, F, 4) for m in ("block", "vcaL", "morph", "lsca", "ffdca", "ffdwarp")]
    with Pool(len(J)) as p: res = p.map(job, J)
    rows = [r for r, _ in res]; json.dump(rows, open(os.path.join(OUT, "synth.json"), "w"), default=float, indent=0)
    np.savez_compressed(os.path.join(OUT, "synth_preds.npz"), **{f'{r["task"]}|{r["method"]}': np.packbits(P) for r, P in res})
    for t in TASKS:
        print(t); [print(f'   {r["method"]:8s} bits {r["bits"]:5.0f} err {r["err"]:.3f} iou {r["iou"]:.2f} bf1 {r["bf1"]:.2f} f1 {r["f1"]:.2f} thin {r["thin_recall"] if r["thin_recall"] is None else round(r["thin_recall"], 2)} '
                     f'cc_excess {r["cc_excess"]:+.2f} holes {r["hole_excess"]:+.2f}') for r in rows if r["task"] == t]
    sheet(res)


def sheet(res, path=None, scale=3):
    """Rows per task: truth, then each method; columns every 3 frames."""
    from PIL import Image, ImageDraw
    meths = ["truth", "block", "vcaL", "morph", "lsca", "ffdca", "ffdwarp"]; cols = list(range(0, 16, 3)); cw, ch = W * scale + 2, H * scale + 2
    img = Image.new("RGB", (70 + len(TASKS) * (len(cols) * cw + 10), len(meths) * ch + 20), (246, 245, 242)); d = ImageDraw.Draw(img)
    P = {(r["task"], r["method"]): p for r, p in res}
    for ti, t in enumerate(TASKS):
        F = task(t); x0 = 70 + ti * (len(cols) * cw + 10); d.text((x0, 2), t, fill=(0, 0, 0))
        for mi, m in enumerate(meths):
            seq = F if m == "truth" else np.concatenate([F[:1], P[(t, m)]])
            if ti == 0: d.text((2, 16 + mi * ch + ch // 2 - 5), m, fill=(0, 0, 0))
            for ci, k in enumerate(cols):
                im = Image.fromarray((np.repeat(np.repeat(seq[k], scale, 0), scale, 1) * 255).astype(np.uint8)).convert("RGB"); img.paste(im, (x0 + ci * cw, 16 + mi * ch))
    img.save(path or os.path.join(OUT, "fig_synth.png"))


def real():
    """Deformation-heavy real windows, 32 frames, oracle events every 8: Bad Apple V0 test clips tagged deformation /
    major_transform, and Fantasmagorie windows tagged the same way by the Path V atlas rules (first 12, fixed order)."""
    import pv_atlas
    from ca import datasets
    os.makedirs(OUT, exist_ok=True); BA = pv_v0.video(); FA = datasets.frames("fantasmagorie")
    A = pv_atlas.build(FA); json.dump(dict(rules="pv_atlas v2 applied unchanged to the Fantasmagorie lines_tm3 film", windows=A), open(os.path.join(OUT, "atlas_fantasmagorie.json"), "w"), indent=0)
    from ca import segment
    cutF = segment.signals(FA)["cut"]
    fa_clips = [w["start"] for w in A if w["tag"] in ("deformation", "major_transform") and w["start"] + 32 <= len(FA) and not cutF[w["start"] + 1:w["start"] + 32].any()][:12]
    ba_clips = [c["start"] for c in json.load(open(pv_v0.MAN))["clips"] if c["split"] == "test" and c["cat"] in ("deformation", "major_transform")]
    J = []
    for ds, X, clips in (("bad_apple", BA, ba_clips), ("fantasmagorie", FA, fa_clips)):
        for s in clips:
            F = X[s:s + 32]
            J += [(f"{ds}:{s}", "hold", F, None), (f"{ds}:{s}", "keyframe16", F, None)] + [(f"{ds}:{s}", m, F, 8) for m in ("block", "vcaL", "morph", "lsca", "ffdca", "ffdwarp")]
    with Pool(64) as p: res = p.map(job, J, chunksize=1)
    rows = [r for r, _ in res]; json.dump(rows, open(os.path.join(OUT, "real.json"), "w"), default=float, indent=0)
    np.savez_compressed(os.path.join(OUT, "real_preds.npz"), **{f'{r["task"]}|{r["method"]}': np.packbits(P) for r, P in res})
    for ds in ("bad_apple", "fantasmagorie"):
        print(ds)
        for m in ("hold", "keyframe16", "block", "vcaL", "morph", "lsca", "ffdca", "ffdwarp"):
            x = [r for r in rows if r["task"].startswith(ds) and r["method"] == m]
            th = [r["thin_recall"] for r in x if r["thin_recall"] is not None]
            print(f'   {m:10s} n={len(x):2d} bits {np.mean([r["bits"] for r in x]):6.0f} err {np.mean([r["err"] for r in x]):.3f} iou {np.mean([r["iou"] for r in x]):.3f} bf1 {np.mean([r["bf1"] for r in x]):.3f} '
                  f'f1 {np.mean([r["f1"] for r in x]):.3f} thin {np.mean(th) if th else float("nan"):.2f} cc_abs {np.mean([r["cc_abs"] for r in x]):.2f} holes {np.mean([r["hole_excess"] for r in x]):+.2f}')


if __name__ == "__main__":
    dict(synth=synth, real=real)[sys.argv[1]]()
