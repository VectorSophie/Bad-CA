"""Path V0: oracle motion-field feasibility of the velocity-carrying CA on real Bad Apple. CPU only.

ORACLE CONTROL: every control code below is chosen by looking at the target frames of its own segment (sequential
greedy, coordinate descent per region, objective = pixel mismatches summed over the segment's frames, state carried
from the previous segment). The runtime then receives only the start frame and the fixed schedule; no target frame is
injected. The persistence test ("extrap") is the exception that measures unseen frames: one event fitted on frames
1..8 only, scored on frames 9..31, which the control never saw.

Methods (each with the same oracle search and the same metrics):
  hold            repeat the start frame (0 bits)
  vcaL            velocity-carrying CA, lagrangian, codes (vy, vx, g) in [-4, 4]^3 (729 codes, 9.51 bits per region-event)
  vcaL_v          same, velocity only (g = 0; 81 codes, 6.34 bits)
  vcaE            eulerian velocity field (same rule and alphabet, velocity belongs to the grid)
  block           whole-image integer block shift per region, (dy, dx) in [-2, 2]^2 px/frame (25 codes, 4.64 bits)
  codebook        Stage 0 shared m0 K = 8 table, global, one code per segment (3 bits)
  keyframe        signed-distance interpolation between true keyframes every 8 / 16 / 31 frames (bits = zlib size)
Grids 1x1, 2x2, 3x3; events every 4, 8, 16 frames or once (frame 0). Clips: 32 frames (31 predicted) from the frozen
manifest runs/path_v/v0_manifest.json (dev clips were used only to debug; test clips are scored once).
usage: pv_v0.py prereg | run dev|test"""
import json, math, os, sys, time, zlib
from multiprocessing import Pool
import numpy as np
from scipy import ndimage
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path[:0] = [ROOT, os.path.dirname(os.path.abspath(__file__))]
import pv_vca as V
OUT = os.path.join(ROOT, "runs", "path_v"); MAN = os.path.join(OUT, "v0_manifest.json"); N = 32
CATS = ("translation", "growth", "deformation", "multi_motion", "major_transform", "cut_wipe")
GRIDS = ((1, 1), (2, 2), (3, 3)); EVERY = (4, 8, 16, None)
VEL = [(a, b) for a in range(-4, 5) for b in range(-4, 5)]; GR = list(range(-4, 5)); SHIFTS = [(a, b) for a in range(-2, 3) for b in range(-2, 3)]
ST8 = ndimage.generate_binary_structure(2, 2)
_X = None


def video():
    global _X
    if _X is None:
        from exp45_c12 import target
        _X = target("0:6572").astype(np.uint8)
    return _X


# ---------------------------------------------------------------- manifest ----------------------------------------------------------------
def prereg():
    if os.path.exists(MAN): sys.exit("manifest exists (frozen)")
    from ca import segment
    X = video(); cut = segment.signals(X)["cut"]; A = json.load(open(os.path.join(OUT, "atlas.json")))["windows"]; rng = np.random.default_rng(20261114)
    clips = []; used = np.zeros(len(X), bool)
    for cat in CATS:
        cand = [w["start"] for w in A if w["tag"] == cat and w["start"] + N + 8 <= len(X) and (cat == "cut_wipe" or not cut[w["start"] + 1:w["start"] + N].any())]
        rng.shuffle(cand); picked = []
        for s in cand:
            if used[max(0, s - N):s + N + 8].any(): continue
            picked.append(s); used[s:s + N + 8] = True
            if len(picked) == 8: break
        for i, s in enumerate(picked):
            split = "dev" if i < 2 else "test"; clips.append(dict(cat=cat, start=s, split=split, offset=0))
            if split == "test" and (cat == "cut_wipe" or not cut[s + 9:s + 8 + N].any()): clips.append(dict(cat=cat, start=s + 8, split=split, offset=8))
    json.dump(dict(fixed_at=time.strftime("%Y-%m-%d %H:%M:%S"), seed=20261114, clip_frames=N,
                   rule="per atlas tag: up to 8 non-overlapping 32-frame clips starting at an atlas window of that tag, cut-free (except cut_wipe); "
                        "first 2 = dev (debugging only), rest = test; each test clip also gets a second start 8 frames later (same scene) when cut-free",
                   clips=clips), open(MAN, "w"), indent=1)
    from collections import Counter
    print(Counter((c["cat"], c["split"]) for c in clips))


# ---------------------------------------------------------------- metrics ----------------------------------------------------------------
def holes(fg):
    lab, n = ndimage.label(~fg); border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
    return sum(1 for i in range(1, n + 1) if i not in border)


def metrics(P, F):
    """P: predicted frames for F[1:] (n, H, W). F: true frames incl. the start (n + 1, H, W)."""
    import pd_vis
    m = pd_vis.sequence_metrics(np.concatenate([F[:1], P]), F, first_free=1); bg = pd_vis.bg_colour(F[0])
    fgp, fgt = P != bg, F[1:] != bg
    cc = [ndimage.label(a, ST8)[1] - ndimage.label(b, ST8)[1] for a, b in zip(fgp, fgt)]
    ho = [holes(a) - holes(b) for a, b in zip(fgp, fgt)]
    iou = [float((a & b).sum() / max(1, (a | b).sum())) for a, b in zip(fgp, fgt)]
    return dict(err=m["err"], hold=m["hold"], f1=m["lag1"]["f1"], prec=m["lag1"]["prec"], rec=m["lag1"]["rec"], f1_4=m["lag4"]["f1"], iou=m["iou"],
                bf1=m["boundary_f1"], collapse=m["first_iou_below_0_5"], period=m["terminal_period_pred"], cc_excess=float(np.mean(cc)),
                cc_abs=float(np.mean(np.abs(cc))), hole_excess=float(np.mean(ho)), err_t=(P != F[1:]).mean((1, 2)).round(4).tolist(), iou_t=np.round(iou, 3).tolist())


# ---------------------------------------------------------------- oracle fits ----------------------------------------------------------------
def copy_state(S, B):
    return {k: (np.repeat(v[:1], B, 0).copy() if isinstance(v, np.ndarray) else v) for k, v in S.items()}


SUB = 1   # substeps per frame (V1 "S2" variant sets 2: speed cap 2 px/frame with the same alphabet and bits)


def sim(S, codes, rmap, T, mode, ph, event=True):
    """Simulate T frames from state S (batch 1) for a batch of code sets (B, R, 3). Returns (frames (B, T, H, W), S, ph)."""
    B = len(codes); S = copy_state(S, B); ph = {k: np.repeat(v[:1], B, 0).copy() for k, v in ph.items()}
    if event:
        if mode == "L": V.apply_codes(S, codes, rmap)
        else: ph = {k: np.full_like(v, V.Q // 2) for k, v in ph.items()}
    out = []
    for _ in range(T):
        for _ in range(SUB): V.step(S, None if mode == "L" else (codes, rmap, ph))
        out.append(V.visible(S))
    return np.stack(out, 1), S, ph


def fit_vca(F, grid, every, mode="L", growth=True, fit_frames=None):
    """Oracle sequential greedy. F: (N, H, W) true frames, F[0] = start. Returns predicted (N-1, H, W) and the schedule."""
    rmap = V.region_map(grid); R = grid[0] * grid[1]; n = len(F) - 1; ev = list(range(0, n, every)) if every else [0]
    S = V.init_state(F[0]); ph = {p: np.full((1, R), V.Q // 2, np.int16) for p in ("py", "px", "pg")}
    codes = np.zeros((R, 3), int); sched = {}; preds = []
    for k, t0 in enumerate(ev):
        t1 = ev[k + 1] if k + 1 < len(ev) else n; T = t1 - t0
        tgt = F[t0 + 1:t0 + 1 + (min(T, fit_frames) if fit_frames else T)]; Tf = len(tgt)
        for sweep in range(2):
            for r in range(R):
                for opts, cols in ((VEL, (0, 1)), (GR if growth else [], (2,))):
                    if not opts: continue
                    C = np.repeat(codes[None], len(opts), 0)
                    for i, o in enumerate(opts): C[i, r, list(cols)] = o if isinstance(o, tuple) else [o]
                    fr, _, _ = sim(S, C, rmap, Tf, mode, ph); e = (fr != tgt[None]).sum((1, 2, 3)); codes = C[int(np.argmin(e))]
        fr, S, ph = sim(S, codes[None], rmap, T, mode, ph); S = {k: (v[:1] if isinstance(v, np.ndarray) else v) for k, v in S.items()}
        ph = {k: v[:1] for k, v in ph.items()}; preds.append(fr[0]); sched[t0] = codes.tolist()
    return np.concatenate(preds), sched


def fit_block(F, grid, every, fit_frames=None):
    rmap = V.region_map(grid); R = grid[0] * grid[1]; n = len(F) - 1; ev = list(range(0, n, every)) if every else [0]
    cur = F[0]; d = np.zeros((R, 2), int); sched = {}; preds = []
    for k, t0 in enumerate(ev):
        t1 = ev[k + 1] if k + 1 < len(ev) else n; T = t1 - t0; tgt = F[t0 + 1:t0 + 1 + (min(T, fit_frames) if fit_frames else T)]
        for sweep in range(2):
            for r in range(R):
                C = np.repeat(d[None], len(SHIFTS), 0)
                for i, o in enumerate(SHIFTS): C[i, r] = o
                fr = V.block_shift(cur, len(tgt), {0: C}, rmap); d = C[int(np.argmin((fr != tgt[None]).sum((1, 2, 3))))]
        fr = V.block_shift(cur, T, {0: d[None]}, rmap)[0]; preds.append(fr); cur = fr[-1]; sched[t0] = d.tolist()
    return np.concatenate(preds), sched


def fit_codebook(F, every, fit_frames=None):
    import s0_codebook as s0
    Ts = np.load(os.path.join(ROOT, "runs", "context_s0", "codebooks.npz"))["m0_K8_global"]; n = len(F) - 1; ev = list(range(0, n, every)) if every else [0]
    prev, cur = F[0], F[0]; preds = []; sched = {}
    for k, t0 in enumerate(ev):
        t1 = ev[k + 1] if k + 1 < len(ev) else n; T = t1 - t0; tgt = F[t0 + 1:t0 + 1 + (min(T, fit_frames) if fit_frames else T)]
        rs = [s0.roll(Ts[j:j + 1], prev, cur, len(tgt), "m0") for j in range(len(Ts))]; j = int(np.argmin([(r != tgt).sum() for r in rs]))
        fr = s0.roll(Ts[j:j + 1], prev, cur, T, "m0"); preds.append(fr); prev, cur = (fr[-2] if T > 1 else cur), fr[-1]; sched[t0] = j
    return np.concatenate(preds), sched


def sdf(fg): return ndimage.distance_transform_edt(~fg) - ndimage.distance_transform_edt(fg)


def keyframe(F, spacing):
    """Signed-distance interpolation between the start frame and true keyframes at spacing (the last frame always)."""
    import pd_vis
    bg = pd_vis.bg_colour(F[0]); n = len(F) - 1; keys = sorted(set(list(range(spacing, n, spacing)) + [n])); prev = 0; out = []; bits = 0
    for kf in keys:
        a, b = sdf(F[prev] != bg), sdf(F[kf] != bg); bits += 8 * len(zlib.compress(np.packbits(F[kf]).tobytes(), 9))
        for t in range(prev + 1, kf + 1): w = (t - prev) / (kf - prev); out.append(np.where(((1 - w) * a + w * b) < 0, 1 - bg, bg).astype(np.uint8))
        prev = kf
    return np.stack(out), bits


# ---------------------------------------------------------------- run ----------------------------------------------------------------
def bits_per_event(method, grid):
    R = grid[0] * grid[1]
    return {"vcaL": R * math.log2(729), "vcaE": R * math.log2(729), "vcaL_v": R * math.log2(81), "block": R * math.log2(25), "codebook": 3.0}[method]


def job(a):
    clip, method, grid, every = a; X = video(); s = clip["start"]; F = X[s:s + N]; t0 = time.process_time(); extra = {}
    if method == "hold": P = np.repeat(F[:1], N - 1, 0); sched = {}; bits = 0.0
    elif method.startswith("keyframe"): P, bits = keyframe(F, int(method[8:])); sched = {}
    elif method == "codebook": P, sched = fit_codebook(F, every); bits = len(sched) * 3.0
    elif method == "block": P, sched = fit_block(F, grid, every); bits = len(sched) * bits_per_event("block", grid)
    else:
        P, sched = fit_vca(F, grid, every, mode="E" if method == "vcaE" else "L", growth=method != "vcaL_v"); bits = len(sched) * bits_per_event(method, grid)
    r = dict(clip=clip, method=method, grid=list(grid) if grid else None, every=every, bits=bits, sched_bytes=len(json.dumps(sched)), **metrics(P, F))
    # persistence: one event fitted on frames 1..8 only, scored on unseen frames 9..31
    if every is None and method in ("vcaL", "vcaL_v", "vcaE", "block", "codebook"):
        if method == "block": Pe, _ = fit_block(F, grid, None, fit_frames=8)
        elif method == "codebook": Pe, _ = fit_codebook(F, None, fit_frames=8)
        else: Pe, _ = fit_vca(F, grid, None, mode="E" if method == "vcaE" else "L", growth=method != "vcaL_v", fit_frames=8)
        Fe = F[8:]; r["extrap"] = metrics(Pe[8:], Fe)
    r["cpu_s"] = round(time.process_time() - t0, 2)
    return r, (P if (method, tuple(grid) if grid else None, every) in KEEP else None)


KEEP = {("hold", None, None), ("vcaL", (3, 3), 8), ("vcaL", (2, 2), None), ("vcaL", (3, 3), None), ("block", (3, 3), 8), ("vcaE", (3, 3), 8), ("keyframe16", None, None), ("vcaL", (1, 1), 8)}


def jobs_for(clips):
    J = []
    for c in clips:
        J += [(c, "hold", None, None), (c, "keyframe8", None, None), (c, "keyframe16", None, None), (c, "keyframe31", None, None)]
        J += [(c, "codebook", None, e) for e in EVERY]
        for g in GRIDS:
            for e in EVERY:
                J += [(c, m, g, e) for m in ("vcaL", "vcaL_v", "vcaE", "block")]
    return J


def run(split):
    man = json.load(open(MAN)); clips = [c for c in man["clips"] if c["split"] == split]; J = jobs_for(clips); t0 = time.time()
    J.sort(key=lambda j: -(0 if j[2] is None else j[2][0] * j[2][1]) * (1 if j[3] else 0.5))
    rows, keep = [], {}
    with Pool(64) as p:
        for r, P in p.imap_unordered(job, J, chunksize=1):
            rows.append(r)
            if P is not None: keep[f'{r["clip"]["start"]}|{r["method"]}|{r["grid"]}|{r["every"]}'] = np.packbits(P)
    json.dump(rows, open(os.path.join(OUT, f"v0_{split}.json"), "w"), default=float)
    np.savez_compressed(os.path.join(OUT, f"v0_{split}_preds.npz"), **keep)
    print(split, len(rows), "rows", round(time.time() - t0, 1), "s", round(sum(r["cpu_s"] for r in rows) / 3600, 3), "core-h (process time)")


if __name__ == "__main__":
    if sys.argv[1] == "prereg": prereg()
    else: run(sys.argv[2])
