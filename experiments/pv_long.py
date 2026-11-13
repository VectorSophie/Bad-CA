"""Path V failure isolation: 96-frame rollouts across a scene transition with the oracle velocity CA (vcaL, 3x3).
Segments: for every third preregistered scene (FINDINGS §10 windows; 8 scenes), frames [e - 48, e + 48) where e is the
scene's end (a cut or wipe, as recorded in prereg_windows.json). Runs (all ORACLE control, see pv_v0):
  cont      events every 8 frames from e - 48, state carried through the transition (an event falls exactly on e)
  restart   DIAGNOSTIC ONLY: the same, but re-initialised from the true frame e (not a continuous reproduction)
  once_pre  one event fitted on frames 1..8 after e - 48, then no events (persistence before the transition)
  once_post DIAGNOSTIC: one event fitted on frames 1..8 after e, started from the true frame e
Divergence = first frame with foreground IoU < 0.5 for 3 consecutive frames; useful = frames until the rollout's
error exceeds the error of holding its own start frame for 3 consecutive frames.
usage: pv_long.py  -> runs/path_v/long.json, runs/path_v/fig_long_*.png"""
import json, os, sys
from multiprocessing import Pool
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path[:0] = [ROOT, os.path.dirname(os.path.abspath(__file__))]
import pv_v0, pv_vca as V
OUT = pv_v0.OUT; HALF = 48


def iou_t(P, T):
    import pd_vis
    bg = pd_vis.bg_colour(T[0]); return np.array([((p != bg) & (t != bg)).sum() / max(1, ((p != bg) | (t != bg)).sum()) for p, t in zip(P, T)])


def first_run(mask, k=3):
    for i in range(len(mask) - k + 1):
        if mask[i:i + k].all(): return i
    return None


def seg_job(e):
    X = pv_v0.video(); a, b = e - HALF, e + HALF; F = X[a:b]; out = dict(start=a, transition=e)
    P, sched = pv_v0.fit_vca(F, (3, 3), 8); runs = {"cont": (P, F, a, sorted(map(int, sched)))}
    P1, s1 = pv_v0.fit_vca(X[a:e + 1], (3, 3), 8); P2, s2 = pv_v0.fit_vca(X[e:b], (3, 3), 8)
    runs["restart"] = (np.concatenate([P1, P2]), F, a, sorted(map(int, s1)) + [HALF + int(t) for t in s2])
    Pp, _ = pv_v0.fit_vca(F, (3, 3), None, fit_frames=8); runs["once_pre"] = (Pp, F, a, [0])
    Pq, _ = pv_v0.fit_vca(X[e:b], (3, 3), None, fit_frames=8); runs["once_post"] = (Pq, X[e:b], e, [0])
    for k, (P, G, st, evs) in runs.items():
        T = G[1:]; err = (P != T).mean((1, 2)); io = iou_t(P, T); hold = (G[0] != T).mean((1, 2))
        dv = first_run(io < 0.5); us = first_run(err > hold + 1e-9)
        out[k] = dict(start=st, events=evs, err=err.round(4).tolist(), iou=io.round(3).tolist(), hold=hold.round(4).tolist(),
                      divergence=None if dv is None else st + 1 + dv, useful_frames=len(T) if us is None else us,
                      err_before=float(err[:HALF - 1].mean()) if k in ("cont", "restart") else None, err_after=float(err[HALF - 1:].mean()) if k in ("cont", "restart") else None)
        out[k]["_pred"] = np.packbits(P).tobytes().hex()
    return out


def timeline(r, path):
    """Error (red) and IoU (blue) per frame for cont and restart, event ticks, the transition line, divergence marks."""
    from PIL import Image, ImageDraw
    W, H, pad = 900, 300, 40; img = Image.new("RGB", (W, H), (250, 249, 246)); d = ImageDraw.Draw(img); n = 2 * HALF - 1
    x = lambda t: pad + (W - 2 * pad) * t / n; y = lambda v: H - pad - (H - 2 * pad) * v
    d.line([(pad, y(0)), (W - pad, y(0))], fill=(120, 120, 120)); d.line([(pad, y(1)), (W - pad, y(1))], fill=(220, 220, 220))
    d.line([(x(HALF - 1), pad - 10), (x(HALF - 1), H - pad)], fill=(0, 0, 0), width=2); d.text((x(HALF - 1) + 3, pad - 12), f"transition {r['transition']}", fill=(0, 0, 0))
    for t in r["cont"]["events"]: d.line([(x(t), H - pad), (x(t), H - pad + 6)], fill=(90, 90, 90))
    cols = {"cont": ((200, 40, 40), (40, 80, 200)), "restart": ((240, 150, 150), (150, 170, 240))}
    for k in ("restart", "cont"):
        e, i = r[k]["err"], r[k]["iou"]
        d.line([(x(t), y(min(1, v * 2))) for t, v in enumerate(e)], fill=cols[k][0], width=2)
        d.line([(x(t), y(v)) for t, v in enumerate(i)], fill=cols[k][1], width=2)
        if r[k]["divergence"] is not None: t = r[k]["divergence"] - r[k]["start"] - 1; d.ellipse([x(t) - 4, y(0.5) - 4, x(t) + 4, y(0.5) + 4], outline=cols[k][1], width=2)
    d.line([(x(t), y(min(1, v * 2))) for t, v in enumerate(r["cont"]["hold"])], fill=(160, 160, 160))
    d.text((pad, 6), f"frames {r['start']}-{r['start'] + 2 * HALF - 1}: oracle vcaL 3x3, events every 8 (ticks). red = pixel error x2, blue = IoU, "
                     f"grey = hold error x2; dark = continuous, light = DIAGNOSTIC restart at the transition; circle = divergence", fill=(30, 30, 30))
    img.save(path)


if __name__ == "__main__":
    PRE = json.load(open(os.path.join(ROOT, "runs", "readable_rules", "gen", "prereg_windows.json")))["windows"]
    ends = [w["scene"][1] for i, w in enumerate(PRE) if i % 3 == 0 and w["scene"][1] - HALF >= 0 and w["scene"][1] + HALF <= 6572]
    with Pool(len(ends)) as p: res = p.map(seg_job, ends)
    for r in res: timeline(r, os.path.join(OUT, f"fig_long_{r['transition']}.png"))
    json.dump([{k: ({kk: vv for kk, vv in v.items() if kk != "_pred"} if isinstance(v, dict) else v) for k, v in r.items()} for r in res], open(os.path.join(OUT, "long.json"), "w"), indent=0)
    np.savez_compressed(os.path.join(OUT, "long_preds.npz"), **{f'{r["transition"]}_{k}': np.frombuffer(bytes.fromhex(r[k]["_pred"]), np.uint8) for r in res for k in ("cont", "restart", "once_pre", "once_post")})
    for r in res:
        print(r["transition"], " | ".join(f'{k}: div {r[k]["divergence"]} useful {r[k]["useful_frames"]}' + (f' err pre/post {r[k]["err_before"]:.3f}/{r[k]["err_after"]:.3f}' if r[k]["err_before"] is not None else "")
                                          for k in ("cont", "restart", "once_pre", "once_post")))
