"""Exp46 (Campaign 12): evaluate ncab-v2 exports (phased NCA) from the FILE ALONE against the preprocessed binary original.
Adds to ca/diag10.full_report: per-phase error, error at frames within +-3 of each schedule boundary (before/after),
error within +-3 of detected cut/wipe onsets (ca/segment.signals), missed fades (original uniform, replay not), worst
16-frame window, the C12 gates (docs/CAMPAIGN_12.md section 4), replay time/memory, and (--node) the independent Node
replay: binary mismatches and max/mean continuous difference per quarter.
usage: exp46_eval12.py OUT_JSON EXPORT [EXPORT ...] [--node]"""
import json, os, resource, subprocess, sys, time
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca import phase_nca, segment
from ca.diag10 import full_report
from ca.motion9 import lag_terms
from exp45_c12 import target


def evaluate(path, node=False):
    hd, E, sched, h0, f0 = phase_nca.load(path); m = hd["meta"]; T = hd["T"]
    t0 = time.perf_counter(); B, V = phase_nca.rollout_np(E, f0, sched, hd["S"], h0); dt = time.perf_counter() - t0
    F = target(f"{m['src_start']}:{T}"); assert (F[0] == f0).all()
    r = score(B, V, F, hd["schedule"])
    r["file_bytes"] = os.path.getsize(path); r["header"] = {k: hd[k] for k in ("mode", "K", "C", "Fh", "S", "T", "init", "weight_dtype")}
    r["replay"] = dict(numpy_s=round(dt, 3), frames_per_s=round((T - 1) / dt, 1), peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1))
    if node:
        tmp = path + ".node.f32"; t1 = time.perf_counter()
        subprocess.run(["node", os.path.join(ROOT, "player", "nca_replay.mjs"), path, tmp], capture_output=True, check=True); nd_s = time.perf_counter() - t1
        Vn = np.fromfile(tmp, np.float32).reshape(V.shape); os.remove(tmp); Bn = np.concatenate([f0[None], Vn[1:] > 0.5]).astype(np.uint8); d = np.abs(Vn[1:] - V[1:])
        r["verify"] = dict(binary_mismatch_px=int((Bn != B).sum()), max_abs_dv=float(d.max()), mean_abs_dv=float(d.mean()),
                           max_dv_by_quarter=[float(q.max()) for q in np.array_split(d.reshape(len(d), -1).max(1), 4)], node_s=round(nd_s, 2))
    return r, B, V


def score(B, V, F, schedule):
    """All metrics and C12 gates for a binary replay B (T,H,W) against F; V may be None; schedule [{start,end,...}] or []."""
    T = len(F); r = full_report(B, V, F); r.pop("err_curve"); E_t = (B != F).mean((1, 2)); r["err_curve"] = [round(float(x), 4) for x in E_t]
    idx = np.array_split(np.arange(T), 3); r["thirds_lag4"] = []
    for ix in idx: lt = lag_terms(B[ix[0]:ix[-1] + 1], F[ix[0]:ix[-1] + 1], 4); r["thirds_lag4"].append(dict(f1=lt["f1_tol"], mass=lt["mass_ratio"]))
    e3 = [x["err"] for x in r["thirds"]]
    sustained = all(x["f1"] >= 0.5 and x["mass"] is not None and 0.5 <= x["mass"] <= 1.5 for x in r["thirds_lag4"]) and (e3[2] <= 3 * e3[0] or e3[2] <= 0.03)
    win = max(float(E_t[a:a + 16].mean()) for a in range(1, T - 15)); sig = segment.signals(F)
    onsets = [t for t in range(1, T) if sig["cut"][t] or (sig["change"][t] >= segment.WIPE and sig["change"][t - 1] < segment.WIPE)]
    near = sorted({u for t in onsets for u in range(max(1, t - 3), min(T, t + 4))})
    fade = [t for t in range(1, T) if sig["uniform"][t]]; missed = [t for t in fade if (B[t] != F[t]).mean() > 0.05]
    r["phases"] = [dict(p, err=float(E_t[p["start"]:p["end"]].mean()), before=float(E_t[max(1, p["start"] - 3):p["start"]].mean()) if p["start"] > 1 else None,
                        after=float(E_t[p["start"]:p["start"] + 3].mean())) for p in schedule]
    r["transitions"] = dict(onsets=onsets, err_near=float(E_t[near].mean()) if near else None, fade_frames=len(fade), missed_fades=missed)
    r["worst16"] = win
    r["gates"] = dict(sustained=bool(sustained), full_err=r["all"]["err"] <= 0.035, worst16=win <= 0.09,
                      transitions=(r["transitions"]["err_near"] is None or r["transitions"]["err_near"] <= 0.10) and not missed)
    r["gates"]["success"] = all(r["gates"].values())
    return r


def baselines(F):
    """Known loopholes, scored with the same gates: hold, 2-px edge checkerboard flicker on the original, the original
    eroded by 1 px (shape shrink), a 3x3 majority blur of the original (smoothing that erases thin parts)."""
    from scipy import ndimage
    T = len(F); st = np.ones((1, 3, 3), bool); B = {}
    B["hold"] = np.repeat(F[:1], T, 0)
    edge = F.astype(bool) ^ ndimage.binary_erosion(F.astype(bool), st); cb = (np.indices(F.shape[1:]).sum(0) % 2).astype(bool)
    B["edge_flicker"] = np.where(edge & (cb[None] ^ (np.arange(T) % 2 == 1)[:, None, None]), 1 - F, F)
    B["shrink_1px"] = ndimage.binary_erosion(F.astype(bool), st).astype(np.uint8)
    B["majority_blur"] = (ndimage.uniform_filter(F.astype(float), (1, 3, 3)) > 0.5).astype(np.uint8)
    for k in B: B[k][0] = F[0]
    return B


if __name__ == "__main__":
    out = sys.argv[1]; node = "--node" in sys.argv; res = json.load(open(out)) if os.path.exists(out) else {}
    if "--baselines" in sys.argv:   # usage: exp46_eval12.py OUT --baselines START:LEN
        spec = sys.argv[sys.argv.index("--baselines") + 1]; F = target(spec)
        for k, B in baselines(F).items():
            r = score(B, None, F, []); res[f"baseline:{spec}:{k}"] = r
            print(f"baseline {k} {spec}: err {r['all']['err']:.4f} worst16 {r['worst16']:.4f} lag4 thirds {[(round(x['f1'], 2), round(x['mass'] or 0, 2)) for x in r['thirds_lag4']]} thin fg/bg {r['thin']['thin_fg_recall']:.2f}/{r['thin']['thin_bg_recall']:.2f} gates {r['gates']}")
        json.dump(res, open(out, "w"), indent=1, default=float); sys.exit(0)
    for p in [a for a in sys.argv[2:] if not a.startswith("--")]:
        r, B, V = evaluate(p, node); res[os.path.relpath(p, ROOT)] = r; g = r["gates"]
        print(f"{os.path.relpath(p, ROOT)}: err {r['all']['err']:.4f} thirds {[round(x['err'], 4) for x in r['thirds']]} worst16 {r['worst16']:.4f} "
              f"near-transitions {r['transitions']['err_near']} missed fades {len(r['transitions']['missed_fades'])}/{r['transitions']['fade_frames']} "
              f"lag4 thirds {[(round(x['f1'], 2), round(x['mass'] or 0, 2)) for x in r['thirds_lag4']]} gates {g} bytes {r['file_bytes']}"
              + (f" node mism {r['verify']['binary_mismatch_px']} max|dv| {r['verify']['max_abs_dv']:.1e}" if node else ""), flush=True)
    json.dump(res, open(out, "w"), indent=1, default=float)
