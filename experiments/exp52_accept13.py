"""Exp52 (Campaign 13): supplementary full-song acceptance criteria (fixed in docs/CAMPAIGN_13.md before the whole-song
result), computed on the numpy replay of an export (or on a baseline array), next to the C12 gates.
Per 240-frame window (so quiet frames cannot hide hard parts) and globally:
  W1 window error <= 0.06 in >= 90 % of windows and <= 0.15 in every window
  W2 frames inside failing 16-frame windows (error > 0.09) <= 5 % of all frames
  M  in windows where the original changes (lag-4 change mass > 1 %): lag-4 tolerant change F1 >= 0.5 and change mass
     ratio 0.5..1.5 in >= 90 % of them
  T  thin parts and thin gaps recall >= 0.70 (global)
  K  small specks <= 6 x original and small holes <= 6 x original (global; calibrated on the user-accepted 0-1467 result: 3.6x / 5.0x)
  B  spurious one-frame flips <= 1.0 per frame (global)
  P  mean error within +-3 frames of phase boundaries <= 2 x the global error
  F  fade frames (original uniform) reproduced (frame error <= 0.05) for >= 90 % of them
  S  numerical: max |state| <= 1e3 at every frame; with --node: Node vs numpy binary mismatch <= 0.1 % of pixels overall
     and <= 1 % in every frame
Writes the per-frame error curve and the list of failing windows. usage: exp52_accept13.py OUT_JSON EXPORT [--node]
                                                                       exp52_accept13.py OUT_JSON --baseline NAME START:LEN"""
import json, os, subprocess, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca import nca, phase_nca, segment
from ca.diag10 import blips, small_cc, thin_recall
from ca.motion9 import lag_terms
from exp45_c12 import target
from exp46_eval12 import baselines, score


def accept(B, F, boundaries=(), smax=None, node=None):
    T = len(F); e = (B != F).mean((1, 2)); r = {}
    win = [(a, min(a + 240, T)) for a in range(1, T, 240) if min(a + 240, T) - a >= 60]
    we = [float(e[a:b].mean()) for a, b in win]; r["windows"] = []
    mot_ok = []
    for (a, b), x in zip(win, we):
        lt = lag_terms(B[a - 1:b], F[a - 1:b], 4); tm = float((F[a + 3:b] != F[a - 1:b - 4]).mean()) if b - a > 4 else 0.0
        ok = None if tm <= 0.01 else bool(lt["f1_tol"] >= 0.5 and lt["mass_ratio"] is not None and 0.5 <= lt["mass_ratio"] <= 1.5)
        if ok is not None: mot_ok.append(ok)
        r["windows"].append(dict(start=a, end=b, err=x, lag4_f1=lt["f1_tol"], lag4_mass=lt["mass_ratio"], true_change=tm, motion_ok=ok))
    w16 = np.array([e[a:a + 16].mean() for a in range(1, T - 15)]); bad = np.zeros(T, bool)
    for a in np.nonzero(w16 > 0.09)[0] + 1: bad[a:a + 16] = True
    runs = []; t = 0
    while t < T:
        if bad[t]:
            s = t
            while t < T and bad[t]: t += 1
            runs.append(dict(start=s, end=t - 1, peak16=float(w16[max(0, s - 1):t - 1].max()) if t - 1 > s else float(e[s])))
        else: t += 1
    th = thin_recall(B, F); cc_p, cc_t = small_cc(B[1:]).sum(0), small_cc(F[1:]).sum(0); bl = int((blips(B) & ~blips(F)).sum())
    near = sorted({u for b in boundaries for u in range(max(1, b - 3), min(T, b + 4))}); uni = segment.signals(F)["uniform"]; fades = np.nonzero(uni)[0]
    g = float(e[1:].mean())
    crit = dict(W1=bool(np.mean(np.array(we) <= 0.06) >= 0.9 and max(we) <= 0.15), W2=bool(bad.mean() <= 0.05),
                M=bool(np.mean(mot_ok) >= 0.9) if mot_ok else None, T=bool(th["thin_fg_recall"] >= 0.7 and th["thin_bg_recall"] >= 0.7),
                K=bool(cc_p[0] <= 6 * max(cc_t[0], 1) and cc_p[1] <= 6 * max(cc_t[1], 1)), B=bool(bl / (T - 2) <= 1.0),
                P=bool(not near or e[near].mean() <= 2 * g), F=bool(len(fades) == 0 or np.mean(e[fades] <= 0.05) >= 0.9))
    if smax is not None: crit["S_state"] = bool(max(smax) <= 1e3)
    if node is not None: crit["S_node"] = bool(node["mismatch_frac"] <= 1e-3 and node["max_frame_frac"] <= 0.01)
    crit["accept"] = all(v for v in crit.values() if v is not None)
    r.update(T=T, err=g, window_err_max=max(we), window_err_frac_le_006=float(np.mean(np.array(we) <= 0.06)), failing16_frac=float(bad.mean()), failing16_runs=runs,
             motion_ok_frac=float(np.mean(mot_ok)) if mot_ok else None, thin=th, specks=[int(cc_p[0]), int(cc_t[0])], holes=[int(cc_p[1]), int(cc_t[1])],
             spurious_blips_per_frame=bl / (T - 2), boundary_err=float(e[near].mean()) if near else None, fade_frames=int(len(fades)),
             fade_ok_frac=float(np.mean(e[fades] <= 0.05)) if len(fades) else None, state_max=max(smax) if smax is not None else None, node=node,
             criteria=crit, err_curve=[round(float(x), 4) for x in e])
    return r


if __name__ == "__main__":
    out = sys.argv[1]; res = json.load(open(out)) if os.path.exists(out) else {}
    if "--baseline" in sys.argv:
        name, spec = sys.argv[sys.argv.index("--baseline") + 1:sys.argv.index("--baseline") + 3]; F = target(spec)
        B = baselines(F)[name]; r = accept(B, F); r["c12_gates"] = score(B, None, F, [])["gates"]; key = f"baseline:{name}:{spec}"
    else:
        path = sys.argv[2]; hd, E, sched, h0, f0 = phase_nca.load(path); T = hd["T"]; C = hd["C"]
        s = np.zeros((C,) + f0.shape, np.float32); s[0] = f0
        if h0 is not None: s[1:] = h0
        V, smax = [s[0].copy()], [float(np.abs(s).max())]
        for t in range(1, T):
            p = {k: E[k][sched[t]] for k in phase_nca.PNAMES}
            for _ in range(hd["S"]): s = nca.step_np(p, s)
            V.append(s[0].copy()); smax.append(float(np.abs(s).max()))
        V = np.stack(V); B = (V > 0.5).astype(np.uint8); B[0] = f0; F = target(f"{hd['meta']['src_start']}:{T}")
        node = None
        if "--node" in sys.argv:
            tmp = path + ".node.f32"; subprocess.run(["node", os.path.join(ROOT, "player", "nca_replay.mjs"), path, tmp], capture_output=True, check=True)
            Vn = np.fromfile(tmp, np.float32).reshape(V.shape); os.remove(tmp); Bn = (Vn > 0.5).astype(np.uint8); Bn[0] = f0; mm = (Bn != B).reshape(T, -1).mean(1)
            d = np.abs(Vn[1:] - V[1:]).reshape(T - 1, -1).max(1)
            node = dict(mismatch_frac=float(mm.mean()), max_frame_frac=float(mm.max()), mismatch_px=int((Bn != B).sum()), max_abs_dv=float(d.max()),
                        max_dv_by_tenth=[float(q.max()) for q in np.array_split(d, 10)])
        r = accept(B, F, [p["start"] for p in hd["schedule"][1:]], smax, node); r["c12_gates"] = score(B, V, F, hd["schedule"])["gates"]; key = os.path.relpath(path, ROOT)
        r["file_bytes"] = os.path.getsize(path); r["K"] = hd["K"]
    res[key] = r; json.dump(res, open(out, "w"), default=float)
    c = r["criteria"]; print(key, f"err {r['err']:.4f} win max {r['window_err_max']:.3f} win<=.06 {r['window_err_frac_le_006']:.2f} fail16 {r['failing16_frac']:.3f} "
          f"motion {r['motion_ok_frac']} thin {r['thin']['thin_fg_recall']:.2f}/{r['thin']['thin_bg_recall']:.2f} specks {r['specks']} holes {r['holes']} blips/f {r['spurious_blips_per_frame']:.2f} "
          f"fade ok {r['fade_ok_frac']} criteria {c} c12 {r['c12_gates']['success']}", flush=True)
