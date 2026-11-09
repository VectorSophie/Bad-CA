"""Exp26 (Campaign 08): objective/metric loophole check on fixed baselines, BEFORE any search.
Baselines on each clip: hold, all-black, all-white, invert-every-frame (flicker), shift-right 1 px/frame
of frame 0 (simple translation), random flips (5% of pixels per frame from frame 0, rng 0), the true clip
delayed by 1 frame (lag, an upper reference for 'late but right' motion), and the C7 best CGP (decoded export).
Writes runs/c8/loopholes.json."""
import json, os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca import container
from ca.motion import report
from c7_common import target

out = {}
for clip, c7 in [("ba48_4500_60", "runs/c7/c4_cgp/ba48_4500_60__k1_S2__G32__s3.best.cab"), ("ba48_1800_60", "runs/c7/c4_cgp/ba48_1800_60__k1_S2__G16__s3.best.cab")]:
    F = target(clip); T = len(F); rng = np.random.default_rng(0)
    B = {"hold": np.repeat(F[:1], T, 0), "black": np.zeros_like(F), "white": np.ones_like(F)}
    B["flicker"] = np.stack([F[0] ^ (t % 2) for t in range(T)])
    B["shift_right"] = np.stack([np.pad(F[0], ((0, 0), (t, 0)))[:, :F.shape[2]] for t in range(T)])
    X = [F[0].copy()]
    for t in range(1, T): X.append(X[-1] ^ (rng.random(F[0].shape) < 0.05))
    B["random_flips_5pct"] = np.stack(X)
    B["true_lag1"] = np.concatenate([F[:1], F[:-1]])
    B["C7_best_cgp"] = container.decode(open(os.path.join(ROOT, c7), "rb").read())[1]
    for k in ("black", "white"): B[k][0] = F[0]   # every replay starts from the true first frame
    out[clip] = {}
    for name, P in B.items():
        r = report(P.astype(np.uint8), F, lam=1.0); out[clip][name] = r
        print(clip, f"{name:18s}", " | ".join(f"{s}: err {r[s]['err']:.3f}/hold {r[s]['hold']:.3f} F1tol {r[s]['f1_tol']:.3f} P {r[s]['prec_tol']:.2f} R {r[s]['rec_tol']:.2f} mass {r[s]['change_mass_ratio']:.2f}" for s in ("early", "mid", "late")),
              f"| J(lam=1) {r['J']:.3f} period {r['late_period']}")
json.dump(out, open(os.path.join(ROOT, "runs", "c8", "loopholes.json"), "w"), indent=1, default=float)
