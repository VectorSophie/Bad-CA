"""Exp32 (Campaign 09, before training): loophole baselines + feasibility certificates on the two task windows.
Baselines (all start from the window's true first frame): hold, black, white, invert-every-frame, edge checkerboard
flicker (pixels within 1 px of the first frame's silhouette edge toggled in a (x+y+t) checkerboard), local
on-then-revert blips (random 20% of edge pixels flipped on odd frames, back on even frames, rng 0), shift right 1 px/frame,
plus references: true clip delayed by 1 frame, and the C8 representative (autonomous from the clip's first frame).
Certificates (CHECKPOINT_01 tools): light cone from the window start (any radius-1 rule, constant hidden init) for S in
1, 2, 4; history conflicts (F_priv, S=1, any k); memoryless state repeat. Writes runs/c9/window_baselines.json."""
import json, os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca import container
from ca.motion9 import window_report
from ca.sat import history_conflicts, light_cone_conflicts, state_repeat_conflict
from c7_common import target

TASKS = {"main": ("ba48_4500_60", 40), "confirm": ("ba48_1800_60", 44)}
C8 = {x["name"]: x for x in json.load(open(os.path.join(ROOT, "runs", "c8", "media", "video_index.json"))) if "selection" in x["name"]}
out = {}
for task, (clip, t0) in TASKS.items():
    full = target(clip); F = full[t0:t0 + 16]; T = len(F); f0 = F[0].astype(bool); rng = np.random.default_rng(0)
    edge = np.zeros_like(f0)
    p = np.pad(f0, 1)
    for dy in range(3):
        for dx in range(3): edge |= p[dy:dy + f0.shape[0], dx:dx + f0.shape[1]] != f0
    yy, xx = np.mgrid[:f0.shape[0], :f0.shape[1]]
    B = {"hold": np.repeat(F[:1], T, 0), "black": np.zeros_like(F), "white": np.ones_like(F), "invert": np.stack([F[0] ^ (t % 2) for t in range(T)]),
         "edge_checker_flicker": np.stack([f0 ^ (edge & ((yy + xx + t) % 2 == 0) & (t > 0)) for t in range(T)]),
         "local_on_revert": np.stack([f0 ^ (edge & (rng.random(f0.shape) < 0.2) & (t % 2 == 1)) for t in range(T)]),
         "shift_right": np.stack([np.pad(F[0], ((0, 0), (t, 0)))[:, :F.shape[2]] for t in range(T)]),
         "true_lag1": np.concatenate([F[:1], F[:-1]])}
    for k in ("black", "white"): B[k][0] = F[0]
    rep = json.load(open(os.path.join(ROOT, C8[("main_" if task == "main" else "second_") + clip + "_selection"]["representative"])))
    B["C8_rep_autonomous_from_clip_start"] = container.decode(open(os.path.join(ROOT, rep["export"]["file"]), "rb").read())[1][t0:t0 + 16]
    out[task] = dict(clip=clip, t0=t0, baselines={})
    for name, P in B.items():
        r = window_report(P.astype(np.uint8), F); r.pop("err_curve"); out[task]["baselines"][name] = r
        print(task, f"{name:34s} err/hold {r['err_over_hold']:.2f}  lag1 F1 {r['lag1']['f1_tol']:.2f} P {r['lag1']['prec_tol']:.2f}  lag4 F1 {r['lag4']['f1_tol']:.2f} P {r['lag4']['prec_tol']:.2f} R {r['lag4']['rec_tol']:.2f}  revert {r['revert_pred']}")
    out[task]["certs"] = dict(light_cone_window={S: light_cone_conflicts(F, S) for S in (1, 2, 4)},
                              light_cone_C_prefix={S: light_cone_conflicts(full[:t0 + 16], S) for S in (1, 2, 4)},
                              history_conflict_window=history_conflicts(F), state_repeat_window=state_repeat_conflict(F))
    print(task, "certs", json.dumps(out[task]["certs"]))
json.dump(out, open(os.path.join(ROOT, "runs", "c9", "window_baselines.json"), "w"), indent=1, default=float)
