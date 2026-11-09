"""Exp02a: is the 'floor ~= change rate' result a resolution artifact?
Metric: explained_change = 1 - floor/change_rate (share of changed pixels a memoryless LUT can predict).
Exp02b: SOP minimisation of a fitted Moore LUT using unseen contexts as don't-cares; does rollout survive?"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ca.core import conflict_report, transition_counts, fit_majority_lut, rollout, metrics
from ca.video import binarize, gray_frames
from ca.export import export_lut
import subprocess

mp4 = os.path.join(os.path.dirname(__file__), "..", "refs", "Bad-Apple-Starter", "BadApple!!.mp4")
out = {}
for W in (16, 32, 64, 128):
    B = binarize(gray_frames(mp4, W)[0])
    rows = []
    for start_s in (10, 30, 62, 100, 150):
        clip = B[start_s * 30:start_s * 30 + 90]
        ch = (clip[1:] != clip[:-1]).mean()
        f1 = conflict_report(transition_counts(clip, 1, 0)[0])["min_one_step_error_rate"]
        f2 = conflict_report(transition_counts(clip, 2, 0)[0])["min_one_step_error_rate"]
        rows.append((ch, 1 - f1 / ch if ch else 1, 1 - f2 / ch if ch else 1))
    out[W] = np.array(rows).mean(0).round(3).tolist()
    print(f"width {W:4d}: mean change rate {out[W][0]:.3f}  explained by r1 {out[W][1]:.3f}  by r2 {out[W][2]:.3f}")

# 02b minimisation
from sympy import symbols
from sympy.logic import SOPform
from sympy import count_ops
B = binarize(gray_frames(mp4, 64)[0]); clip = B[900:930]
lut, cnt = fit_majority_lut(clip, boundary=1)
seen = cnt.sum(1) > 0
ones = [i for i in range(512) if seen[i] and lut[i]]
dc = [i for i in range(512) if not seen[i]]
xs = symbols("x0:9")
t = time.perf_counter()
# sympy minterm bit order is MSB-first over the variable list; reverse so x_k == bit k
expr = SOPform(list(reversed(xs)), [[int(b) for b in format(i, "09b")] for i in ones],
               [[int(b) for b in format(i, "09b")] for i in dc])
dt = time.perf_counter() - t
from sympy import lambdify
f = lambdify(xs, expr, "numpy")
idx = np.arange(512)
lut_min = np.asarray(f(*[(idx >> k) & 1 for k in range(9)]), dtype=np.uint8) * np.ones(512, np.uint8)
assert (lut_min[seen] == lut[seen]).all(), "minimised rule must agree on every observed context"
p0, p1 = rollout(clip[0], lut, 30, boundary=1), rollout(clip[0], lut_min, 30, boundary=1)
ex = export_lut(os.path.join(os.path.dirname(__file__), "..", "runs", "exp01", "ba_t30s_30f_min.json"), lut_min, clip[0], 30, boundary=1,
                meta=dict(family="moore-r1-lut", fit="majority+SOP-dontcare-min", sop=str(expr)), expected=p1)
js = json.loads(subprocess.run(["node", "player/replay.mjs", "runs/exp01/ba_t30s_30f_min.json"], capture_output=True, text=True, check=True).stdout)
print("seen contexts", int(seen.sum()), "ones", len(ones), "dontcares", len(dc))
print("SOP:", expr, "\nops", count_ops(expr), "terms", len(expr.args) if expr.func.__name__ == "Or" else 1, f"min time {dt:.1f}s")
print("rollout full-LUT", metrics(p0, clip)["pixel_err"], "minimised", metrics(p1, clip)["pixel_err"], "js match", js["sha256"] == ex["expected_sha256"])
