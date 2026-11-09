"""Exp01: exact-LUT rule search baselines + independent replay.

Hypothesis: for 1 generation/frame and no hidden state, the best rule in a LUT family is
determined by majority vote over observed transitions (no gradient search needed); its
teacher-forced error is a hard floor, and autonomous rollout error will be far above it.
Budget: CPU only, seconds per clip.
"""
import hashlib, json, os, subprocess, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ca.core import (bs_classes, conflict_report, fit_majority_lut, life_like_lut, metrics,
                     neigh_index, rollout, transition_counts)
from ca.export import export_lut
from ca.video import binarize, gray_frames

OUT = os.path.join(os.path.dirname(__file__), "..", "runs", "exp01")
os.makedirs(OUT, exist_ok=True)
CONWAY = life_like_lut({3}, {2, 3})
R = {}


def replay_check(name, lut, seed, n, r=1, boundary=0, expected=None, meta=None):
    path = os.path.join(OUT, f"{name}.json")
    ex = export_lut(path, lut, seed, n, r, boundary, meta=meta, expected=expected)
    out = json.loads(subprocess.run(["node", os.path.join(os.path.dirname(__file__), "..", "player", "replay.mjs"), path],
                                    capture_output=True, text=True, check=True).stdout)
    return out["sha256"] == ex["expected_sha256"]


def naive_life(g):  # independent per-cell reference, wrap boundary
    H, W = g.shape; o = np.zeros_like(g)
    for y in range(H):
        for x in range(W):
            n = sum(g[(y + dy) % H, (x + dx) % W] for dy in (-1, 0, 1) for dx in (-1, 0, 1)) - g[y, x]
            o[y, x] = 1 if (n == 3 or (g[y, x] and n == 2)) else 0
    return o


# ---------- 1. Conway correctness ----------
rng = np.random.default_rng(0)
soup = (rng.random((32, 32)) < 0.35).astype(np.uint8)
traj = rollout(soup, CONWAY, 30, boundary="wrap")
ref = [soup]
for _ in range(29):
    ref.append(naive_life(ref[-1]))
glider = np.zeros((16, 16), np.uint8); glider[1, 2] = glider[2, 3] = glider[3, 1] = glider[3, 2] = glider[3, 3] = 1
g4 = rollout(glider, CONWAY, 5, boundary="wrap")[-1]
R["conway_correctness"] = dict(
    lut_vs_naive_30steps=bool((traj == np.stack(ref)).all()),
    glider_translates_1_1_after_4=bool((np.roll(glider, (1, 1), (0, 1)) == g4).all()),
    js_replay_matches=replay_check("conway_soup", CONWAY, soup, 30, boundary="wrap", expected=traj))

# ---------- 2. Recover Conway from its own trajectory ----------
big = rollout((rng.random((64, 64)) < 0.35).astype(np.uint8), CONWAY, 20, boundary="wrap")
lut, counts = fit_majority_lut(big, boundary="wrap")
seen = counts.sum(1) > 0
bsc, _ = transition_counts(big, boundary="wrap", classes=bs_classes())
R["conway_recovery"] = dict(
    full_lut=conflict_report(counts),
    full_lut_agrees_on_seen=bool((lut[seen] == CONWAY[seen]).all()),
    unseen_entries=int((~seen).sum()),
    bs_family_recovered=sorted(f"{'S' if k // 9 else 'B'}{k % 9}" for k in range(18) if bsc[k, 1] > bsc[k, 0]),
    rollout_exact=bool((rollout(big[0], lut, 20, boundary="wrap") == big).all()))


# ---------- helpers for target sequences ----------
def evaluate(name, frames, boundary=0, export=True):
    n = len(frames); res = {}
    hold = np.repeat(frames[:1], n, 0)
    res["hold_first_frame"] = metrics(hold, frames)
    res["conway"] = metrics(rollout(frames[0], CONWAY, n, boundary=boundary), frames)
    # Life-like (outer totalistic) best fit: 18 classes
    cls = bs_classes()
    c18, _ = transition_counts(frames, boundary=boundary, classes=cls)
    bs_lut = np.where(c18[cls, 1] > c18[cls, 0], 1, np.where(c18[cls, 1] == c18[cls, 0], (np.arange(512) >> 4) & 1, 0)).astype(np.uint8)
    res["bs_family"] = dict(conflicts=conflict_report(c18), **metrics(rollout(frames[0], bs_lut, n, boundary=boundary), frames))
    t = time.perf_counter()
    lut, cnt = fit_majority_lut(frames, boundary=boundary)
    fit_s = time.perf_counter() - t
    pred = rollout(frames[0], lut, n, boundary=boundary)
    res["moore_lut"] = dict(conflicts=conflict_report(cnt), fit_seconds=round(fit_s, 4), **metrics(pred, frames))
    c25, _ = transition_counts(frames, r=2, boundary=boundary)
    res["r2_lut_diag"] = conflict_report(c25)
    if export:
        res["moore_lut"]["js_replay_matches"] = replay_check(name, lut, frames[0], n, boundary=boundary, expected=pred,
                                                             meta=dict(source=name, family="moore-r1-lut", fit="majority"))
    return res, lut, pred


# ---------- 3. Synthetic motion ----------
from ca.synthetic import grow_shrink, rotate_bar, translate

R["synthetic"] = {k: evaluate(f"syn_{k}", f())[0] for k, f in
                  [("translate", translate), ("grow_shrink", grow_shrink), ("rotate_bar", rotate_bar)]}

# ---------- 4. Bad Apple clips, 64x48, threshold 128 ----------
gray, ts = gray_frames(os.path.join(os.path.dirname(__file__), "..", "refs", "Bad-Apple-Starter", "BadApple!!.mp4"), 64)
B = binarize(gray, 128)
chg = (B[1:] != B[:-1]).mean((1, 2))
R["badapple"] = {}
for start_s in [10, 30, 62, 100, 150, 200]:
    for L in [30, 90]:
        s = int(round(start_s * 30)); clip = B[s:s + L]
        key = f"ba_t{start_s}s_{L}f"
        uniform0 = clip[0].min() == clip[0].max()
        best = None
        per_boundary = {}
        for bd in (0, 1, "wrap"):
            res, lut, pred = evaluate(f"{key}_b{bd}", clip, boundary=bd, export=(bd == 0))
            per_boundary[str(bd)] = res
        R["badapple"][key] = dict(t_start=float(ts[s]), frames=L, first_frame_uniform=bool(uniform0),
                                  mean_frac_changed_per_frame=float(chg[s:s + L - 1].mean()),
                                  white_frac=float(clip.mean()), by_boundary=per_boundary)

with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(R, f, indent=1)
print(json.dumps({k: R[k] for k in ("conway_correctness", "conway_recovery")}, indent=1))
