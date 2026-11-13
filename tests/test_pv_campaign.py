"""Path V campaign checks (atlas, metrics, oracle fits, replay integrity).
1. Atlas reproducibility: recomputing the first 40 atlas windows from the video gives the stored metrics and tags.
2. Metric correctness: holes() counts enclosed background components; a perfect prediction gives err 0, change F1 1,
   IoU 1; holding the start frame gives err == hold and change F1 0 (frame alignment of prediction vs target).
3. Keyframe baseline: frames at keyframe positions equal the truth.
4. Oracle fits are deterministic (same schedule and frames twice).
5. Replay integrity: running the plain runtime (pv_vca.run, no target frames) on the start frame plus the saved schedule
   reproduces the oracle fit's frames exactly, for lagrangian and eulerian modes and for a 3x3 grid.
6. The block-shift baseline's saved schedule replays exactly too."""
import json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
import pv_atlas, pv_v0, pv_vca as V

X = pv_v0.video()
# 1. atlas
A = json.load(open(os.path.join(pv_v0.OUT, "atlas.json")))["windows"]; B = pv_atlas.build(X[:40 * 16 + 1])
for a, b in zip(A[:40], B[:40]):
    for k in ("tag", "change", "g", "r", "m", "births", "deaths", "iou_first_last", "cut", "wipe"):
        assert (a[k] == b[k]) if not isinstance(a[k], float) else abs(a[k] - b[k]) < 1e-9, (a["start"], k)

# 2. metrics
f = np.zeros((36, 48), bool); f[5:15, 5:15] = True; f[8:10, 8:10] = False; assert pv_v0.holes(f) == 1 and pv_v0.holes(np.zeros((36, 48), bool)) == 0
s = 2304; F = X[s:s + pv_v0.N]
m = pv_v0.metrics(F[1:], F); assert m["err"] == 0 and m["f1"] == 1.0 and abs(m["iou"] - 1) < 1e-12
m = pv_v0.metrics(np.repeat(F[:1], len(F) - 1, 0), F); assert m["err"] == m["hold"] > 0 and m["f1"] == 0.0

# 3. keyframe
P, bits = pv_v0.keyframe(F, 8)
for k in (8, 16, 24, 31): assert (P[k - 1] == F[k]).all()
assert bits > 0

# 4. + 5. determinism and replay integrity
G = X[s:s + 17]
for mode, grid in (("L", (1, 1)), ("L", (3, 3)), ("E", (2, 2))):
    P1, s1 = pv_v0.fit_vca(G, grid, 8, mode=mode); P2, s2 = pv_v0.fit_vca(G, grid, 8, mode=mode)
    assert (P1 == P2).all() and s1 == s2
    ev = {int(t): np.array([c]) for t, c in s1.items()}
    Rp = V.run(G[0], len(G) - 1, ev, V.region_map(grid), mode="lagrangian" if mode == "L" else "eulerian")[0]
    assert (Rp == P1).all(), (mode, grid)

# 6. block replay
Pb, sb = pv_v0.fit_block(G, (2, 2), 8); cur = G[0]; out = []
for k, t0 in enumerate(sorted(map(int, sb))):
    t1 = sorted(map(int, sb))[k + 1] if k + 1 < len(sb) else len(G) - 1
    fr = V.block_shift(cur, t1 - t0, {0: np.array([sb[t0]])}, V.region_map((2, 2)))[0]; out.append(fr); cur = fr[-1]
assert (np.concatenate(out) == Pb).all()
print("test_pv_campaign: all checks passed")
