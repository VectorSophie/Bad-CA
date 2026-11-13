"""Path V velocity-carrying CA (experiments/pv_vca.py): synthetic sanity checks of the hand-written reference rule.
1. Translation: a random silhouette with velocity (vy, vx) is displaced by exactly floor((Q/2 + n v) / Q) px per axis
   after n frames (so (0, +Q) is 1 px/frame, (0, +1) is 1 px every Q frames); mass is conserved except what leaves the
   grid. Diagonal and negative speeds included (the floor makes +v and -v move on alternating frames: a convention).
2. Persistence: one event, then no events: the motion continues for 24 frames.
3. Opposing regional directions: left half moves west, right half moves east; mass conserved, the two parts separate.
4. Grid boundary: movers off the grid vanish; nothing enters.
5. Mid-trajectory event: east for 4 frames, then south; positions exact.
6. Collision: two blocks moving toward each other lose exactly the overlapping mass and the rest stays; deterministic.
7. Growth / shrink: g = +Q dilates one 4-neighbour layer per frame, g = -Q erodes one layer; g = +1 every Q frames.
8. Lagrangian vs Eulerian: an object crossing a region boundary keeps its velocity (lagrangian) but takes the new
   region's velocity (eulerian).
9. Batch rows equal separate runs; visible = fg XOR background colour for both polarities.
10. Block-shift baseline equals an integer image shift."""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
import pv_vca as V
Q = V.Q; rng = np.random.default_rng(0); R1 = V.region_map((1, 1))


def blob(y0, y1, x0, x1, bg=0):
    f = np.full((36, 48), bg, np.uint8); m = (rng.random((y1 - y0, x1 - x0)) < 0.7); m[0, :] = m[-1, :] = True
    f[y0:y1, x0:x1] = np.where(m, 1 - bg, bg); return f


def ev(code, R=1): return {0: np.array([[code] * R])}


# 1. translation at several speeds
f = blob(10, 20, 10, 20); fg0 = f.astype(bool)
disp = lambda v, n: (Q // 2 + n * v) // Q                            # displacement after n frames (phase starts at Q/2, floor)
for code in ((0, Q, 0), (0, 1, 0), (Q, -Q, 0), (-2, 0, 0), (0, -1, 0), (3, 2, 0)):
    out = V.run(f, 12, ev(code), R1)[0]
    for t in range(12):
        want = V.sh(fg0, disp(code[0], t + 1), disp(code[1], t + 1))
        assert (out[t].astype(bool) == want).all(), (code, t)
    assert out[-1].sum() == want.sum()                 # mass conserved except what left the grid

# 2. persistence (same as 1 over a longer horizon, no further events)
out = V.run(f, 24, ev((0, Q, 0)), R1)[0]; assert (out[-1].astype(bool) == V.sh(fg0, 0, 24)).all()

# 3. opposing regions (1 x 2 grid): left half west, right half east
g = np.zeros((36, 48), np.uint8); g[10:20, 16:32] = 1; R2 = V.region_map((1, 2))
out = V.run(g, 6, {0: np.array([[(0, -Q, 0), (0, Q, 0)]])}, R2)[0]
assert out[-1].sum() == g.sum() and (out[-1][10:20, 10:18] == 1).all() and (out[-1][10:20, 30:38] == 1).all() and (out[-1][10:20, 18:30] == 0).all()

# 4. boundary
h = np.zeros((36, 48), np.uint8); h[5:10, 44:48] = 1
out = V.run(h, 3, ev((0, Q, 0)), R1)[0]; assert out[0].sum() == 15 and out[2].sum() == 5 and out[2][5:10, 47].all()

# 5. mid-trajectory event
out = V.run(f, 8, {0: np.array([[(0, Q, 0)]]), 4: np.array([[(Q, 0, 0)]])}, R1)[0]
assert (out[-1].astype(bool) == V.sh(fg0, 4, 4)).all()

# 6. collision
c = np.zeros((36, 48), np.uint8); c[10:14, 16:20] = 1; c[10:14, 26:30] = 1; R2b = V.region_map((1, 2))
a1 = V.run(c, 6, {0: np.array([[(0, Q, 0), (0, -Q, 0)]])}, R2b)[0]; a2 = V.run(c, 6, {0: np.array([[(0, Q, 0), (0, -Q, 0)]])}, R2b)[0]
assert (a1 == a2).all() and a1[-1].sum() < c.sum() and a1[-1].sum() > 0
# blocks meet after 3 frames (gap 6 px closing at 2 px/frame); each further frame the overlap is absorbed
assert a1[2].sum() == 32 and a1[3].sum() < 32

# 7. growth / shrink
d = np.zeros((36, 48), np.uint8); d[15:21, 20:26] = 1
gr = V.run(d, 2, ev((0, 0, Q)), R1)[0]
from scipy import ndimage
cross = ndimage.generate_binary_structure(2, 1)
assert (gr[0].astype(bool) == ndimage.binary_dilation(d, cross)).all() and (gr[1].astype(bool) == ndimage.binary_dilation(d, cross, 2)).all()
sk = V.run(d, 1, ev((0, 0, -Q)), R1)[0]; assert (sk[0].astype(bool) == ndimage.binary_erosion(d, cross)).all()
slow = V.run(d, 8, ev((0, 0, 1)), R1)[0]; assert slow[0].sum() == d.sum() and slow[1].sum() > d.sum()   # phase Q/2: first layer after 2 frames

# 8. lagrangian vs eulerian across a region boundary (1 x 2 grid: left moves east, right is still)
e = np.zeros((36, 48), np.uint8); e[10:14, 18:22] = 1
lag = V.run(e, 8, {0: np.array([[(0, Q, 0), (0, 0, 0)]])}, R2)[0]; eul = V.run(e, 8, {0: np.array([[(0, Q, 0), (0, 0, 0)]])}, R2, mode="eulerian")[0]
assert lag[-1][10:14, 26:30].all()                       # kept moving 8 px
assert eul[-1][:, 24:].sum() < 16 and eul[-1].sum() <= 16  # stalls at the boundary (cells entering the still region stop)

# 9. batch and polarity
fw = blob(8, 18, 8, 22, bg=1); codes = np.array([[(0, Q, 0)], [(Q, 0, 0)]])
b = V.run(fw, 5, {0: codes}, R1, B=2)
for i in range(2): assert (b[i] == V.run(fw, 5, {0: codes[i:i + 1]}, R1)[0]).all()
assert b[0, -1].mean() > 0.5                              # white background preserved

# 10. block shift
bs = V.block_shift(f, 3, {0: np.array([[(1, -2)]])}, R1)[0]; assert (bs[2] == V.sh(f, 3, -6)).all()
print("test_pv_vca: all checks passed")
