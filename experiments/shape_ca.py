"""Track S: shape-aware CA candidates, hand-written (NOT learned), NumPy, batched over a leading axis like pv_vca.

S1  lsca   level-set CA. State: integer signed distance phi in [-D, D] (D = 6), visible = phi < 0. One frame, radius 1:
           (a) normal motion: accumulator pg += g (g in [-4, 4], quarter layers per frame); on overflow the outside
               cells at distance 1 become inside (grow) or the inside cells at distance 1 become outside (shrink);
           (b) re-distancing that never changes a cell's side: outside cells phi = min(phi, min over 4-neighbours + 1)
               in [1, D], inside cells phi = max(phi, max over 4-neighbours - 1) in [-D, -1] (two sweeps per frame).
           BUG FIXED 2026-11-13 (found by tests/test_gen_shape.py): the first version let boundary cells reach 0 and
           vanish, i.e. it eroded one layer per frame even at g = 0; runs/shape/*_v1_lsca_bug.json keep those results;
           (c) optional smoothing 'k' (curvature-flow proxy): a cell flips sign if >= 6 of its 8 neighbours have the
               other sign. Control per region of a grid: (g, smooth on/off).
S2  ffdca  structural control + local transport. A coarse lattice of (gy+1) x (gx+1) control NODES carries velocities
           (vy, vx) in [-4, 4] quarter-px/frame. Each cell's velocity is the bilinear interpolation of its 4 surrounding
           nodes (this read has radius = lattice spacing, i.e. NOT radius 1: the hybrid, non-local part). The cell then
           behaves like a radius-1 CA: phase accumulators, and on overflow it PULLS the value of the neighbour opposite
           to its step (every cell pulls exactly one value: no holes or collisions by construction, but compression
           duplicates and stretching drops pixels). Optional local cleanup: a cell whose 8-neighbourhood holds >= 7 of
           the other colour flips (removes 1-px specks and pinholes).
     ffdwarp  geometric reference (not a CA): each frame is the event frame warped by the accumulated interpolated
           displacement (nearest-neighbour backward warp from the event frame), so there is no per-cell drift.
Morphology baseline: per event one of {hold, dilate, erode} (4-neighbour, once per frame).
usage: imported; shape_s.py runs the experiments"""
import numpy as np
import pv_vca as V

Q, D = 4, 6


def sh(a, dy, dx, fill=0): return V.sh(a, dy, dx, fill)


# ---------------------------------------------------------------- S1 level-set CA ----------------------------------------------------------------
def to_phi(fg):
    """Exact clamped city-block signed distance from a binary mask (used once at the start)."""
    from scipy import ndimage
    din = ndimage.distance_transform_cdt(fg, metric="taxicab"); dout = ndimage.distance_transform_cdt(~fg, metric="taxicab")
    return np.clip(np.where(fg, -din, dout), -D, D).astype(np.int16)


def redistance(phi, sweeps=2):
    """Local re-distancing that never changes a cell's side: outside stays in [1, D], inside in [-D, -1]."""
    for _ in range(sweeps):
        nb = [sh(phi, dy, dx, fill=D) for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))]
        mn, mx = np.minimum.reduce(nb), np.maximum.reduce(nb)
        phi = np.where(phi > 0, np.clip(np.minimum(phi, mn + 1), 1, D), np.clip(np.maximum(phi, mx - 1), -D, -1))
    return phi.astype(np.int16)


def lsca_run(frame, n, events, rmap, B=1):
    """events: {t: (B, R, 2)} codes (g, smooth). Returns visible frames (B, n, H, W)."""
    bg = int(np.concatenate([frame[0], frame[-1], frame[:, 0], frame[:, -1]]).mean() >= 0.5)
    phi = np.repeat(to_phi(frame != bg)[None], B, 0); pg = np.full(phi.shape, Q // 2, np.int16); out = []; c = None
    for t in range(n):
        if t in events: c = np.asarray(events[t])[:, rmap]; pg[:] = Q // 2
        g = c[..., 0] if c is not None else 0; tt = pg + g; o = np.floor_divide(tt, Q); pg = (tt - o * Q).astype(np.int16)
        phi = np.where((o > 0) & (phi == 1), -1, np.where((o < 0) & (phi == -1), 1, phi)).astype(np.int16)   # the zero level moves one cell
        phi = redistance(phi)
        if c is not None and (c[..., 1] > 0).any():
            neg = phi < 0; cnt = sum(sh(neg, dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)
            flip_in = (~neg) & (cnt >= 6) & (c[..., 1] > 0); flip_out = neg & (cnt <= 2) & (c[..., 1] > 0)
            phi = np.where(flip_in, -1, np.where(flip_out, 1, phi)).astype(np.int16); phi = redistance(phi)
        out.append(((phi < 0) ^ bool(bg)).astype(np.uint8))
    return np.stack(out, 1)


# ---------------------------------------------------------------- S2 lattice-controlled transport ----------------------------------------------------------------
def interp_field(nodes, H=36, W=48):
    """nodes: (B, gy+1, gx+1, 2) -> per-cell field (B, H, W, 2), bilinear between lattice nodes spanning the frame."""
    B, ny, nx, _ = nodes.shape; y = np.linspace(0, ny - 1, H); x = np.linspace(0, nx - 1, W)
    y0 = np.minimum(np.floor(y).astype(int), ny - 2); x0 = np.minimum(np.floor(x).astype(int), nx - 2); wy = (y - y0)[:, None, None]; wx = (x - x0)[None, :, None]
    n = nodes
    a = n[:, y0][:, :, x0]; b = n[:, y0][:, :, x0 + 1]; c = n[:, y0 + 1][:, :, x0]; d = n[:, y0 + 1][:, :, x0 + 1]
    return (1 - wy) * ((1 - wx) * a + wx * b) + wy * ((1 - wx) * c + wx * d)


def ffdca_run(frame, n, events, grid, cleanup=True, B=1):
    """events: {t: (B, gy+1, gx+1, 2) node velocities}. Every cell pulls; whole image (background included) moves."""
    v = np.repeat(frame[None], B, 0).astype(np.uint8); H, W = frame.shape; py = np.full((B, H, W), Q // 2, np.int16); px = py.copy(); out = []; fld = None
    for t in range(n):
        if t in events: fld = np.rint(interp_field(np.asarray(events[t], float), H, W)).astype(np.int16); py[:] = Q // 2; px[:] = Q // 2
        if fld is not None:
            ty, tx = py + fld[..., 0], px + fld[..., 1]; sy, sx = np.floor_divide(ty, Q), np.floor_divide(tx, Q); py, px = (ty - sy * Q).astype(np.int16), (tx - sx * Q).astype(np.int16)
            nv = v.copy()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    m = (sy == dy) & (sx == dx)
                    if m.any() and (dy or dx): nv = np.where(m, sh(v, dy, dx, fill=int(np.median(frame[0]))), nv)   # pull from x - step
            v = nv
        if cleanup:
            cnt = sum(sh(v, dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)
            v = np.where((v == 0) & (cnt >= 7), 1, np.where((v == 1) & (cnt <= 1), 0, v)).astype(np.uint8)
        out.append(v.copy())
    return np.stack(out, 1)


def ffdwarp_run(frame, n, events, grid, B=1):
    """Geometric reference: backward nearest warp of the latest event frame by the accumulated displacement."""
    H, W = frame.shape; yy, xx = np.mgrid[0:H, 0:W]; src = np.repeat(frame[None], B, 0); out = []; fld = None; k = 0; bg = int(np.median(frame[0]))
    for t in range(n):
        if t in events:
            if t > 0: src = out[-1].copy()
            fld = interp_field(np.asarray(events[t], float), H, W) / Q; k = 0
        k += 1
        if fld is None: out.append(src.copy()); continue
        sy = np.rint(yy[None] - k * fld[..., 0]).astype(int); sx = np.rint(xx[None] - k * fld[..., 1]).astype(int)
        ok = (sy >= 0) & (sy < H) & (sx >= 0) & (sx < W); bi = np.arange(len(src))[:, None, None]
        out.append(np.where(ok, src[bi, np.clip(sy, 0, H - 1), np.clip(sx, 0, W - 1)], bg).astype(np.uint8))
    return np.stack(out, 1)


def morph_run(frame, n, events, rmap, B=1):
    """events: {t: (B, R, 1)} op per region: 0 hold, 1 dilate, 2 erode (on the foreground, once per frame)."""
    from scipy import ndimage
    bg = int(np.concatenate([frame[0], frame[-1], frame[:, 0], frame[:, -1]]).mean() >= 0.5); cr = ndimage.generate_binary_structure(2, 1)
    fg = np.repeat((frame != bg)[None], B, 0); out = []; c = None
    for t in range(n):
        if t in events: c = np.asarray(events[t])[:, rmap, 0]
        if c is not None:
            dil = np.stack([ndimage.binary_dilation(f, cr) for f in fg]); ero = np.stack([ndimage.binary_erosion(f, cr, border_value=0) for f in fg])
            fg = np.where(c == 1, dil, np.where(c == 2, ero, fg))
        out.append((fg ^ bool(bg)).astype(np.uint8))
    return np.stack(out, 1)
