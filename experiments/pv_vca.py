"""Path V: velocity-carrying CA (VCA), a hand-written reference rule (NOT learned). NumPy, batched over a leading axis.

State per cell (all integers):
  fg      1 bit   foreground (visible = fg XOR background colour; background colour fixed per run from the start frame)
  vy, vx  in [-Q, Q]   velocity in 1/Q px per frame (|component| <= 1 px/frame)
  g       in [-Q, Q]   growth rate in 1/Q boundary layers per frame (> 0 grow, < 0 shrink)
  py, px, pg in [0, Q) sub-pixel phase accumulators
Background cells carry no motion (all zero). Q = 4.

One frame = one synchronous step, radius 1:
  1. phase: t = p + v; step s = floor(t / Q) in {-1, 0, 1}; p = t - s Q   (fg cells only). With p = Q/2 at an event the
     displacement after n frames is floor((Q/2 + n v) / Q): exact average speed v/Q; +v and -v step on alternating frames.
  2. transport (gather): cell x becomes the first candidate, in the fixed order
       resident staying (s = 0), then movers whose step is N, S, W, E, NW, NE, SW, SE and lands on x;
     no candidate -> background (a mover leaves a hole unless something moves in). Other movers into the same cell
     are absorbed (mass loss at collisions, documented). Nothing enters from outside the grid; movers off the grid vanish.
  3. growth: t = pg + g; o = floor(t / Q). o = +1: each background 4-neighbour of the cell becomes fg and copies the
     state of the first such parent (parent below, above, right, left). o = -1: the cell becomes background if it has a background
     4-neighbour (boundary erosion).
Control: a regional code (vy, vx, g) per region of a coarse grid, applied at events.
  lagrangian (default): at an event every fg cell inside region r takes code r, with phases reset to Q/2; between
             events the state travels with the cells (velocity is carried).
  eulerian (baseline): every frame every fg cell takes the code of the region it is in now, and the phases are one
             shared counter per region (velocity belongs to the grid, not the cells).
usage: imported by pv_v0.py and tests/test_pv_vca.py"""
import numpy as np

Q = 4
ORDER = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]   # (dy, dx) of the step that lands here
FIELDS = ("vy", "vx", "g", "py", "px", "pg")


def sh(a, dy, dx, fill=0):
    """out[..., y, x] = a[..., y - dy, x - dx] (zero / fill outside)."""
    H, W = a.shape[-2:]; o = np.full_like(a, fill)
    o[..., max(dy, 0):H + min(dy, 0), max(dx, 0):W + min(dx, 0)] = a[..., max(-dy, 0):H + min(-dy, 0), max(-dx, 0):W + min(-dx, 0)]
    return o


def region_map(grid, H=36, W=48):
    ys, xs = np.linspace(0, H, grid[0] + 1).astype(int), np.linspace(0, W, grid[1] + 1).astype(int); m = np.zeros((H, W), np.int64)
    for i in range(grid[0]):
        for j in range(grid[1]): m[ys[i]:ys[i + 1], xs[j]:xs[j + 1]] = i * grid[1] + j
    return m


def init_state(frame, B=1):
    bg = int(np.concatenate([frame[0], frame[-1], frame[:, 0], frame[:, -1]]).mean() >= 0.5)
    S = {"fg": np.repeat((frame != bg)[None], B, 0)}
    for f in FIELDS: S[f] = np.zeros(S["fg"].shape, np.int16)
    S["bg"] = bg; return S


def visible(S): return (S["fg"] ^ bool(S["bg"])).astype(np.uint8)


def apply_codes(S, codes, rmap, mask=None):
    """Lagrangian event. codes: (B, R, 3) int (vy, vx, g). Every fg cell takes its region's code; phases reset to Q/2."""
    c = np.asarray(codes)[:, rmap]                                  # (B, H, W, 3)
    fg = S["fg"] if mask is None else S["fg"] & mask
    for i, f in enumerate(("vy", "vx", "g")): S[f] = np.where(fg, c[..., i], S[f]).astype(np.int16)
    for f in ("py", "px", "pg"): S[f] = np.where(fg, Q // 2, S[f]).astype(np.int16)


def step(S, euler=None):
    """One frame. euler = (codes (B, R, 3), rmap, phase dict) for the Eulerian baseline; None = Lagrangian."""
    fg = S["fg"]
    if euler is not None:
        codes, rmap, ph = euler; c = np.asarray(codes)
        for i, (f, p) in enumerate((("vy", "py"), ("vx", "px"), ("g", "pg"))):
            t = ph[p] + c[..., i]; s = np.floor_divide(t, Q); ph[p] = t - s * Q          # per-region shared counters (B, R)
            S[f] = np.where(fg, c[:, rmap, i], 0).astype(np.int16); S["_s" + f] = s[:, rmap]
        sy, sx, og_grid = np.where(fg, S.pop("_svy"), 0), np.where(fg, S.pop("_svx"), 0), S.pop("_sg")
    else:
        ty, tx = S["py"] + S["vy"], S["px"] + S["vx"]; sy, sx = np.floor_divide(ty, Q), np.floor_divide(tx, Q)
        S["py"], S["px"] = np.where(fg, ty - sy * Q, 0).astype(np.int16), np.where(fg, tx - sx * Q, 0).astype(np.int16)
        sy, sx = np.where(fg, sy, 0), np.where(fg, sx, 0)
    # transport
    new = np.zeros_like(fg); took = np.zeros_like(fg); out = {f: np.zeros_like(S[f]) for f in FIELDS}
    for dy, dx in ORDER:
        cand = sh(fg & (sy == dy) & (sx == dx), dy, dx) & ~took
        for f in FIELDS: out[f] = np.where(cand, sh(S[f], dy, dx), out[f])
        new |= cand; took |= cand
    fg = new; S.update(out); S["fg"] = fg
    # growth / shrink
    if euler is None:
        t = S["pg"] + S["g"]; og = np.where(fg, np.floor_divide(t, Q), 0); S["pg"] = np.where(fg, t - np.floor_divide(t, Q) * Q, 0).astype(np.int16)
    else:
        og = np.where(fg, og_grid, 0)                                     # the region's growth step where the cell is now
    if (og != 0).any():
        bgn = np.zeros_like(fg)
        for dy, dx in ORDER[1:5]: bgn |= ~sh(fg, dy, dx, fill=True)            # has a background 4-neighbour (outside counts as fg)
        die = fg & (og < 0) & bgn
        grow = fg & (og > 0); born = np.zeros_like(fg); st = {f: np.zeros_like(S[f]) for f in FIELDS}
        for dy, dx in ORDER[1:5]:
            cand = sh(grow, dy, dx) & ~fg & ~born
            for f in FIELDS: st[f] = np.where(cand, sh(S[f], dy, dx), st[f])
            born |= cand
        for f in FIELDS: S[f] = np.where(born, st[f], np.where(die, 0, S[f])).astype(np.int16)
        S["fg"] = (fg | born) & ~die
    return S


def run(frame, n, events, rmap, mode="lagrangian", B=1):
    """Roll n frames from a start frame. events: {t: codes (B, R, 3)} applied before step t (t = 0 .. n-1).
    Eulerian: the latest event's codes are the field. Returns visible frames (B, n, H, W)."""
    S = init_state(frame, B); out = []; cur = None; R = rmap.max() + 1
    ph = {p: np.full((B, R), Q // 2, np.int16) for p in ("py", "px", "pg")}
    for t in range(n):
        if t in events:
            cur = np.asarray(events[t])
            if mode == "lagrangian": apply_codes(S, cur, rmap)
            else: ph = {p: np.full((B, R), Q // 2, np.int16) for p in ("py", "px", "pg")}
        step(S, None if mode == "lagrangian" else (cur if cur is not None else np.zeros((B, R, 3), int), rmap, ph))
        out.append(visible(S))
    return np.stack(out, 1)


# ---------------------------------------------------------------- whole-image block-shift baseline ----------------------------------------------------------------
def block_shift(frame, n, events, rmap):
    """Every pixel (background included) pulls from x - d(region of x) each frame, integer d per region (Eulerian
    piecewise shift); pixels pulled from outside take the background colour. events: {t: (B, R, 2)}."""
    bg = int(np.concatenate([frame[0], frame[-1], frame[:, 0], frame[:, -1]]).mean() >= 0.5)
    B = len(next(iter(events.values()))); v = np.repeat(frame[None], B, 0).astype(np.uint8); out = []; d = None
    for t in range(n):
        if t in events: d = np.asarray(events[t])[:, rmap]           # (B, H, W, 2)
        nv = np.full_like(v, bg)
        for dy in np.unique(d[..., 0]):
            for dx in np.unique(d[..., 1]):
                m = (d[..., 0] == dy) & (d[..., 1] == dx)
                if m.any(): nv = np.where(m, sh(v, int(dy), int(dx), fill=bg), nv)
        v = nv; out.append(v.copy())
    return np.stack(out, 1)
