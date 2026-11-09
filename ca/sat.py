"""Exact CA synthesis as SAT: find LUTs + all unobserved bits such that the autonomous run
from frame 0 (hidden = hidden_init) reproduces every observed frame exactly.

Variables: L[c][e] (LUT entry e of output channel c) and every state bit that is not
pinned (hidden channels; visible channel at generations between observed frames).
For each cell/generation and each assignment of its unknown context bits:
    (context == e) -> (next_c == L[c][e]).
Clause count grows as 2^(#unknown context bits) per cell-generation.
"""
import itertools
import numpy as np


def encode(frames, taps, channels, steps_per_frame=1, boundary=0, hidden_init=0, sink=None):
    """Returns (clauses, L, nvars). With sink=callable, each clause is passed to it instead of
    being stored (clauses is then the clause count) -- long clips do not fit as Python lists."""
    e = Encoder(taps, channels, steps_per_frame, boundary, hidden_init, sink)
    e.extend(frames)
    return (e.clauses if sink is None else e.n_cl), e.L, e.nv


class Encoder:
    """Incremental form of `encode`: extend(frames) adds the clauses for generations not yet
    encoded, reusing every LUT/state variable already created. Clauses for generation g only
    involve states g and g+1, and frame-time visible bits stay pinned, so earlier clauses remain
    valid when frames are appended (same variables and clauses as one cold `encode` of the clip)."""

    def __init__(self, taps, channels, steps_per_frame=1, boundary=0, hidden_init=0, sink=None):
        self.taps, self.C, self.S, self.boundary, self.hidden_init = taps, channels, steps_per_frame, boundary, hidden_init
        self.nv, self.L, self.X, self.g, self.n_cl = 0, {}, {}, 0, 0
        self.clauses = []
        self.add = sink if sink is not None else self.clauses.append

    def _new(self):
        self.nv += 1; return self.nv

    def lut(self, c, e):
        if (c, e) not in self.L: self.L[c, e] = self._new()
        return self.L[c, e]

    def _val(self, frames, g, c, y, x):  # -> ('c', bit) constant or ('v', var)
        T, H, W = frames.shape
        S = self.S
        if not (0 <= y < H and 0 <= x < W):
            if self.boundary == "wrap": y, x = y % H, x % W
            else: return ("c", self.boundary)
        if c == 0 and g % S == 0: return ("c", int(frames[g // S, y, x]))
        if c > 0 and g == 0: return ("c", self.hidden_init)
        if (g, c, y, x) not in self.X: self.X[g, c, y, x] = self._new()
        return ("v", self.X[g, c, y, x])

    def extend(self, frames):
        T, H, W = frames.shape
        val = lambda *a: self._val(frames, *a)
        add = self.add
        for g in range(self.g, (T - 1) * self.S):
            for y in range(H):
                for x in range(W):
                    ctx = [val(g, c, y + dy, x + dx) for c, dy, dx in self.taps]
                    base = sum(b << k for k, (t, b) in enumerate(ctx) if t == "c")
                    unk = [(k, v) for k, (t, v) in enumerate(ctx) if t == "v"]
                    outs = [val(g + 1, c, y, x) for c in range(self.C)]
                    for bits in itertools.product((0, 1), repeat=len(unk)):
                        e = base + sum(b << k for (k, _), b in zip(unk, bits))
                        neg = [(-v if b else v) for (_, v), b in zip(unk, bits)]
                        for c, (t, o) in enumerate(outs):
                            l = self.lut(c, e)
                            if t == "c":
                                add(neg + [l if o else -l]); self.n_cl += 1
                            else:
                                add(neg + [-l, o]); add(neg + [l, -o]); self.n_cl += 2
        self.g = max(self.g, (T - 1) * self.S)
        return self


def luts_from_model(model, L, channels, n_taps):
    pos = set(v for v in model if v > 0)
    luts = np.zeros((channels, 2 ** n_taps), np.uint8)  # entries absent from the CNF are never consulted -> 0
    for (c, e), v in L.items():
        luts[c, e] = v in pos
    return luts


def history_conflicts(frames, boundary=0):
    """F_priv(k), 1 step/frame, ANY k: a cell's hidden state is a function of its own observed
    3x3 history, so two cells with identical histories (frames 0..t) but different values at
    t+1 make the clip infeasible for every k. Returns (#conflicting cell-times, first (t, n_classes))."""
    from ca.core import neigh_index
    T = len(frames)
    hist = np.zeros(frames[0].size, np.int64)
    bad, first = 0, None
    for t in range(T - 1):
        idx = neigh_index(frames[t], 1, boundary).ravel()
        _, hist = np.unique(np.stack([hist, idx], 1), axis=0, return_inverse=True)
        hist = hist.ravel()
        nxt = frames[t + 1].ravel().astype(np.int64)
        n_cls = hist.max() + 1
        on = np.bincount(hist, nxt, n_cls); tot = np.bincount(hist, None, n_cls)
        mixed = (on > 0) & (on < tot)
        k = int(np.minimum(on, tot - on)[mixed].sum())
        if k and first is None:
            first = (t, int(n_cls))
        bad += k
    return bad, first


def light_cone_conflicts(frames, S, boundary=0):
    """Any radius-1 homogeneous rule, ANY hidden channels/visibility, constant hidden init:
    frame t at a cell is a function of frame 0 within Chebyshev radius t*S. Cells whose
    radius-(t*S) frame-0 patches are identical must agree at frame t.
    Returns list of (t, #conflicting cells) with conflicts (empty -> condition passes)."""
    T, H, W = frames.shape
    out = []
    for t in range(1, T):
        R = t * S
        if R >= max(H, W):  # every patch sees the boundary differently -> all cells distinct
            break
        p = np.pad(frames[0], R, constant_values=boundary) if boundary != "wrap" else np.pad(frames[0], R, mode="wrap")
        patches = np.lib.stride_tricks.sliding_window_view(p, (2 * R + 1, 2 * R + 1)).reshape(H * W, -1)
        _, cls = np.unique(patches, axis=0, return_inverse=True)
        cls = cls.ravel(); v = frames[t].ravel().astype(np.int64)
        on = np.bincount(cls, v); tot = np.bincount(cls)
        bad = int(np.minimum(on, tot - on).sum())
        if bad:
            out.append((t, bad))
    return out


def state_repeat_conflict(frames):
    """Memoryless (visible-only) proof of infeasibility for ANY steps/frame: identical full
    states at frames i<j with different successors. Returns (i, j) or None."""
    T = len(frames)
    for i in range(T - 1):
        for j in range(i + 1, T - 1):
            if (frames[i] == frames[j]).all() and not (frames[i + 1] == frames[j + 1]).all():
                return i, j
    return None
