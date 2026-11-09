"""Multi-channel binary CA rule = explicit input taps + one LUT per output channel.

State: (C,H,W) uint8, channel 0 is the visible/observed channel, 1..C-1 hidden.
taps: list of (channel, dy, dx); tap k contributes bit k of the LUT index.
luts: (C, 2**len(taps)) uint8; luts[c][idx] = next value of channel c.
Boundary: constant 0/1 for every channel outside the grid, or "wrap".
"""
import numpy as np

MOORE = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]


def family_taps(hidden, hidden_seen_by="self"):
    """Primary family F_priv(k): visible Moore 3x3 (bits 0..8) + the cell's own hidden bits.
    hidden_seen_by='moore' gives the larger F_shared(k) (neighbours see hidden bits too)."""
    taps = [(0, dy, dx) for dy, dx in MOORE]
    for c in range(1, hidden + 1):
        taps += [(c, 0, 0)] if hidden_seen_by == "self" else [(c, dy, dx) for dy, dx in MOORE]
    return taps


def index(state, taps, boundary=0):
    C, H, W = state.shape
    p = np.pad(state, ((0, 0), (1, 1), (1, 1)), mode="wrap") if boundary == "wrap" else \
        np.pad(state, ((0, 0), (1, 1), (1, 1)), constant_values=boundary)
    idx = np.zeros((H, W), np.int64)
    for k, (c, dy, dx) in enumerate(taps):
        idx |= p[c, 1 + dy:1 + dy + H, 1 + dx:1 + dx + W].astype(np.int64) << k
    return idx


def rollout(visible0, taps, luts, n_frames, steps_per_frame=1, boundary=0, hidden_init=0):
    """Autonomous run. Returns (visible frames (T,H,W), states at frame times (T,C,H,W),
    sorted LUT indices consulted -- every other entry is a don't-care for this rollout)."""
    s = np.full((luts.shape[0],) + visible0.shape, hidden_init, np.uint8); s[0] = visible0
    states, seen = [s], set()
    for _ in range(n_frames - 1):
        for _ in range(steps_per_frame):
            idx = index(s, taps, boundary)
            seen.update(np.unique(idx).tolist())
            s = luts[:, idx].astype(np.uint8)
        states.append(s)
    st = np.stack(states)
    return st[:, 0], st, sorted(seen)
