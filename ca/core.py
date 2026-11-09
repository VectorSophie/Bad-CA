"""Binary radius-r cellular automata as lookup tables, with conflict diagnostics.

Neighbourhood index convention (shared with player/replay.mjs):
offsets are enumerated row-major over dy in [-r..r], dx in [-r..r]; offset k
contributes bit k (LSB = top-left). For r=1 the centre cell is bit 4.
"""
import numpy as np


def offsets(r):
    return [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1)]


def neigh_index(grid, r=1, boundary=0):
    """grid: (H,W) uint8 {0,1}. boundary: 0/1 constant fill, or 'wrap'."""
    H, W = grid.shape
    if boundary == "wrap":
        p = np.pad(grid, r, mode="wrap")
    else:
        p = np.pad(grid, r, mode="constant", constant_values=boundary)
    idx = np.zeros((H, W), np.int64)
    for k, (dy, dx) in enumerate(offsets(r)):
        idx |= p[r + dy:r + dy + H, r + dx:r + dx + W].astype(np.int64) << k
    return idx


def step(grid, lut, r=1, boundary=0):
    return lut[neigh_index(grid, r, boundary)].astype(np.uint8)


def rollout(seed, lut, n_frames, steps_per_frame=1, r=1, boundary=0):
    out = [seed.astype(np.uint8)]
    g = out[0]
    for _ in range(n_frames - 1):
        for _ in range(steps_per_frame):
            g = step(g, lut, r, boundary)
        out.append(g)
    return np.stack(out)


def life_like_lut(birth, survive):
    """Outer-totalistic B/S rule as a 512-entry Moore LUT."""
    lut = np.zeros(512, np.uint8)
    for i in range(512):
        c = (i >> 4) & 1
        n = bin(i).count("1") - c
        lut[i] = (n in survive) if c else (n in birth)
    return lut


# ---- fitting / diagnostics (teacher-forced: one observed frame -> next) ----

def transition_counts(frames, r=1, boundary=0, classes=None):
    """counts[class, next_bit] over all consecutive-frame transitions.
    classes: optional array mapping neighbourhood index -> coarser class id."""
    idx = np.concatenate([neigh_index(f, r, boundary).ravel() for f in frames[:-1]])
    nxt = np.concatenate([f.ravel() for f in frames[1:]]).astype(np.int64)
    if classes is not None:
        idx = classes[idx]
    if r == 1 or classes is not None:
        n = int(idx.max()) + 1
        c = np.zeros((n, 2), np.int64)
        np.add.at(c, (idx, nxt), 1)
        return c, None
    keys, inv = np.unique(idx, return_inverse=True)  # sparse for large r
    c = np.zeros((len(keys), 2), np.int64)
    np.add.at(c, (inv, nxt), 1)
    return c, keys


def conflict_report(counts):
    """Irreducible one-step error of the best rule in this family (majority vote)."""
    seen = counts.sum(1) > 0
    both = (counts[:, 0] > 0) & (counts[:, 1] > 0)
    total = counts.sum()
    return dict(transitions=int(total), contexts_seen=int(seen.sum()),
                contexts_conflicting=int(both.sum()),
                min_one_step_errors=int(np.minimum(counts[:, 0], counts[:, 1]).sum()),
                min_one_step_error_rate=float(np.minimum(counts[:, 0], counts[:, 1]).sum() / total))


def fit_majority_lut(frames, r=1, boundary=0, unseen="keep_centre"):
    """Best teacher-forced LUT. Unseen contexts: keep the centre cell (a prior, not data)."""
    counts, keys = transition_counts(frames, r, boundary)
    assert keys is None, "dense LUT only for r=1"
    lut = (counts[:, 1] > counts[:, 0]).astype(np.uint8)
    unseen_mask = counts.sum(1) == 0
    centre = (np.arange(len(lut)) >> ((2 * r + 1) ** 2 // 2)) & 1
    lut[unseen_mask] = centre[unseen_mask] if unseen == "keep_centre" else 0
    tie = (counts[:, 0] == counts[:, 1]) & ~unseen_mask
    lut[tie] = centre[tie]
    return lut, counts


def bs_classes():
    i = np.arange(512)
    c = (i >> 4) & 1
    n = np.array([bin(x).count("1") for x in i]) - c
    return c * 9 + n  # 18 classes


# ---- metrics on autonomous rollouts ----

def metrics(pred, target):
    pred = pred.astype(bool); target = target.astype(bool)
    err = (pred != target).mean(axis=(1, 2))
    inter = (pred & target).sum(axis=(1, 2)); union = (pred | target).sum(axis=(1, 2))
    iou_fg = np.where(union > 0, inter / np.maximum(union, 1), 1.0)
    # background (black=0) IoU on the inverted image, so either polarity can't hide failure
    ib = (~pred & ~target).sum(axis=(1, 2)); ub = (~pred | ~target).sum(axis=(1, 2))
    iou_bg = np.where(ub > 0, ib / np.maximum(ub, 1), 1.0)
    # temporal: did we change the pixels that actually changed?
    dt = target[1:] ^ target[:-1]; dp = pred[1:] ^ pred[:-1]
    chg = (dt & dp).sum() / max(1, (dt | dp).sum())
    return dict(pixel_err=float(err.mean()), pixel_err_last=float(err[-1]),
                mean_iou_white=float(iou_fg.mean()), mean_iou_black=float(iou_bg.mean()),
                change_iou=float(chg))
