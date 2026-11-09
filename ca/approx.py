"""Phase 07 evaluation of a replay `P` against target `F` (both (T,H,W) uint8, frame 0 identical).

All per-frame metrics are over frames 1..T-1. No single combined score.
  err[t]        pixel error
  hold[t]       error of freezing frame 0
  ratio[t]      err / hold (inf-safe)
  change_iou    |pred change ∩ true change| / |pred change ∪ true change|, change = differs from previous frame
  tol_space[t]  pixel counts as wrong only if no pixel of the predicted value lies in the target's 3x3 window
  tol_time[t]   min over target frames t-1, t, t+1 of the pixel error
  horizon       first frame (>=1) whose error exceeds the hold error at that frame (T if never)
Tolerant metrics can hide a systematic 1-px / 1-frame lag; the plain error is always reported with them.
"""
import numpy as np


def _dilate_has(F, v):
    """For each pixel: does the 3x3 window of F contain value v?"""
    T, H, W = F.shape
    p = np.pad(F == v, ((0, 0), (1, 1), (1, 1)))
    out = np.zeros((T, H, W), bool)
    for dy in range(3):
        for dx in range(3):
            out |= p[:, dy:dy + H, dx:dx + W]
    return out


def evaluate(P, F):
    P = P.astype(np.uint8); F = F.astype(np.uint8); T = len(F)
    assert P.shape == F.shape and (P[0] == F[0]).all()
    err = (P[1:] != F[1:]).mean((1, 2)); hold = (F[1:] != F[:1]).mean((1, 2))
    pc = P[1:] != P[:-1]; tc = F[1:] != F[:-1]
    inter, union = (pc & tc).sum(), (pc | tc).sum()
    has1, has0 = _dilate_has(F, 1), _dilate_has(F, 0)
    tol_ok = np.where(P == 1, has1, has0)
    tol_space = (~tol_ok[1:]).mean((1, 2))
    tol_time = np.array([min((P[t] != F[u]).mean() for u in range(max(0, t - 1), min(T, t + 2))) for t in range(1, T)])
    worse = np.nonzero(err > hold)[0]
    return dict(
        T=T, mean_err=float(err.mean()), mean_hold=float(hold.mean()), mean_ratio_to_hold=float(err.mean() / max(hold.mean(), 1e-12)),
        change_iou=float(inter / union) if union else 1.0, pred_changes=int(pc.sum()), true_changes=int(tc.sum()),
        mean_tol_space=float(tol_space.mean()), mean_tol_time=float(tol_time.mean()),
        horizon=int(worse[0] + 1) if len(worse) else T, exact=bool((P == F).all()),
        final_err=float(err[-1]), final_hold=float(hold[-1]),
        err_curve=[round(float(x), 5) for x in err], hold_curve=[round(float(x), 5) for x in hold],
        tol_space_curve=[round(float(x), 5) for x in tol_space])


def triptych_gif(P, F, path, scale=6, ms=100):
    """original | replay | diff (red = target white missing, blue = extra white, else grey/black/white)."""
    from PIL import Image
    T, H, W = F.shape; frames = []
    for t in range(T):
        f, p = F[t].astype(bool), P[t].astype(bool)
        a = np.stack([f * 255] * 3, -1).astype(np.uint8); b = np.stack([p * 255] * 3, -1).astype(np.uint8)
        d = np.full((H, W, 3), 40, np.uint8); d[f & p] = 200
        d[f & ~p] = (230, 40, 40); d[~f & p] = (40, 90, 230)
        sep = np.full((H, 2, 3), 128, np.uint8)
        img = np.concatenate([a, sep, b, sep, d], 1)
        frames.append(Image.fromarray(img).resize((img.shape[1] * scale, H * scale), Image.NEAREST))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=ms, loop=0)


def contact_sheet(P, F, path, every=6, scale=3):
    from PIL import Image
    T, H, W = F.shape; rows = []
    for X in (F, P, (F != P)):
        tiles = [np.pad(X[t].astype(np.uint8) * 255, 1, constant_values=128) for t in range(0, T, every)]
        rows.append(np.concatenate(tiles, 1))
    img = np.concatenate(rows, 0)
    Image.fromarray(img).resize((img.shape[1] * scale, img.shape[0] * scale), Image.NEAREST).save(path)
