"""Campaign 10 shape/timing metrics on arrays (never on rendered video).
P: binary replay (T,H,W), V: continuous visible (T,H,W) or None, F: original binary frames (T,H,W); frame 0 is given
(P[0] == F[0]); every metric uses frames 1..T-1 unless stated.
Regions of the ORIGINAL frame t: band = 3x3 neighbourhood (edge-replicated) holds both values (2 px wide, both sides
of the true contour), interior = F & ~band, background = ~F & ~band.
Contour: inner boundary B(X) = X & ~erode4(X) (edge-replicated); tolerant precision/recall with a 3x3 window.
Blip: a pixel that flips at t and flips back at t+1; 'spurious' = a blip of P where F has no blip at that pixel/time.
Error runs: per pixel, maximal runs of consecutive wrong frames.
Small components: 8-connected components of size <= 3 of foreground (bumps/specks) and of background (holes)."""
import numpy as np
from scipy import ndimage
from ca.motion import _dil
from ca.motion9 import lag_terms


def _pad_e(X): return np.pad(X, ((0, 0), (1, 1), (1, 1)), mode="edge")


def band(F):
    F = F.astype(bool); p = _pad_e(F); H, W = F.shape[1:]; mx = np.zeros_like(F); mn = np.ones_like(F)
    for dy in range(3):
        for dx in range(3):
            s = p[:, dy:dy + H, dx:dx + W]; mx |= s; mn &= s
    return mx & ~mn


def boundary(X):
    X = X.astype(bool); p = _pad_e(X); H, W = X.shape[1:]
    er = X & p[:, :-2, 1:-1][:, :H] & p[:, 2:, 1:-1] & p[:, 1:-1, :-2] & p[:, 1:-1, 2:]
    return X & ~er


def contour_prf(P, F):
    bp, bf = boundary(P[1:]), boundary(F[1:])
    p = (bp & _dil(bf)).sum() / max(bp.sum(), 1); r = (bf & _dil(bp)).sum() / max(bf.sum(), 1)
    return dict(prec=float(p), rec=float(r), f1=float(2 * p * r / (p + r)) if p + r else 0.0)


def blips(X):
    X = X.astype(bool); return (X[1:-1] != X[:-2]) & (X[2:] == X[:-2])          # at t = 1..T-2


def error_runs(E):
    """E (T,H,W) bool -> list of run lengths (frames)."""
    T = len(E); runs = []; cur = np.zeros(E.shape[1:], int)
    for t in range(T):
        ended = (~E[t]) & (cur > 0); runs += cur[ended].tolist(); cur = np.where(E[t], cur + 1, 0)
    return runs + cur[cur > 0].tolist()


def small_cc(X, maxsize=3):
    st = np.ones((3, 3), bool); out = []
    for x in X.astype(bool):
        n = [0, 0]
        for i, m in enumerate((x, ~x)):
            lab, k = ndimage.label(m, st)
            if k: n[i] = int((np.bincount(lab.ravel())[1:] <= maxsize).sum())
        out.append(n)
    return np.array(out)                                                          # (T, 2): fg specks, bg holes


def frame_report(P, V, F, frames):
    """Metrics over the frame index list `frames` (subset of 1..T-1)."""
    P = P.astype(bool); F = F.astype(bool); fr = np.asarray(frames)
    Pp, Fp = P[fr], F[fr]; bd = band(Fp); E = Pp != Fp; n = Pp.size
    out = dict(n_frames=len(fr), err=float(E.mean()), fp=float((Pp & ~Fp).mean()), fn=float((~Pp & Fp).mean()),
               err_band=float(E[bd].sum() / n), err_interior=float(E[Fp & ~bd].sum() / n), err_background=float(E[~Fp & ~bd].sum() / n),
               frac_err_in_band=float(E[bd].sum() / max(E.sum(), 1)))
    if V is not None:
        d = np.abs(V[fr] - 0.5)[E]
        out.update(err_margin_lt_0_1=float((d < 0.1).mean()) if d.size else None, err_margin_0_1_to_0_25=float(((d >= 0.1) & (d < 0.25)).mean()) if d.size else None,
                   err_margin_ge_0_25=float((d >= 0.25).mean()) if d.size else None, all_margin_lt_0_1=float((np.abs(V[fr] - 0.5) < 0.1).mean()))
    return out


def full_report(P, V, F, win=None):
    """Whole-span report (frames 1..T-1), thirds, and (optionally) a window (t_a, t_b) inclusive of original-aligned frames."""
    P = P.astype(np.uint8); F = F.astype(np.uint8); T = len(F); idx = np.arange(1, T)
    hold = float((F[1:] != F[:1]).mean())
    r = dict(T=T, all=frame_report(P, V, F, idx), thirds=[frame_report(P, V, F, s) for s in np.array_split(idx, 3)],
             hold=hold, contour=contour_prf(P, F), lag1=lag_terms(P, F, 1), lag4=lag_terms(P, F, 4))
    r["err_over_hold"] = r["all"]["err"] / hold if hold else None
    bp, bf = blips(P), blips(F); r["blips_pred"] = int(bp.sum()); r["blips_true"] = int(bf.sum()); r["blips_spurious"] = int((bp & ~bf).sum())
    runs = np.array(error_runs(P[1:] != F[1:]) or [0]); r["err_runs"] = dict(n=int((runs > 0).sum()), mean=float(runs.mean()), max=int(runs.max()),
                                                                            frac_pixelframes_in_runs_ge4=float(runs[runs >= 4].sum() / max(runs.sum(), 1)))
    sp, sf = small_cc(P[1:]), small_cc(F[1:]); r["small_cc"] = dict(pred_specks=int(sp[:, 0].sum()), true_specks=int(sf[:, 0].sum()),
                                                                     pred_holes=int(sp[:, 1].sum()), true_holes=int(sf[:, 1].sum()))
    r["err_curve"] = [round(float((P[t] != F[t]).mean()), 4) for t in range(T)]
    r["thin"] = thin_recall(P, F)
    if win:
        a, b = win; r["window"] = frame_report(P, V, F, np.arange(a, b + 1))
        r["window"]["lag4"] = lag_terms(P[a:b + 1], F[a:b + 1], 4); r["window"]["lag1"] = lag_terms(P[a:b + 1], F[a:b + 1], 1)
        r["window"]["contour"] = contour_prf(P[a:b + 1], F[a:b + 1])
    return r


def thin_recall(P, F):
    """Campaign 11: are thin parts and thin gaps of the ORIGINAL kept? thin_fg = F minus its 3x3 opening (fingers, cane,
    1-2 px limbs); thin_bg = closing(F) minus F (narrow gaps, e.g. between arm and body). Recall over frames 1..T-1."""
    st = np.ones((1, 3, 3), bool); P = P[1:].astype(bool); F = F[1:].astype(bool)
    tf = F & ~ndimage.binary_opening(F, st); tb = ndimage.binary_closing(F, st, border_value=0) & ~F
    return dict(thin_fg_px=int(tf.sum()), thin_fg_recall=float(P[tf].mean()) if tf.any() else None,
                thin_bg_px=int(tb.sum()), thin_bg_recall=float((~P[tb]).mean()) if tb.any() else None)


def shift_curve(P, F, frames, ds=range(-3, 4)):
    """POST-HOC diagnostic only: mean err(P_t, F_{t+d}) over t in frames (where t+d in range). d < 0 lowest: replay lags behind the original; d > 0 lowest: replay runs ahead."""
    T = len(F); return {int(d): float(np.mean([(P[t] != F[t + d]).mean() for t in frames if 0 <= t + d < T])) for d in ds}


if __name__ == "__main__":   # self-check on toy frames
    F = np.zeros((6, 8, 8), np.uint8); F[:, 2:6, 2:6] = 1; P = F.copy(); P[3, 4, 4] = 0
    b = band(F[:1])[0]; assert b[2, 2] and b[1, 2] and not b[4, 4] and not b[0, 0]
    r = full_report(P, None, F); assert r["all"]["fn"] > 0 and r["all"]["err_interior"] > 0 and r["all"]["err_band"] == 0
    assert r["small_cc"]["pred_holes"] == 1 and r["blips_spurious"] == 1 and r["err_runs"]["max"] == 1
    assert boundary(F[:1])[0].sum() == 12; assert shift_curve(F, F, range(1, 6))[0] == 0.0
    print("diag10 self-check ok")
