"""Campaign 08 motion metrics and objective. P, F: (T,H,W) binary, OBSERVED frames only (CA generations
between observed frames are never compared; 'change' always means observed frame t-1 -> t).

Directional changes, t = 1..T-1:  on = X_t & ~X_{t-1},  off = ~X_t & X_{t-1}.
Exact:     TP = |Pon & Ton| + |Poff & Toff|.
Tolerant (radius 1, 3x3): a predicted on/off change is a hit if a TRUE change of the SAME direction lies in its
  3x3 window (precision side); a true change is recovered if a PREDICTED change of the same direction lies in
  its 3x3 window (recall side).  P_tol = hitsP / nP, R_tol = hitsT / nT, F1 = 2PR/(P+R).
Aggregation per segment = micro sums over its frames. Empty cases: nP = 0 -> precision := 1 (nothing claimed),
  nT = 0 -> recall := 1 (nothing to find); both empty -> F1 = 1; F1 = 0 if P + R = 0.
Segments: thirds of frames 1..T-1 (T=60: early 1-19 | mid 20-39 | late 40-59).
Objective (minimise), segment-balanced:  J = mean_seg [ err_seg / max(hold_seg, 1e-3) + lam * (1 - F1tol_seg) ].
Diagnostics: per-segment pixel/hold error, on/off split, predicted/true change mass ratio (flicker),
  smallest period p <= 12 of the late replay (p = 1 = frozen).
"""
import numpy as np


def _dil(X):
    T, H, W = X.shape; p = np.pad(X, ((0, 0), (1, 1), (1, 1))); out = np.zeros_like(X)
    for dy in range(3):
        for dx in range(3):
            out |= p[:, dy:dy + H, dx:dx + W]
    return out


def segments(T):
    idx = np.arange(1, T); return [s for s in np.array_split(idx, 3)]


def frame_terms(P, F):
    P = P.astype(bool); F = F.astype(bool)
    Pon, Poff = P[1:] & ~P[:-1], ~P[1:] & P[:-1]; Ton, Toff = F[1:] & ~F[:-1], ~F[1:] & F[:-1]
    dTon, dToff, dPon, dPoff = _dil(Ton), _dil(Toff), _dil(Pon), _dil(Poff)
    s = lambda X: X.reshape(len(X), -1).sum(1)
    return dict(err=(P[1:] != F[1:]).mean((1, 2)), hold=(F[1:] != F[:1]).mean((1, 2)),
                nPon=s(Pon), nPoff=s(Poff), nTon=s(Ton), nToff=s(Toff),
                tp_on=s(Pon & Ton), tp_off=s(Poff & Toff),
                hitP_on=s(Pon & dTon), hitP_off=s(Poff & dToff), hitT_on=s(Ton & dPon), hitT_off=s(Toff & dPoff))


def _prf(hp, np_, ht, nt):
    p = hp / np_ if np_ else 1.0; r = ht / nt if nt else 1.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def seg_metrics(ft, sl):
    g = {k: v[sl - 1] for k, v in ft.items()}; S = {k: float(v.sum()) for k, v in g.items()}
    nP, nT = S["nPon"] + S["nPoff"], S["nTon"] + S["nToff"]
    pt, rt, ft1 = _prf(S["hitP_on"] + S["hitP_off"], nP, S["hitT_on"] + S["hitT_off"], nT)
    pe, re_, fe = _prf(S["tp_on"] + S["tp_off"], nP, S["tp_on"] + S["tp_off"], nT)
    _, _, f_on = _prf(S["hitP_on"], S["nPon"], S["hitT_on"], S["nTon"]); _, _, f_off = _prf(S["hitP_off"], S["nPoff"], S["hitT_off"], S["nToff"])
    iou = (S["tp_on"] + S["tp_off"]) / (nP + nT - S["tp_on"] - S["tp_off"]) if (nP + nT) else 1.0
    return dict(err=float(g["err"].mean()), hold=float(g["hold"].mean()), prec_tol=pt, rec_tol=rt, f1_tol=ft1, prec=pe, rec=re_, f1=fe,
                change_iou=iou, f1_tol_on=f_on, f1_tol_off=f_off, pred_changes=int(nP), true_changes=int(nT),
                change_mass_ratio=(nP / nT if nT else (float("inf") if nP else 1.0)))


def period(P, sl, pmax=12):
    """Smallest p <= pmax with P[t] == P[t-p] for every t in segment frames sl (None if none)."""
    for p in range(1, pmax + 1):
        if all((P[t] == P[t - p]).all() for t in sl if t - p >= 0): return p
    return None


def objective(P, F, lam):
    ft = frame_terms(P, F); segs = segments(len(F))
    return float(np.mean([seg_metrics(ft, s)["err"] / max(seg_metrics(ft, s)["hold"], 1e-3) + lam * (1 - seg_metrics(ft, s)["f1_tol"]) for s in segs]))


def report(P, F, lam=None):
    ft = frame_terms(P, F); segs = segments(len(F)); names = ("early", "mid", "late")
    out = {n: seg_metrics(ft, s) for n, s in zip(names, segs)}
    out["all"] = seg_metrics(ft, np.arange(1, len(F)))
    out["late_period"] = period(P.astype(np.uint8), segs[2]); out["late_frozen"] = out["late_period"] == 1
    out["late_flicker"] = out["late"]["change_mass_ratio"] > 3
    out["exact"] = bool((P == F).all())
    if lam is not None: out["J"] = float(np.mean([out[n]["err"] / max(out[n]["hold"], 1e-3) + lam * (1 - out[n]["f1_tol"]) for n in names]))
    return out
