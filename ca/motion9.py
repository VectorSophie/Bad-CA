"""Campaign 09 window metrics (binary frames P, F of one task window, P[0] == F[0]).
For lag k in (1, 4): changes X_t vs X_{t-k} for t = k..T-1, split into on/off; tolerant directional precision/recall/F1
as in ca.motion (3x3 window, same direction; nP = 0 -> precision 1, nT = 0 -> recall 1). Lag 4 measures change that
ACCUMULATES over 4 frames; a period-2 flicker cancels at even lags and earns nothing there.
Also: pixel error and hold error over frames 1..T-1, revert rate (changes at t undone at t+1) of P and of F."""
import numpy as np
from ca.motion import _dil, _prf


def lag_terms(P, F, k):
    P = P.astype(bool); F = F.astype(bool)
    Pon, Poff = P[k:] & ~P[:-k], ~P[k:] & P[:-k]; Ton, Toff = F[k:] & ~F[:-k], ~F[k:] & F[:-k]
    hp = (Pon & _dil(Ton)).sum() + (Poff & _dil(Toff)).sum(); ht = (Ton & _dil(Pon)).sum() + (Toff & _dil(Poff)).sum()
    nP, nT = int(Pon.sum() + Poff.sum()), int(Ton.sum() + Toff.sum())
    p, r, f = _prf(hp, nP, ht, nT)
    return dict(prec_tol=p, rec_tol=r, f1_tol=f, pred=nP, true=nT, mass_ratio=nP / nT if nT else None)


def revert(X):
    X = X.astype(bool); ch = X[1:] != X[:-1]; den = ch[:-1].sum()
    return float((ch[:-1] & ch[1:]).sum() / den) if den else None


def window_report(P, F):
    P = P.astype(np.uint8); F = F.astype(np.uint8)
    err = float((P[1:] != F[1:]).mean()); hold = float((F[1:] != F[:1]).mean())
    return dict(T=len(F), err=err, hold=hold, err_over_hold=err / hold if hold else None, lag1=lag_terms(P, F, 1), lag4=lag_terms(P, F, 4),
                revert_pred=revert(P), revert_true=revert(F), exact=bool((P == F).all()),
                err_curve=[round(float((P[t] != F[t]).mean()), 4) for t in range(len(F))])
