"""Automatic phase segmentation of a binary frame sequence (stage 1: video analysis only, before any rule training).

Frame t (t >= 1) is produced by the phase that owns t; a phase is a half-open frame range [a, b).
Signals per transition t-1 -> t:
  change(t)   = fraction of pixels that flip;
  cut(t)      = change >= CUT (0.25) or one of the two frames is uniform (fade/blank);
  wipe score  = longest run of consecutive frames with change >= WIPE (0.08) (sustained large change).
MDL cost of a phase [a, b): code length (bits) of its transitions under ONE empirical table
  next centre bit | 3x3 pattern of the previous frame (zero padding)   (sum over patterns of n * H(next | pattern))
  + BIC table cost 0.5 * log2(n) per observed pattern.  This is a cheap proxy for "one local rule explains this
  stretch"; it is not the CA's actual radius (S=4 substeps reach 4 px), so stage 2 (trained residuals) decides.
best_k(F, K): exact DP over boundaries for exactly K phases, min length LMIN; returns boundaries + per-boundary reasons.
Reasons: 'cut' (a cut transition lies within 1 frame), 'wipe' (sustained change), else 'mdl' with the bit gain of
splitting there vs merging the two neighbours."""
import numpy as np

CUT, WIPE, LMIN = 0.25, 0.08, 8


def signals(F):
    F = F.astype(bool); T = len(F); ch = np.zeros(T); ch[1:] = (F[1:] != F[:-1]).mean((1, 2))
    uni = F.reshape(T, -1).all(1) | ~F.reshape(T, -1).any(1); cut = np.zeros(T, bool)
    cut[1:] = (ch[1:] >= CUT) | (uni[1:] != uni[:-1]) | (uni[1:] & uni[:-1] & (F[1:] != F[:-1]).any((1, 2)))
    return dict(change=ch, uniform=uni, cut=cut)


def transition_counts(F):
    """(T, 512, 2) counts of (3x3 pattern of frame t-1, centre of frame t); row 0 is zero."""
    F = F.astype(np.uint8); T, H, W = F.shape; p = np.pad(F, ((0, 0), (1, 1), (1, 1))); code = np.zeros((T, H, W), np.int32)
    for k, (dy, dx) in enumerate([(a, b) for a in range(3) for b in range(3)]): code |= p[:, dy:dy + H, dx:dx + W].astype(np.int32) << k
    out = np.zeros((T, 512, 2), np.int64)
    for t in range(1, T): np.add.at(out[t], (code[t - 1].ravel(), F[t].ravel()), 1)
    return out


def _cost(c):
    n = c.sum(-1); m = n > 0; c = c[m].astype(float); n = n[m].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.nansum(np.where(c > 0, c * np.log2(c / n[:, None]), 0.0))
    return float(h + 0.5 * np.log2(n).sum())


def best_k(F, K, lmin=LMIN, grid=1, lmax=None, cutpen=0.0):
    """Exact DP: boundaries b_0=1 < b_1 < ... < b_K = T over frames 1..T-1 (frame 0 is the given first frame).
    grid > 1 restricts boundaries to multiples of grid plus every cut frame (for long inputs)."""
    T = len(F); C = np.cumsum(transition_counts(F), 0); sig = signals(F)
    cand = sorted(set([1, T] + [t for t in range(1, T) if t % grid == 0] + [t for t in range(1, T) if sig["cut"][t]] + [t + 1 for t in range(1, T - 1) if sig["cut"][t]]))
    idx = {b: i for i, b in enumerate(cand)}; n = len(cand); INF = float("inf")
    hard = np.cumsum((sig["change"] >= CUT).astype(int))   # cutpen: bits charged per hard cut strictly inside a phase
    seg0 = lambda a, b: _cost(C[b - 1] - C[a - 1])          # transitions of frames a..b-1
    seg = (lambda a, b: seg0(a, b) + cutpen * (hard[b - 1] - hard[a])) if cutpen else seg0
    D = np.full((K + 1, n), INF); back = np.zeros((K + 1, n), int); D[0, 0] = 0.0
    cache = {}
    for k in range(1, K + 1):
        for j in range(1, n):
            b = cand[j]
            for i in range(j - 1, -1, -1):
                a = cand[i]
                if lmax and b - a > lmax: break
                if b - a < lmin or D[k - 1, i] == INF: continue
                if (a, b) not in cache: cache[(a, b)] = seg(a, b)
                v = D[k - 1, i] + cache[(a, b)]
                if v < D[k, j]: D[k, j] = v; back[k, j] = i
    if D[K, n - 1] == INF: raise ValueError(f"cannot split {T} frames into {K} phases of >= {lmin}")
    bs = [n - 1]
    for k in range(K, 0, -1): bs.append(back[k, bs[-1]])
    bounds = [cand[i] for i in reversed(bs)]
    phases = []
    for i in range(K):
        a, b = bounds[i], bounds[i + 1]; r = dict(start=int(a), end=int(b), mdl_bits=round(seg0(a, b), 1))
        if i > 0:
            near = [t for t in range(max(1, a - 1), min(T, a + 2)) if sig["cut"][t]]
            run = 0; best = 0
            for t in range(max(1, a - 12), min(T, a + 12)):
                run = run + 1 if sig["change"][t] >= WIPE else 0; best = max(best, run)
            gain = seg0(bounds[i - 1], b) - seg0(bounds[i - 1], a) - seg0(a, b)
            r["reason"] = "cut" if near else ("wipe" if best >= 3 else "mdl")
            r["reason_detail"] = dict(cut_frames=[int(t) for t in near], change_at=round(float(sig["change"][a]), 3), sustained_change_run=int(best), split_gain_bits=round(gain, 1))
        else:
            r["reason"] = "start"
        phases.append(r)
    return dict(K=K, total_bits=round(float(D[K, n - 1]), 1), phases=phases, lmin=lmin, grid=grid, lmax=lmax, cutpen=cutpen,
                hard_cuts_inside=int(sum(hard[p["end"] - 1] - hard[p["start"]] for p in phases)), hard_cuts=int(hard[-1]))


def uniform_k(T, K):
    e = np.linspace(1, T, K + 1).round().astype(int); return [dict(start=int(e[i]), end=int(e[i + 1]), reason="uniform") for i in range(K)]


def frame_phase(phases, T):
    """phase id of each frame 0..T-1 (frame 0 gets phase 0; it is given, not produced)."""
    ph = np.zeros(T, np.int32)
    for k, p in enumerate(phases): ph[p["start"]:p["end"]] = k
    return ph


if __name__ == "__main__":   # self-check: a sequence with one hard cut is split exactly at the cut
    rng = np.random.default_rng(0); A = np.zeros((40, 12, 12), np.uint8)
    for t in range(40): A[t, 3:7, (t % 8):(t % 8) + 3] = 1
    A[20:] = 1 - A[20:]          # inversion cut at frame 20
    r = best_k(A, 2)
    assert r["phases"][1]["start"] in (20, 21) and r["phases"][1]["reason"] == "cut", r
    assert frame_phase(uniform_k(40, 4), 40)[-1] == 3
    print("segment self-check ok", [(p["start"], p["end"], p["reason"]) for p in r["phases"]])
