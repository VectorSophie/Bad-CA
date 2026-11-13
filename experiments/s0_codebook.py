"""Stage 0 (context codebook): can the per-window local laws of Bad Apple be explained by a small SHARED codebook of K
lookup-table rules? CPU only, no learned circuits. Every rule is an exact table "local pattern -> next value of the
centre cell"; a window is explained by one code (or one code per region in the regional diagnostic).

Rules (keys, bit p = 3x3 position p in row-major order, centre p = 4, zero padding as in FINDINGS §10a and D4):
  m0  memoryless 3x3 of frame t                      (9 bits, the §10a R = 1 table)
  m1  3x3 of frame t + 3x3 of frame t-1 (bits 9..17) (18 bits, the D4 locality check "3x3 + previous frame")
A table holds 0, 1 or HOLD (2); HOLD (unseen pattern or a tie) keeps the centre cell's value. In flip form the same
function is G(key) = 0 for HOLD else T(key) XOR centre, and next = centre XOR G (identical predictions by construction).

Data and splits: runs/context_s0/manifest.json, written by `prereg` before any fit (refuses to overwrite).
usage: s0_codebook.py prereg | run | interp"""
import json, math, os, sys, time
from multiprocessing import Pool
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path[:0] = [ROOT, os.path.dirname(os.path.abspath(__file__))]
OUT = os.path.join(ROOT, "runs", "context_s0"); MAN = os.path.join(OUT, "manifest.json")
L, P_OBS, MARGIN, HOLD = 16, 2, 2, 2          # window length (frames), observation prefix (frames before s), split margin
NBITS = {"m0": 9, "m1": 18}
KS = (1, 2, 4, 8, 16, 32)
RESTARTS = 8
GRIDS = ((1, 2), (2, 2), (3, 3))              # regional diagnostic: rows x cols of coarse regions
MOVING = 0.005                                # a window "moves" if its mean true change per transition >= this
STRATA = ((0.005, 0.015), (0.015, 0.04), (0.04, 1.01))
DUP_PX = 9                                    # near-duplicate frame: Hamming distance <= 9 px (0.5 %) to a pool frame


# ------------------------------------------------------------------ data ------------------------------------------------------------------
_V = None
def video():
    global _V
    if _V is None:
        from exp45_c12 import target
        _V = target("0:6572").astype(np.uint8)
    return _V


def nb(f):
    """(..., H, W) -> (..., H, W, 9) 3x3 patches, zero padding, row-major, centre index 4."""
    H, W = f.shape[-2:]; p = np.pad(f, [(0, 0)] * (f.ndim - 2) + [(1, 1), (1, 1)])
    return np.stack([p[..., dy:dy + H, dx:dx + W] for dy in range(3) for dx in range(3)], -1)


def keys(cur, prev, mode):
    """Rule input keys for frame(s) cur with previous frame(s) prev (ignored for m0)."""
    k = (nb(cur).astype(np.int64) << np.arange(9)).sum(-1)
    if mode == "m1": k = k | ((nb(prev).astype(np.int64) << np.arange(9)).sum(-1) << 9)
    return k


def centre(k): return ((k >> 4) & 1).astype(np.uint8)


def window_xy(V, s, mode, n=L - 1):
    """Teacher-forced transitions t -> t+1 for t = s .. s+n-1: keys (n, H, W), targets (n, H, W)."""
    return keys(V[s:s + n], V[s - 1:s + n - 1], mode), V[s + 1:s + n + 1]


def counts(k, y, mask=None):
    """Sparse counts of (key -> [n0, n1]); mask selects cells (regional units)."""
    k, y = k.reshape(len(k), -1), y.reshape(len(y), -1)
    if mask is not None: k, y = k[:, mask.ravel()], y[:, mask.ravel()]
    k, y = k.ravel(), y.ravel(); u, inv = np.unique(k, return_inverse=True)
    n1 = np.bincount(inv, weights=y, minlength=len(u)); n0 = np.bincount(inv, minlength=len(u)) - n1
    return (u, n0.astype(np.int64), n1.astype(np.int64))


# ------------------------------------------------------------------ tables ------------------------------------------------------------------
def fit_table(cs, nbits, tie="hold"):
    """Majority table from summed sparse counts. Unseen -> HOLD; tie -> HOLD (tie='zero': 0, the §10a lut_fit rule)."""
    n0 = np.zeros(2 ** nbits, np.int64); n1 = np.zeros(2 ** nbits, np.int64)
    for u, a, b in cs: np.add.at(n0, u, a); np.add.at(n1, u, b)
    T = np.full(2 ** nbits, HOLD, np.int8); T[n1 > n0] = 1; T[n0 > n1] = 0
    if tie == "zero": T[(n0 == n1) & (n0 > 0)] = 0
    return T


def predict(T, k):
    p = T[k]; return np.where(p == HOLD, centre(k), p).astype(np.uint8)


def to_flip(T):
    """Flip form: G(key) = 1 iff the rule changes the centre cell."""
    c = centre(np.arange(len(T))); return np.where(T == HOLD, 0, T.astype(np.uint8) ^ c).astype(np.uint8)


def to_direct(T):
    """Direct binary form D(key) = next value (HOLD resolved to the centre)."""
    return predict(T, np.arange(len(T)))


def losses(Ts, c):
    """Mismatch count of each table in Ts (K, 2^n) on sparse counts c."""
    u, n0, n1 = c; p = Ts[:, u]; p = np.where(p == HOLD, centre(u)[None], p)
    return (np.where(p == 0, n1[None], n0[None])).sum(1)


def hard_em(cs, K, nbits, seed, iters=200):
    """Hard-EM (k-means over windows with exact majority tables). k-means++-style init by loss. Monotone non-increasing
    total loss; stops when assignments repeat. Empty clusters take the currently worst-explained unit."""
    rng = np.random.default_rng(seed); n = len(cs)
    Ts = [fit_table([cs[rng.integers(n)]], nbits)]
    while len(Ts) < K:
        lo = np.min([losses(np.stack(Ts), c) for c in cs], 1).astype(float)
        Ts.append(fit_table([cs[rng.choice(n, p=lo / lo.sum()) if lo.sum() > 0 else rng.integers(n)]], nbits))
    Ts = np.stack(Ts); a = None
    for it in range(iters):
        Lm = np.stack([losses(Ts, c) for c in cs]); na = Lm.argmin(1)
        for k in range(K):                      # refill empty clusters with the worst-explained unit
            if not (na == k).any():
                cur = Lm[np.arange(n), na]; j = int(np.argmax(np.where(np.bincount(na, minlength=K)[na] > 1, cur, -1))); na[j] = k
        if a is not None and (na == a).all(): break
        a = na; Ts = np.stack([fit_table([cs[i] for i in np.flatnonzero(a == k)], nbits) for k in range(K)])
    Lm = np.stack([losses(Ts, c) for c in cs])
    return Ts, a, int(Lm[np.arange(n), a].sum()), it


def roll(Tmap, V0prev, V0, n, mode, codemap=None):
    """Autonomous rollout from (frame s-1, frame s). Tmap: (K, 2^n) tables; codemap (H, W) code per cell (global code:
    a constant map). Returns (n, H, W)."""
    prev, cur = V0prev, V0; out = []; cm = np.zeros(V0.shape, np.int64) if codemap is None else codemap
    for _ in range(n):
        k = keys(cur, prev, mode); p = Tmap[cm, k]; nxt = np.where(p == HOLD, centre(k), p).astype(np.uint8)
        prev, cur = cur, nxt; out.append(nxt)
    return np.stack(out)


def onestep(Tmap, V, s, mode, codemap=None, n=L - 1):
    k, y = window_xy(V, s, mode, n); cm = np.zeros(V.shape[1:], np.int64) if codemap is None else codemap
    p = Tmap[cm[None], k]; return np.where(p == HOLD, centre(k), p).astype(np.uint8), y


# ------------------------------------------------------------------ metrics ------------------------------------------------------------------
def change_prf(pred, cur, y):
    """One-step change precision / recall / F1, pooled: predicted change pred != cur vs true change y != cur."""
    pc, tc = pred != cur, y != cur; hit = (pc & tc).sum(); npc, ntc = pc.sum(), tc.sum()
    pr = hit / npc if npc else (1.0 if not ntc else 0.0); rc = hit / ntc if ntc else (1.0 if not npc else 0.0)
    return dict(prec=float(pr), rec=float(rc), f1=float(2 * pr * rc / (pr + rc)) if pr + rc else 0.0)


def m_onestep(pred, V, s):
    cur, y = V[s:s + len(pred)], V[s + 1:s + len(pred) + 1]
    return dict(err=float((pred != y).mean()), hold=float((cur != y).mean()), **change_prf(pred, cur, y))


def m_roll(B, V, s):
    import pd_vis
    F = V[s:s + len(B) + 1]; m = pd_vis.sequence_metrics(np.concatenate([F[:1], B]), F, first_free=1)
    e = (B != F[1:]).mean((1, 2))
    return dict(err=m["err"], hold=m["hold"], iou=m["iou"], bf1=m["boundary_f1"], f1=m["lag1"]["f1"], prec=m["lag1"]["prec"], rec=m["lag1"]["rec"],
                period=m["terminal_period_pred"], **{f"err_h{h}": float(e[h - 1]) for h in (1, 4, 8, 15) if h <= len(e)})


# ------------------------------------------------------------------ split manifest (frozen before any fit) ------------------------------------------------------------------
def prereg():
    if os.path.exists(MAN): sys.exit("manifest exists; refusing to overwrite (frozen)")
    PRE = json.load(open(os.path.join(ROOT, "runs", "readable_rules", "gen", "prereg_windows.json")))
    W = PRE["windows"]; n = len(W); perm = np.random.default_rng(20261113).permutation(n)
    role = {int(i): "fit" for i in perm[:12]}; role.update({int(i): "val" for i in perm[12:18]}); role.update({int(i): "test" for i in perm[18:]})
    ev = []
    for i, w in enumerate(W):
        names = ("train", "adjacent", "distant", "cross") if role[i] != "fit" else ("adjacent", "distant", "cross")
        for nm in names:
            split = "val" if role[i] == "val" else "test"
            group = ("cut" if nm == "cross" else ("same_scene" if role[i] == "fit" else "cross_scene"))
            vis = "same" if nm == "train" else w.get(f"visual_{nm}", w.get("cross_kind"))
            ev.append(dict(scene=i, scene_range=w["scene"], name=nm, start=w[nm][0], end=w[nm][1], split=split, group=group, visual=vis))
    forb = np.zeros(6572, bool)
    for i, w in enumerate(W):
        if role[i] != "fit": forb[max(0, w["scene"][0] - MARGIN):w["scene"][1] + MARGIN] = True
    for e in ev: forb[max(0, e["start"] - P_OBS - 1 - MARGIN):e["end"] + MARGIN] = True     # incl. observation prefix and s-1
    pool = [s for s in range(P_OBS + 1, 6572 - L + 1, L) if not forb[s - P_OBS - 1:s + L].any()]
    m = dict(fixed_at=time.strftime("%Y-%m-%d %H:%M:%S"), source="runs/readable_rules/gen/prereg_windows.json (24 scenes, windows of 16 frames, fixed 2026-11-12)",
             scene_split_seed=20261113, scene_roles={str(i): role[i] for i in range(n)},
             rules=dict(window_frames=L, transitions_per_window=L - 1, observation_prefix_frames=P_OBS, margin=MARGIN,
                        pool="16-frame tiles at stride 16 whose frames [s-3, s+16) avoid every val/test scene (+-2) and every eval window with its prefix (+-2)",
                        moving=f"mean true change per transition >= {MOVING}", strata=STRATA, near_duplicate=f"frame within {DUP_PX} px of a pool frame; a window is near-dup if >= 8 of its 16 frames are",
                        K=KS, restarts=RESTARTS, codebook_choice="restart with the lowest TRAINING loss", grids=GRIDS,
                        K_selection="smallest K whose VAL oracle-1step gap recovery (moving windows) >= 0.9 x the maximum over K (m0); test is read once after",
                        observation_only="code = argmin one-step mismatches on the P_OBS transitions before s (frames s-2..s; m1 uses 1 transition, its prev frame s-2); ties -> most frequent pool code",
                        knn="5-NN on prefix motion features (area change, centroid shift, births, deaths of frames s-2..s) against pool windows labelled by their assigned code",
                        provisional_thresholds=dict(K_max=16, beats_hold_onestep_oracle=">70% of moving windows", gap_recovery=">=0.5",
                                                    autonomous="rollout (15 steps) error < hold on >=50% of moving windows and lag-1 change F1 >= 0.2",
                                                    observation_only="obs assignment keeps >=50% of the oracle's one-step gain over K=1 (GO), 20-50% PARTIAL"),
                        groups="same_scene = adjacent/distant windows of fit scenes (their train windows are pool-eligible); cross_scene = train/adjacent/distant of val/test scenes; cut = cross windows straddling a wipe/cut"),
             eval_windows=ev, pool=pool)
    os.makedirs(OUT, exist_ok=True); json.dump(m, open(MAN, "w"), indent=1)
    print("pool tiles", len(pool), "eval", {s: sum(e["split"] == s for e in ev) for s in ("val", "test")})


# ------------------------------------------------------------------ assignment ------------------------------------------------------------------
def obs_code(Ts, Vpre, mode, prior_order):
    """Observation-only choice. Vpre = frames s-3 .. s (4 frames); nothing at or after s+1 is passed in."""
    if mode == "m0": k = keys(Vpre[1:3], None, mode); y = Vpre[2:4]
    else: k = keys(Vpre[2:3], Vpre[1:2], mode); y = Vpre[3:4]
    lo = losses(Ts, counts(k, y)); best = lo.min()
    return int(next(c for c in prior_order if lo[c] == best))


def motion_feats(Vpre):
    """Frames s-2..s (last three of Vpre): area change, centroid shift (dy, dx), births, deaths per pixel."""
    import pd_vis
    f = Vpre[-3:]; bg = pd_vis.bg_colour(f[-1]); fg = (f != bg)
    cen = lambda m: np.array(np.nonzero(m)).mean(1) if m.any() else np.array([18.0, 24.0])
    d = cen(fg[-1]) - cen(fg[0]); b = ((f[1:] == 1) & (f[:-1] == 0)).mean(); de = ((f[1:] == 0) & (f[:-1] == 1)).mean()
    return np.array([fg[-1].mean() - fg[0].mean(), d[0], d[1], b, de])


# ------------------------------------------------------------------ run ------------------------------------------------------------------
def _em_job(a):
    mode, K, seed, units = a; cs = UNITS[units][mode]; t0 = time.process_time()
    Ts, asg, loss, it = hard_em(cs, K, NBITS[mode], seed); return dict(mode=mode, K=K, seed=seed, units=units, loss=loss, iters=it, cpu=time.process_time() - t0), Ts, asg


UNITS = {}
def build_units(V, pool):
    """Sparse counts for every pool window (global units) and every (window, region) for each grid."""
    U = {"global": {}}
    for mode in NBITS:
        U["global"][mode] = [counts(*window_xy(V, s, mode)) for s in pool]
    for g in GRIDS:
        nm = f"grid{g[0]}x{g[1]}"; U[nm] = {}; masks = region_masks(g, V.shape[1:])
        for mode in NBITS:
            U[nm][mode] = [counts(*window_xy(V, s, mode), mask=m) for s in pool for m in masks]
    return U


def region_masks(g, shape):
    H, W = shape; ys = np.linspace(0, H, g[0] + 1).astype(int); xs = np.linspace(0, W, g[1] + 1).astype(int); out = []
    for i in range(g[0]):
        for j in range(g[1]):
            m = np.zeros(shape, bool); m[ys[i]:ys[i + 1], xs[j]:xs[j + 1]] = True; out.append(m)
    return out


def table_bits(T):
    """Ideal enumerative code length of the flip set and of the direct 1-set (bits), plus raw sizes."""
    N = len(T); nf = int(to_flip(T).sum()); nd = int(to_direct(T).sum())
    lb = lambda n, k: (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)) / math.log(2) + math.log2(n + 1)
    return dict(flip_entries=nf, direct_ones=nd, flip_bits=lb(N, nf), direct_bits=lb(N, nd))


def eval_window(e, V, books, mode, prior_orders, knn):
    """All assignments for one eval window under every frozen codebook of this mode. books: {K: Ts}."""
    s = e["start"]; nb_ = NBITS[mode]; Vpre = V[s - 3:s + 1]; r = dict(e=e, mode=mode)
    k, y = window_xy(V, s, mode); cur = V[s:s + L - 1]; c = counts(k, y)
    r["true_change"] = float((cur != y).mean())
    own = fit_table([c], nb_)                                                        # per-window oracle (in-sample)
    r["perwin"] = dict(one=m_onestep(predict(own, k), V, s), roll=m_roll(roll(own[None], V[s - 1], V[s], L - 1, mode), V, s), bits=table_bits(own))
    from rr_gen import fit_shift, shift_roll, shift
    for nm, d in (("shift_oracle", fit_shift(V[s:s + L])), ("shift_obs", fit_shift(V[s - P_OBS:s + 1]))):
        one = np.stack([shift(V[t], *d) for t in range(s, s + L - 1)])
        r[nm] = dict(d=list(d), one=m_onestep(one, V, s), roll=m_roll(shift_roll(V[s], d, L - 1), V, s))
    for K, Ts in books.items():
        lo = losses(Ts, c); o1 = int(lo.argmin())
        rolls = [roll(Ts[j:j + 1], V[s - 1], V[s], L - 1, mode) for j in range(K)]
        rerr = [float((b != V[s + 1:s + L]).mean()) for b in rolls]; orr = int(np.argmin(rerr))
        ob = obs_code(Ts, Vpre, mode, prior_orders[K]); kn = knn[K](motion_feats(Vpre)) if knn.get(K) else None
        a = {"oracle1": o1, "oracle_roll": orr, "obs": ob, "prior": prior_orders[K][0]}
        if kn is not None: a["knn"] = kn
        r[f"K{K}"] = {nm: dict(code=j, one=m_onestep(predict(Ts[j], k), V, s), roll=m_roll(rolls[j], V, s)) for nm, j in a.items()}
    return r


def eval_regional(e, V, Ts, mode, g):
    """Oracle regional assignment (per region: argmin one-step mismatches on its cells) with a frozen codebook."""
    s = e["start"]; k, y = window_xy(V, s, mode); masks = region_masks(g, V.shape[1:]); cm = np.zeros(V.shape[1:], np.int64); codes = []
    for m in masks:
        j = int(losses(Ts, counts(k, y, mask=m)).argmin()); cm[m] = j; codes.append(j)
    p, _ = onestep(Ts, V, s, mode, cm)
    return dict(codes=codes, one=m_onestep(p, V, s), roll=m_roll(roll(Ts, V[s - 1], V[s], L - 1, mode, cm), V, s))


class KNN:
    """5-NN code vote on standardized prefix motion features (picklable for worker processes); tie -> nearest."""
    def __init__(self, Fz, mu, sd, asg): self.Fz, self.mu, self.sd, self.asg = Fz, mu, sd, asg
    def __call__(self, x):
        d = np.linalg.norm(self.Fz - (x - self.mu) / self.sd, axis=1); nn = np.argsort(d, kind="stable")[:5]; v = np.bincount(self.asg[nn])
        top = np.flatnonzero(v == v.max()); return int(next(self.asg[i] for i in nn if self.asg[i] in top))


def run():
    global UNITS
    man = json.load(open(MAN)); V = video(); pool = man["pool"]; EV = man["eval_windows"]; t0 = time.time()
    UNITS = build_units(V, pool); print("units built", round(time.time() - t0, 1), flush=True)
    jobs = [(mode, K, sd, "global") for mode in NBITS for K in KS for sd in range(RESTARTS)]
    jobs += [(mode, K, sd, f"grid{g[0]}x{g[1]}") for mode in ("m0",) for g in GRIDS for K in (4, 8, 16) for sd in range(RESTARTS)]
    with Pool(min(64, len(jobs))) as p: res = p.map(_em_job, jobs)
    em = [r[0] for r in res]; best = {}
    vk = {}                                                    # VAL one-step oracle error of EVERY restart (seed spread), moving windows
    for mode in NBITS:
        vk[mode] = []
        for e in EV:
            if e["split"] != "val": continue
            k, y = window_xy(V, e["start"], mode)
            if (V[e["start"]:e["start"] + L - 1] != y).mean() >= MOVING: vk[mode].append((counts(k, y), k.size))
    for info, Ts, _ in res:
        if info["units"] == "global":
            info["val_oracle1_err"] = float(np.mean([losses(Ts, c).min() / n for c, n in vk[info["mode"]]]))
    for info, Ts, asg in res:
        key = (info["mode"], info["K"], info["units"])
        if key not in best or info["loss"] < best[key][0]["loss"]: best[key] = (info, Ts, asg)
    np.savez_compressed(os.path.join(OUT, "codebooks.npz"), **{f"{m}_K{K}_{u}": b[1] for (m, K, u), b in best.items()},
                        **{f"asg_{m}_K{K}_{u}": b[2] for (m, K, u), b in best.items()})
    json.dump(em, open(os.path.join(OUT, "em_restarts.json"), "w"), indent=0)
    # training fit (A): pool windows, global codebooks
    train = {}
    for mode in NBITS:
        cs = UNITS["global"][mode]; tot = sum(int(c[1].sum() + c[2].sum()) for c in cs)
        holdm = sum(int(np.where(centre(c[0]) == 0, c[2], c[1]).sum()) for c in cs)
        pw = sum(int(losses(fit_table([c], NBITS[mode])[None], c)[0]) for c in cs)
        train[mode] = dict(hold=holdm / tot, perwin=pw / tot, windows=len(cs), **{f"K{K}": best[(mode, K, "global")][0]["loss"] / tot for K in KS})
    # restart spread of the per-restart TRAINING loss
    spread = {f"{m}_K{K}_{u}": [r["loss"] for r in em if (r["mode"], r["K"], r["units"]) == (m, K, u)] for (m, K, u) in best}
    # prior order (by pool cluster size) and kNN classifier per K
    priors = {m: {K: list(np.argsort(-np.bincount(best[(m, K, "global")][2], minlength=K), kind="stable")) for K in KS} for m in NBITS}
    feats = np.stack([motion_feats(V[s - 3:s + 1]) for s in pool]); mu, sd = feats.mean(0), feats.std(0) + 1e-9; Fz = (feats - mu) / sd
    knns = {m: {K: KNN(Fz, mu, sd, best[(m, K, "global")][2]) for K in KS if K > 1} for m in NBITS}
    books = {m: {K: best[(m, K, "global")][1] for K in KS} for m in NBITS}
    with Pool(32) as p:
        rows = p.starmap(eval_window, [(e, V, books[m], m, priors[m], knns[m]) for m in NBITS for e in EV])
    reg = []
    for g in GRIDS:
        u = f"grid{g[0]}x{g[1]}"
        for K in (4, 8, 16):
            for src in ("frozen_global", "refit_regional"):
                Ts = books["m0"][K] if src == "frozen_global" else best[("m0", K, u)][1]
                with Pool(32) as p: rr = p.starmap(eval_regional, [(e, V, Ts, "m0", g) for e in EV])
                for e, x in zip(EV, rr): reg.append(dict(e=e, grid=list(g), K=K, codebook=src, **x))
    # near-duplicate check (reported, not used to drop anything)
    PF = np.unique(np.concatenate([V[s - 3:s + L] for s in pool]).reshape(-1, V.shape[1] * V.shape[2]), axis=0)
    dup = {}
    for e in EV:
        F = V[e["start"]:e["end"]].reshape(L, -1); d = [int((PF != f).sum(1).min()) for f in F]; dup[f'{e["start"]}'] = dict(min_dist=d, near_dup=sum(x <= DUP_PX for x in d) >= 8)
    json.dump(dict(train=train, restart_train_loss=spread, eval=rows, regional=reg, near_dup=dup, wall_s=round(time.time() - t0, 1)),
              open(os.path.join(OUT, "results.json"), "w"), default=lambda o: int(o) if isinstance(o, np.integer) else float(o))
    print("done", round(time.time() - t0, 1))


if __name__ == "__main__":
    {"prereg": prereg, "run": run}[sys.argv[1]]()
