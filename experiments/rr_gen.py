"""Readable Rules round 4: generalisation and dynamics (G1-G4). Windows and protocol are fixed in
runs/readable_rules/gen/prereg_windows.json before any fit. Rows are appended to JSONL as they finish, so a kill keeps
partial results. usage: rr_gen.py g1 | lut | g2 | g3 | g4"""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca import rules, rr_search, segment
from rr_run import evaluate
from rr_scale import method, objectives, roll, NC, fit_clip, score_model

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); OUT = os.path.join(ROOT, "runs", "readable_rules", "gen")
PRE = json.load(open(os.path.join(OUT, "prereg_windows.json"))); S = 2
_FULL = None


def video():
    global _FULL
    if _FULL is None:
        from exp45_c12 import target
        _FULL = target("0:6572").astype(np.uint8)
    return _FULL


def append(name, rows):
    with open(os.path.join(OUT, name), "a") as f:
        for r in rows: f.write(json.dumps(r, default=float) + "\n")


def z(k, F): return np.zeros((k,) + F.shape[-2:], np.uint8)


def onestep(p, F, k):
    """Teacher-forced one-step error per transition (hidden 0 at every restart)."""
    return np.array([(roll(p, F[t], z(k, F), 1, S)[0] != F[t + 1]).mean() for t in range(len(F) - 1)])


# ---------- shift baseline: one constant integer shift per frame, fitted one-step on the train window ----------
def shift(v, dy, dx):
    H, W = v.shape; o = np.zeros_like(v)
    o[max(dy, 0):H + min(dy, 0), max(dx, 0):W + min(dx, 0)] = v[max(-dy, 0):H + min(-dy, 0), max(-dx, 0):W + min(-dx, 0)]
    return o


def fit_shift(F):
    return min(((dy, dx) for dy in range(-3, 4) for dx in range(-3, 4)), key=lambda d: sum((shift(F[t], *d) != F[t + 1]).sum() for t in range(len(F) - 1)))


def shift_roll(v, d, n):
    out = []
    for _ in range(n): v = shift(v, *d); out.append(v)
    return np.stack(out)


def score(B, G, hold_frame=None):
    """B: predicted frames for G[1:]; returns window metrics. hold_frame: the frame a 'hold' baseline would repeat."""
    ev = evaluate(np.concatenate([G[:1], B]), G); hf = G[0] if hold_frame is None else hold_frame
    return dict(err=ev["err"], hold=float((G[1:] != hf).mean()), lag4_f1=ev["lag4_f1"], mass=ev["lag4_mass"], worst16=ev["worst16"])


# ---------- G1 ----------
METHODS = (("staged", "two_stage_staged", 0), ("staged_k1", "two_stage_staged", 1), ("rollout_only", "rollout_only", 0))


def g1_one(args):
    wi, mname, seed = args; w = PRE["windows"][wi]; X = video(); _, meth, k = next(m for m in METHODS if m[0] == mname)
    t0 = time.process_time(); a, b = w["train"]; F = X[a:b]
    p = method(meth, F, S, np.random.default_rng(seed), k=k)
    model = dict(k=k, S=S, T=len(F), programs=[p], schedule=[dict(start=1, end=len(F), program=0)], first=F[0], h0=None)
    cx = rules.complexity(model)
    row = dict(window=wi, scene=w["scene"], method=mname, k=k, seed=seed, program=p, conditions=cx["conditions"], rules=cx["rules"], file_bytes=cx["file_bytes"])
    row["train"] = score(roll(p, F[0], z(k, F), len(F) - 1, S), F); row["train"]["one_step"] = float(onestep(p, F, k).mean())
    # A. transfer: restart from each window's true first frame
    for wn in ("adjacent", "distant", "cross"):
        c, d = w[wn]; G = X[c:d]
        row["transfer_" + wn] = dict(score(roll(p, G[0], z(k, G), len(G) - 1, S), G), one_step=float(onestep(p, G, k).mean()))
    # B. continuous: from train[0], no injection, to the end of the last window
    end = max(w["distant"][1], w["cross"][1]); C = X[a:end]; B = roll(p, C[0], z(k, C), len(C) - 1, S)
    full = np.concatenate([C[:1], B]); e = (full != C).mean((1, 2)); o = np.concatenate([[0.0], onestep(p, C, k)])
    jump = next((t for t in range(1, len(C)) if e[t] - o[t] > 0.05), None)
    row["continuous"] = dict(sharp_increase_frame=jump, per_frame_err=[round(float(x), 4) for x in e], per_frame_one_step=[round(float(x), 4) for x in o])
    for wn in ("adjacent", "distant", "cross"):
        c, d = w[wn]; G = X[c:d]; Bs = full[c - a + 1:d - a]
        row["continuous_" + wn] = dict(score(Bs, G, hold_frame=C[0]), window_hold=float((G[1:] != G[0]).mean()))
    row["cpu_s"] = round(time.process_time() - t0, 1)
    return row


def g1_shift(wi):
    w = PRE["windows"][wi]; X = video(); a, b = w["train"]; F = X[a:b]; d = fit_shift(F); row = dict(window=wi, method="shift", shift=d)
    row["train"] = score(shift_roll(F[0], d, len(F) - 1), F)
    end = max(w["distant"][1], w["cross"][1]); C = X[a:end]; full = np.concatenate([C[:1], shift_roll(C[0], d, len(C) - 1)])
    for wn in ("adjacent", "distant", "cross"):
        c, e = w[wn]; G = X[c:e]
        row["transfer_" + wn] = score(shift_roll(G[0], d, len(G) - 1), G)
        row["continuous_" + wn] = dict(score(full[c - a + 1:e - a], G, hold_frame=C[0]), window_hold=float((G[1:] != G[0]).mean()))
    return row


def g1():
    n = len(PRE["windows"]); done = set()
    if os.path.exists(os.path.join(OUT, "g1.jsonl")):
        done = {(r["window"], r["method"], r.get("seed")) for r in map(json.loads, open(os.path.join(OUT, "g1.jsonl")))}
    jobs = [(i, m[0], s) for i in range(n) for m in METHODS for s in range(3) if (i, m[0], s) not in done]
    if not any(m == "shift" for _, m, _ in done): append("g1.jsonl", [g1_shift(i) for i in range(n)])
    with Pool(24) as p:
        for r in p.imap_unordered(g1_one, jobs): append("g1.jsonl", [r]); print(r["window"], r["method"], r["seed"], r["cpu_s"], flush=True)


# ---------- LUT diagnostic (no search): exact radius-R patch tables, in-window optimum vs train -> unseen transfer ----------
def patches(F, R):
    T, H, W = F.shape; p = np.pad(F[:-1], ((0, 0), (R, R), (R, R))); cols = [p[:, dy:dy + H, dx:dx + W] for dy in range(2 * R + 1) for dx in range(2 * R + 1)]
    return np.packbits(np.stack(cols, -1), axis=-1).reshape(-1, (len(cols) + 7) // 8), F[1:].reshape(-1), F[:-1].reshape(-1)


def lut_fit(P, y):
    keys = [bytes(r) for r in P]; cnt = {}
    for kk, yy in zip(keys, y): c = cnt.setdefault(kk, [0, 0]); c[yy] += 1
    return {kk: int(c[1] > c[0]) for kk, c in cnt.items()}, sum(min(c) for c in cnt.values()) / len(y)


def lut_one(wi):
    w = PRE["windows"][wi]; X = video(); a, b = w["train"]; F = X[a:b]; row = dict(window=wi)
    for R in (1, 2, 3, 4):
        P, y, cur = patches(F, R); tab, bayes = lut_fit(P, y); r = dict(in_sample_bayes=bayes)
        for wn in ("adjacent", "distant", "cross"):
            G = X[w[wn][0]:w[wn][1]]; Pg, yg, cg = patches(G, R); keys = [bytes(q) for q in Pg]
            pred = np.array([tab.get(kk, c) for kk, c in zip(keys, cg)])   # unseen patch: keep the cell's state
            r[wn] = dict(transfer_err=float((pred != yg).mean()), unseen_frac=float(np.mean([kk not in tab for kk in keys])), own_bayes=lut_fit(Pg, yg)[1],
                         hold=float((cg != yg).mean()))
        row[f"R{R}"] = r
    return row


def lut():
    with Pool(24) as p: append("lut.jsonl", p.map(lut_one, range(len(PRE["windows"]))))



# ---------- G3: phase placement (boundaries use the fitted clip only; this is a fit/compression result, not blind) ----------
G3_TASKS = ("1221:120", "4500:120", "1760:240", "2700:180")   # C1's three + one fixed before any G3 fit (3 scene breaks)
LMIN3 = 4


def events(F):
    """Cut frames and wipe-run starts inside the clip (segment.signals), with their change strength."""
    sg = segment.signals(F); ch = sg["change"]; T = len(F); ev = {t: ch[t] for t in range(1, T) if sg["cut"][t]}; run = 0
    for t in range(1, T):
        run = run + 1 if ch[t] >= segment.WIPE else 0
        if run == 3: ev.setdefault(t - 2, ch[t - 2:t + 1].max())
    return {t: s for t, s in ev.items() if LMIN3 < t < T - LMIN3}


def to_phases(bounds, T): b = [1] + sorted(bounds) + [T]; return [dict(start=b[i], end=b[i + 1]) for i in range(len(b) - 1)]


def cut_aligned(F, K):
    T = len(F); ev = events(F); bs = sorted(sorted(ev, key=lambda t: -ev[t])[:K - 1]); bs = [t for i, t in enumerate(bs) if not i or t - bs[i - 1] >= LMIN3]
    while len(bs) < K - 1:   # too few events: split the longest phase at its midpoint
        b = [1] + bs + [T]; i = max(range(len(b) - 1), key=lambda i: b[i + 1] - b[i]); bs = sorted(bs + [(b[i] + b[i + 1]) // 2])
    return to_phases(bs, T), sorted(ev)


def mdl(F, ph):
    C = np.cumsum(segment.transition_counts(F), 0); return sum(segment._cost(C[p["end"] - 1] - C[p["start"] - 1]) for p in ph)


def optimized(F, K, moves=4):
    """From the cut-aligned boundaries, at most `moves` greedy operations (shift one boundary by +-1/2/4/8 frames, or merge
    two neighbouring phases), each accepted only if it lowers the clip's MDL code length (segment._cost)."""
    ph, ev = cut_aligned(F, K); T = len(F); bs = [p["start"] for p in ph[1:]]; cur = mdl(F, ph); log = []
    for _ in range(moves):
        cands = []
        for i in range(len(bs)):
            for d in (-8, -4, -2, -1, 1, 2, 4, 8):
                nb = sorted(bs[:i] + [bs[i] + d] + bs[i + 1:]); b = [1] + nb + [T]
                if all(b[j + 1] - b[j] >= LMIN3 for j in range(len(b) - 1)) and len(set(nb)) == len(nb): cands.append((nb, f"move {bs[i]}->{bs[i] + d}"))
            cands.append((bs[:i] + bs[i + 1:], f"merge at {bs[i]}"))
        best = min(cands, key=lambda c: mdl(F, to_phases(c[0], T))); v = mdl(F, to_phases(best[0], T))
        if v >= cur: break
        bs, cur = best[0], v; log.append(best[1])
    return to_phases(bs, T), log


def g3_one(args):
    task, K, place, seed = args; from exp45_c12 import target
    F = target(task).astype(np.uint8); t0 = time.process_time(); info = {}
    if place == "uniform": ph = segment.uniform_k(len(F), K)
    elif place == "cut": ph, info["events"] = cut_aligned(F, K)
    else: ph, info["ops"] = optimized(F, K)
    model, fi = fit_clip(F, K, "reuse", S, np.random.default_rng(seed), phases=[dict(start=p["start"], end=p["end"]) for p in ph])
    d = os.path.join(OUT, "g3"); os.makedirs(d, exist_ok=True)
    r = score_model(model, F, os.path.join(d, f"{task.replace(':', '_')}_K{K}_{place}_s{seed}.rule.json"), dict(task=task, K=K, placement=place, seed=seed))
    cx = rules.complexity(rules.load(os.path.join(d, f"{task.replace(':', '_')}_K{K}_{place}_s{seed}.rule.json")))
    return dict(task=task, K=K, placement=place, seed=seed, K_eff=len(ph), boundaries=[p["start"] for p in ph[1:]], reused=cx["reused_programs"],
                mdl_bits=round(mdl(F, ph), 1), **info, **r, phases=fi["phases"], cpu_s=round(time.process_time() - t0, 1))


def g3():
    done = set()
    if os.path.exists(os.path.join(OUT, "g3.jsonl")): done = {(r["task"], r["K"], r["placement"], r["seed"]) for r in map(json.loads, open(os.path.join(OUT, "g3.jsonl")))}
    jobs = [(t, K, pl, s) for t in G3_TASKS for K in (8, 16) for pl in ("uniform", "cut", "optimized") for s in range(3) if (t, K, pl, s) not in done]
    jobs.sort(key=lambda j: -int(j[0].split(":")[1]))   # long clips first
    with Pool(24) as p:
        for r in p.imap_unordered(g3_one, jobs): append("g3.jsonl", [r]); print(r["task"], r["K"], r["placement"], r["seed"], round(r["err"], 4), r["cpu_s"], flush=True)


# ---------- G2: local dynamics diagnostics on the G1 programs ----------
COUNTS = {"n8", "r2", "N", "S", "E", "W", "u", "d", "l", "r"}


def rule_kind(r):
    """always | grow (-> on when neighbour counts are high) | erode (-> off when counts are low) | kill/fill (fires
    almost regardless of neighbours) | other. Classified from the rule text only."""
    c = r["if"]; v = r["then"]["v"]
    if not c: return "always"
    nb = [x for x in c if x[0] in COUNTS]; own = [x for x in c if x[0] not in COUNTS]
    if not nb: return "fill" if v else "kill"
    if own and not all(x[0] == "v" for x in own): return "other"
    if v == 1 and all(op == ">=" for _, op, _ in nb): return "grow"
    if v == 0 and all(op == "<=" for _, op, _ in nb): return "erode"
    return "other"


def traced(p, v, h, n):
    """Rollout that attributes each changed pixel per frame to the rule that last changed it. Returns frames (n,H,W),
    fires (rules+1,), attr (n,H,W) rule index or -1."""
    out, attr = [], []; fires = np.zeros(len(p["rules"]) + 1, np.int64)
    for _ in range(n):
        v0 = v.copy(); who = np.full(v.shape, -1, np.int32)
        for _ in range(S):
            vb = v; v, h, f = rules.step(p, v, h, trace=True); fires += np.bincount(f.ravel() + 1, minlength=len(p["rules"]) + 1); who[v != vb] = f[v != vb]
        who[v == v0] = -1; out.append(v); attr.append(who)
    return np.stack(out), fires, np.stack(attr)


def feat_hist(V):
    """(v, n8) histogram (18 bins) of each frame, normalised."""
    out = []
    for v in V:
        f = rules.features(v, np.zeros((0,) + v.shape, np.uint8)); hh = np.bincount((f["v"] * 9 + f["n8"]).ravel(), minlength=18).astype(float); out.append(hh / hh.sum())
    return np.array(out)


def g2_one(row):
    X = video(); w = PRE["windows"][row["window"]]; p = row["program"]; k = row["k"]; nr = len(p["rules"]); res = dict(window=row["window"], method=row["method"], seed=row["seed"])
    res["kinds"] = [rule_kind(r) for r in p["rules"]]; per = {}
    for wn, (c, d), start in (("train", w["train"], True), ("adjacent", w["adjacent"], False), ("distant", w["distant"], False), ("cross", w["cross"], False)):
        G = X[c:d]; B, fires, attr = traced(p, G[0], z(k, G), len(G) - 1); T1 = G[1:]
        tot = max(1, fires.sum()); st = []
        for i in range(nr):
            m = attr == i; st.append(dict(fire_share=float(fires[i + 1] / tot), changed=int(m.sum()), precision=float((B[m] == T1[m]).mean()) if m.any() else None))
        base = float((B != T1).mean()); abl = []
        for i in range(nr):
            q = {"rules": p["rules"][:i] + p["rules"][i + 1:]}; abl.append(float((roll(q, G[0], z(k, G), len(G) - 1, S) != T1).mean()) - base)
        per[wn] = dict(err=base, default_share=float(fires[0] / tot), rules=st, ablation_delta=abl)
    sh = lambda wn: np.array([r["fire_share"] for r in per[wn]["rules"]] + [per[wn]["default_share"]])
    for wn in ("adjacent", "distant", "cross"): per[wn]["fire_tv_vs_train"] = float(0.5 * np.abs(sh(wn) - sh("train")).sum())
    res["per"] = per
    # greedy condition pruning on the training window (keep a removal if train error rises by <= 0.001)
    a, b = w["train"]; F = X[a:b]; f = lambda q: float((roll(q, F[0], z(k, F), len(F) - 1, S) != F[1:]).mean()); q = json.loads(json.dumps(p)); e0 = f(q); changed = True
    while changed:
        changed = False
        for i, r in enumerate(q["rules"]):
            for j in range(len(r["if"])):
                t = json.loads(json.dumps(q)); del t["rules"][i]["if"][j]
                if f(t) <= e0 + 0.001: q = t; changed = True; break
            if changed: break
    G = X[w["adjacent"][0]:w["adjacent"][1]]
    res["pruned"] = dict(conditions_before=NC(p), conditions_after=NC(q), train_err=f(q), train_err_before=e0,
                         adjacent_transfer_err=float((roll(q, G[0], z(k, G), len(G) - 1, S) != G[1:]).mean()), program=q)
    # local-feature drift along the continuous rollout: TV distance of (v, n8) histograms, rollout vs truth, and hold vs truth
    end = w["distant"][1]; C = X[a:end]; B = roll(p, C[0], z(k, C), len(C) - 1, S); hT, hB, hH = feat_hist(C[1:]), feat_hist(B), feat_hist(np.repeat(C[:1], len(C) - 1, 0))
    res["feature_tv"] = dict(rollout=[round(float(x), 4) for x in 0.5 * np.abs(hB - hT).sum(1)], hold=[round(float(x), 4) for x in 0.5 * np.abs(hH - hT).sum(1)])
    return res


def g2():
    rows = [r for r in map(json.loads, open(os.path.join(OUT, "g1.jsonl"))) if r["method"] != "shift"]
    with Pool(24) as p: append("g2.jsonl", p.map(g2_one, rows, chunksize=2))



# ---------- G4: radius-2 feasibility (exact SAT, zero boundary, k = 0, frame 0 given) ----------
# Families (a rule is a table; entries never observed stay free):
#   tot1  : (v, n8)            radius 1 per substep, 18 entries            = dlrule restricted to {v, n8}
#   tot2  : (v, n8, r2)        radius 2 per substep, 2*9*17 = 306 entries  = dlrule restricted to {v, n8, r2}
#   full2 : any 5x5 pattern    radius 2, up to 2^25 entries                (S = 1 only: at S = 2 the enumeration is 2^25 per cell)
# Neighbour counts of unknown (intermediate-substep) cells are built with exact unary counters, so clause count is
# polynomial: per cell and substep ~ 4*(8*9 + 16*17)/2 counter clauses + one clause per table entry.
G4_CAP = 4_000_000   # 24 workers x ~1 GB of Python clause lists stays well inside RAM


class Tot:
    def __init__(self, S, rings):
        self.S, self.rings, self.nv, self.cl, self.L, self.X = S, rings, 1, [[1]], {}, {}   # var 1 = TRUE

    def new(self): self.nv += 1; return self.nv

    def add(self, c):
        self.cl.append(c)
        if len(self.cl) > G4_CAP: raise OverflowError

    def lit(self, F, g, y, x):   # literal of visible bit at substep g (constant -> +-TRUE)
        T, H, W = F.shape
        if not (0 <= y < H and 0 <= x < W): return -1
        if g % self.S == 0: return 1 if F[g // self.S, y, x] else -1
        if (g, y, x) not in self.X: self.X[g, y, x] = self.new()
        return self.X[g, y, x]

    def counter(self, xs):
        """Unary s[j] <-> (sum xs >= j), j = 0..n (s[0] = TRUE). Constants fold out first."""
        ones = sum(1 for x in xs if x == 1); xs = [x for x in xs if abs(x) != 1]; s = [1] + [-1] * len(xs)
        for x in xs:   # s'[j] <-> s[j] or (s[j-1] and x)
            t = [1]
            for j in range(1, len(s)):
                a, b = s[j], s[j - 1]
                if a == 1 or (b == -1): t.append(a if b == -1 or a == 1 else a); continue
                o = self.new(); self.add([-o, a, b]); self.add([-o, a, x]); self.add([o, -a]); self.add([o, -b, -x]); t.append(o)
            s = t
        out = [1] * (ones + 1) + s[1:]; return out   # out[j] <-> sum >= j

    def eq(self, s, a):   # literals whose conjunction means count == a (as a clause-negation list)
        lo = s[a] if a < len(s) else -1; hi = s[a + 1] if a + 1 < len(s) else -1
        return lo, hi

    def extend(self, F):
        T, H, W = F.shape
        for g in range((T - 1) * self.S):
            for y in range(H):
                for x in range(W):
                    v = self.lit(F, g, y, x); o = self.lit(F, g + 1, y, x)
                    cnt = [self.counter([self.lit(F, g, y + dy, x + dx) for dy, dx in ring]) for ring in self.rings]
                    known = all(abs(c) == 1 for s_ in cnt for c in s_) and abs(v) == 1
                    if known:   # all inputs constant: one table entry
                        key = (int(v == 1),) + tuple(sum(c == 1 for c in s_[1:]) for s_ in cnt)
                        l = self.L.setdefault(key, None) or self.L.__setitem__(key, self.new()) or self.L[key]
                        if abs(o) == 1: self.add([l if o == 1 else -l])
                        else: self.add([-l, o]); self.add([l, -o])
                        continue
                    ranges = [range(len(s_)) for s_ in cnt]
                    for vb in (0, 1):
                        if (v == 1 and vb == 0) or (v == -1 and vb == 1): continue
                        for combo in __import__("itertools").product(*ranges):
                            pre = [] if abs(v) == 1 else [-v if vb else v]; ok = True
                            for s_, a in zip(cnt, combo):
                                lo, hi = self.eq(s_, a)
                                if lo == -1 or hi == 1: ok = False; break
                                if lo != 1: pre.append(-lo)
                                if hi != -1: pre.append(hi)
                            if not ok: continue
                            key = (vb,) + combo
                            if key not in self.L: self.L[key] = self.new()
                            l = self.L[key]
                            if abs(o) == 1: self.add(pre + [l if o == 1 else -l])
                            else: self.add(pre + [-l, o]); self.add(pre + [l, -o])
        return self


RING1 = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]
RING2 = [(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3) if max(abs(dy), abs(dx)) == 2]


def g4_solve(F, fam, S, timeout=120):
    from pysat.solvers import Glucose4
    import threading
    t0 = time.process_time()
    try:
        if fam == "full2":
            assert S == 1; L = {}; cl = []; T, H, W = F.shape; P = np.pad(F, ((0, 0), (2, 2), (2, 2)))
            for t in range(T - 1):
                for y in range(H):
                    for x in range(W):
                        key = P[t, y:y + 5, x:x + 5].tobytes(); L.setdefault(key, len(L) + 1); cl.append([L[key] if F[t + 1, y, x] else -L[key]])
            enc = None
        else:
            enc = Tot(S, [RING1] if fam == "tot1" else [RING1, RING2]).extend(F); cl, L = enc.cl, enc.L
    except OverflowError:
        return dict(status="SIZE", clauses=f">{G4_CAP}")
    enc_s = round(time.process_time() - t0, 1); w0 = time.time()
    with Glucose4(bootstrap_with=cl) as s:
        tm = threading.Timer(timeout, s.interrupt); tm.start(); ok = s.solve_limited(expect_interrupt=True); tm.cancel()
        model = set(x for x in (s.get_model() or []) if x > 0) if ok else None
    r = dict(status="SAT" if ok else ("UNSAT" if ok is False else "TIMEOUT"), clauses=len(cl), entries_seen=len(L), encode_cpu_s=enc_s, solve_wall_s=round(time.time() - w0, 2))
    if ok and fam != "full2":   # export as a dlrule program over {v, n8, r2} and replay it with the shared executor
        names = ["n8"] if fam == "tot1" else ["n8", "r2"]; rs = []
        for key, var in sorted(L.items()):
            out = int(var in model)
            if out != key[0]: rs.append({"if": [["v", "==", key[0]]] + [[n, "==", int(a)] for n, a in zip(names, key[1:])], "then": {"v": out, "h": 0}})
        p = {"rules": rs}; B = roll(p, F[0], np.zeros((0,) + F.shape[1:], np.uint8), len(F) - 1, S)
        r.update(rules=len(rs), replay_exact_in_dlrule=bool((B == F[1:]).all()), program=p)
    return r


def g4_control(seed, T=8, n=16):
    """Synthetic: a radius-2 outer-totalistic rule (birth if n8 + r2 in 6..8, survive if n8 + r2 in 4..10), S = 2."""
    rng = np.random.default_rng(seed); v = (rng.random((n, n)) < 0.35).astype(np.uint8); fr = [v]
    p = {"rules": [{"if": [["v", "==", 0], ["n8", ">=", 3], ["r2", ">=", 3]], "then": {"v": 1, "h": 0}},
                   {"if": [["v", "==", 1], ["n8", "<=", 1]], "then": {"v": 0, "h": 0}},
                   {"if": [["v", "==", 1], ["r2", ">=", 10]], "then": {"v": 0, "h": 0}}]}
    return np.concatenate([v[None], roll(p, v, np.zeros((0, n, n), np.uint8), T - 1, 2)]), p


def g4_one(args):
    task, size, fam, S = args; X = video(); rows = []
    for T in (2, 3, 4, 6, 8):
        if task.startswith("control"):
            F, truth = g4_control(int(task[-1]), T)
        else:
            from rr_scale import crop
            a = int(task); _, yx = crop(X, a, 16, size); F, _ = crop(X, a, T, size, yx)
        r = g4_solve(F.astype(np.uint8), fam, S); r.update(task=task, size=size or "48x36", family=fam, S=S, T=T); rows.append(r)
        if task.startswith("control") and r["status"] == "SAT":   # held-out: same rule on two new random starts
            p = r["program"]; held = []
            for s2 in (7, 8):
                G, _ = g4_control(s2, 8); B = roll(p, G[0], np.zeros((0,) + G.shape[1:], np.uint8), len(G) - 1, S); held.append(float((B != G[1:]).mean()))
            r["held_out_err"] = held
        append("g4.jsonl", [{k: v for k, v in r.items() if k != "program"} | ({"program": r["program"]} if "program" in r and r.get("rules", 999) <= 60 else {})])
        if r["status"] != "SAT": break
    return rows


def g4():
    jobs = [(f"control{s}", None, fam, 2) for s in (0, 1) for fam in ("tot1", "tot2")]
    for t in ("1221", "4500", "1499", "1678"):
        for size in (8, 16, None):
            jobs += [(t, size, "tot1", 2), (t, size, "tot2", 2), (t, size, "tot2", 1), (t, size, "full2", 1)]
    with Pool(24) as p: list(p.imap_unordered(g4_one, jobs))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); globals()[sys.argv[1]]()
