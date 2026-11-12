"""Readable Rules, step 3 (synthesis & scaling; CPU only). Results in runs/readable_rules/scale/.
  a1  : learned initial hidden state. Exact SAT for ONE fixed rule (K=1) + the initial hidden bits, under limited seed
        representations (ordered by free bits): const (one value per channel), tile p (p x p pattern repeated), block b
        (one value per b x b block = low-resolution seed), cell (one value per cell). Hidden bits are seen only by the
        cell itself (family_taps(k, 'self')); families 'tot' (own v, #on Moore neighbours, own hidden) and 'full' (any 3x3).
        (1) synthetic controls with a hidden generator (sprout k=1, timer k=2): recover rule + seed, then test the rule on
            held-out runs (rule entries seen in training pinned, only the seed free);
        (2) Bad Apple crops (translate 1221, contour 4500; 8x8, 16x16, 48x36): per (k, S, family) the clip length grows
            and, per length, the smallest representation that is SAT is recorded (a representation UNSAT at length T is
            skipped for longer clips). For each SAT result: seed bits, zlib bytes of the seed, rule entries used, the
            rule run on to 16 frames (outside the fitted window; reported only) and the same rule with the seed zeroed.
  a2  : rollout-only search vs two-stage search (stage 1: 4 short climbs on the teacher-forced one-frame error from the
        empty program -> 4 candidates; candidate chosen by short-rollout error on the training window; stage 2: rollout
        refinement with objective full | staged (short rollouts, then full) | mixed (0.5 full + 0.5 short)). Same total
        evaluations (4,000) for every method. Windows: search [a, a+n), selection [a+n, a+2n), final [a+2n, a+3n); each
        window starts from its own true first frame. Per method the seed is picked on the selection window only.
  c1  : phased rules on longer clips, K = 1, 4, 8, 16, with the state carried across phases:
        indep (every phase searched from the empty program), shared (one base rule fitted on the whole clip; each phase =
        base + edits, an edit must pay 0.002 pixel error), reuse (shared, then each phase reuses an earlier program when
        that costs <= 0.003 error). Search method: the A2 winner on the controls (two-stage, staged), chosen after A2.
  ac  : 4-way: fixed rule vs shared phased (K=8) x zero hidden vs learned hidden (k=1, a 6x6-block seed searched jointly
        with the rule: 48 bits at 48x36), plus the learned seed zeroed.
usage: rr_scale.py a1ctl|a1|a2|c1|ac"""
import json, os, sys, threading, time, zlib
from multiprocessing import Pool
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca import rules
from ca.rule import family_taps, rollout as lut_roll
from ca.sat import Encoder

OUT = os.path.join(ROOT, "runs", "readable_rules", "scale"); os.makedirs(OUT, exist_ok=True)
CAP = 6_000_000


def reps(H, W):
    """Seed representations ordered by free bits per hidden channel. 'zero' = the old constant-0 start."""
    r = [("zero", 0, 0), ("const", lambda y, x: 0, 1)]
    for p in (2, 4): r.append((f"tile{p}", (lambda p: lambda y, x: (y % p, x % p))(p), p * p))
    for b in (12, 8, 6, 4, 3, 2):
        if b < max(H, W): r.append((f"block{b}", (lambda b: lambda y, x: (y // b, x // b))(b), -(-H // b) * -(-W // b)))
    r.append(("cell", lambda y, x: (y, x), H * W))
    out, seen = [], set()
    for n, f, u in sorted(r, key=lambda t: t[2]):
        if u not in seen and u <= H * W: out.append((n, f, u)); seen.add(u)
    return out


class TooBig(Exception): pass


def tot_key(e, k): return ((e >> 4) & 1, bin(e & 0b111101111).count("1"), e >> 9)


def solve(F, fam, k, S, rep, timeout=30, pin=None, occam=0):
    """Exact SAT for one rule + seed. pin: {(c, e): bit} LUT entries fixed in advance. occam > 0: after SAT, MaxSAT (LSU,
    anytime, occam seconds) minimising table entries that differ from 'keep state' plus seed bits set to 1 -- the simplest
    rule + seed that still replays exactly. Returns dict (+ private _luts/_h0)."""
    from pysat.solvers import Glucose4   # interruptible; CaDiCaL ignores interrupt, so the timeout never fired (FINDINGS §9a)
    taps = family_taps(k, "self"); cl = []
    def sink(c):
        cl.append(c)
        if len(cl) > CAP: raise TooBig()
    t0 = time.process_time(); clips = F if isinstance(F, list) else [F]; encs = []
    try:   # several clips share one rule: later encoders reuse the LUT variables (shared dict) and continue the numbering
        for Fi in clips:
            e_ = Encoder(taps, k + 1, S, 0, rep, sink)
            if encs: e_.nv, e_.L = encs[-1].nv, encs[-1].L
            e_.extend(Fi); encs.append(e_)
    except TooBig: return dict(status="SIZE")
    L = encs[0].L
    if fam == "tot":
        g = {}
        for (c, e), var in L.items(): g.setdefault((c,) + tot_key(e, k), []).append(var)
        for vs in g.values():
            for a in vs[1:]: cl += [[-vs[0], a], [vs[0], -a]]
    for (c, e), b in (pin or {}).items():
        if (c, e) in L: cl.append([L[c, e] if b else -L[c, e]])
    enc_s = round(time.process_time() - t0, 1); w0 = time.time()
    with Glucose4(bootstrap_with=cl) as s:
        tm = threading.Timer(timeout, s.interrupt); tm.start(); ok = s.solve_limited(expect_interrupt=True); tm.cancel()
        model = set(x for x in (s.get_model() or []) if x > 0) if ok else None
    cost = None
    if ok and occam:
        from pysat.formula import WCNF
        from pysat.examples.lsu import LSU
        w = WCNF(); [w.append(c) for c in cl]
        for (c, e), var in L.items():
            keep = (e >> 4) & 1 if c == 0 else (e >> (8 + c)) & 1; w.append([var if keep else -var], weight=1)
        for enc in encs:
            for key, var in enc.X.items():
                if key[0] == "h0": w.append([-var], weight=1)
        m = LSU(w, solver="g4", expect_interrupt=True)   # g4: interruptible (CaDiCaL ignores the time limit here)
        tm = threading.Timer(occam, m.interrupt); tm.start(); m.solve(); tm.cancel()
        try: mm = m.get_model()
        except AttributeError: mm = None   # interrupted before the first model: keep the plain SAT solution
        if mm is not None: model = set(x for x in mm if x > 0); cost = dict(cost=m.cost, optimal=bool(m.found_optimum()))
        else: cost = dict(cost=None, optimal=False)
        m.delete()
    out = dict(status="SAT" if ok else ("UNSAT" if ok is False else "TIMEOUT"), clauses=len(cl), encode_cpu_s=enc_s, solve_wall_s=round(time.time() - w0, 2))
    if ok:
        n_e = 2 ** len(taps); luts = np.zeros((k + 1, n_e), np.uint8)
        for (c, e), var in L.items(): luts[c, e] = var in model
        h0s, seen, exact = [], set(), True
        for Fi, enc in zip(clips, encs):
            T, H, W = Fi.shape; h0 = np.zeros((k + 1, H, W), np.uint8)
            if callable(rep):
                for c in range(1, k + 1):
                    for y in range(H):
                        for x in range(W):
                            var = enc.X.get(("h0", c, rep(y, x)))
                            h0[c, y, x] = var in model if var else 0
            vis, _, sn = lut_roll(Fi[0], taps, luts, T, S, hidden_init=h0); seen |= set(sn); exact &= bool((vis == Fi).all()); h0s.append(h0)
        h0 = h0s[0] if len(h0s) == 1 else np.concatenate([x.reshape(k + 1, -1) for x in h0s], 1); seen = sorted(seen)
        out.update(exact=exact, entries_used=len(seen), entries_not_keep=int(sum(any(luts[c, e] != ((e >> 4) & 1 if c == 0 else (e >> (8 + c)) & 1) for c in range(k + 1)) for e in seen)),
                   seed_ones=int(h0[1:].sum()), occam=cost, _luts=luts, _h0=h0, _seen=seen)
        if fam == "tot": out["tot_entries_used"] = len({(c,) + tot_key(e, k) for c in range(k + 1) for e in seen})
    return out


def seed_bytes(h0, k):
    return len(zlib.compress(np.packbits(h0[1:k + 1].ravel()).tobytes(), 9)) if k else 0


# ---------------- synthetic controls (generator hidden from the solver) ----------------
GEN = {
    "sprout": (1, {"rules": [{"if": [["h0", "==", 1]], "then": {"v": 1, "h": 0}},
                             {"if": [["v", "==", 0], ["n8", ">=", 3]], "then": {"v": 1, "h": 0}}]}),
    "timer": (2, {"rules": [{"if": [["v", "==", 0], ["h0", "==", 1], ["h1", "==", 0]], "then": {"v": 1, "h": 0}},
                            {"if": [["h0", "==", 0], ["h1", "==", 1]], "then": {"v": 0, "h": 1}},
                            {"if": [["h0", "==", 1], ["h1", "==", 1]], "then": {"v": 0, "h": 2}},
                            {"if": [["v", "==", 1], ["n8", "<=", 1]], "then": {"v": 0, "h": 0}}]}),
}


def gen_run(name, seed, T=10, H=16, W=16):
    k, prog = GEN[name]; rng = np.random.default_rng(seed)
    f0 = np.zeros((H, W), np.uint8)
    for _ in range(2): y, x = rng.integers(0, H - 4, 2); f0[y:y + 3, x:x + 3] = 1
    h0 = np.zeros((k, H, W), np.uint8)
    hv = rng.integers(0, 2 ** k, (H, W)) * (rng.random((H, W)) < 0.08)
    for b in range(k): h0[b] = (hv >> b) & 1
    m = dict(k=k, S=1, T=T, programs=[prog], schedule=[dict(start=1, end=T, program=0)], first=f0, h0=h0)
    return rules.run(m)[0].astype(np.uint8), h0


def gen_lut(name):
    """The generator as a LUT over (3x3 visible, own hidden), for functional comparison on the entries actually used."""
    k, prog = GEN[name]; taps = family_taps(k, "self"); luts = np.zeros((k + 1, 2 ** len(taps)), np.uint8)
    for e in range(2 ** len(taps)):
        v = np.array([(e >> i) & 1 for i in range(9)], np.uint8).reshape(3, 3); h = np.zeros((k, 3, 3), np.uint8)
        for b in range(k): h[b, 1, 1] = (e >> (9 + b)) & 1
        v2, h2, _ = rules.step(prog, v, h); luts[0, e] = v2[1, 1]
        for b in range(k): luts[b + 1, e] = h2[b, 1, 1]
    return luts


def a1ctl():
    res = []
    for name in GEN:
        k = GEN[name][0]; F = [gen_run(name, s, T=12, H=32, W=32)[0] for s in (10, 11, 12, 13)]; T, H, W = F[0].shape; G = gen_lut(name)
        for fam, occam in (("tot", 0), ("full", 0), ("tot", 120), ("full", 120)):
            for rn, rf, units in reps(H, W):
                if rn not in ("zero", "const", "cell"): continue   # the generator's seed is per cell; coarse seeds were UNSAT above
                r = solve(F, fam, k, 1, rf, timeout=60, occam=occam); luts, h0, seen = r.pop("_luts", None), r.pop("_h0", None), r.pop("_seen", None)
                row = dict(control=name, k=k, family=fam, occam_s=occam, rep=rn, train_runs=len(F), seed_bits=len(F) * units * k if rn != "zero" else 0, **r)
                if r["status"] == "SAT":
                    vis_seen = [(c, e) for c in range(k + 1) for e in seen]
                    row["agrees_with_generator_on_used_entries"] = float(np.mean([luts[c, e] == G[c, e] for c, e in vis_seen]))
                    row["seed_bytes_zlib"] = seed_bytes(h0, k)
                    # held-out runs: the WHOLE rule pinned (entries unused in training = keep state, the dlrule default); only the seed is free
                    # 'tot': an unseen entry takes the value of its (own v, #neighbours, own hidden) class if that class was seen
                    pin, ss = {}, set(seen); cls = {(c,) + tot_key(e, k): int(luts[c, e]) for c in range(k + 1) for e in ss}
                    for e in range(luts.shape[1]):
                        for c in range(k + 1):
                            keep = (e >> 4) & 1 if c == 0 else (e >> (8 + c)) & 1
                            pin[c, e] = int(luts[c, e]) if e in ss else cls.get((c,) + tot_key(e, k), keep) if fam == "tot" else keep
                    held = []
                    for s in (1, 2, 3):
                        Fh, _ = gen_run(name, s); rh = solve(Fh, "full", k, 1, lambda y, x: (y, x), timeout=60, pin=pin)
                        held.append(dict(seed=s, status=rh["status"]))
                    row["held_out"] = held
                res.append(row); print(json.dumps(row), flush=True)
                if r["status"] == "SAT": break
    json.dump(res, open(os.path.join(OUT, "a1_controls.json"), "w"), indent=1)


# ---------------- Bad Apple ----------------
def crop(full, a, n, size, yx=None):
    if size is None: return full[a:a + n], (0, 0)
    if yx is None:
        ch = (full[a + 1:a + 16] != full[a:a + 15]).sum(0).astype(float); H, W = ch.shape; best = -1
        for y in range(0, H - size + 1, 2):
            for x in range(0, W - size + 1, 2):
                s = ch[y:y + size, x:x + size].sum()
                if s > best: best, yx = s, (y, x)
    y, x = yx; return full[a:a + n, y:y + size, x:x + size], yx


def a1_ladder(args):
    task, a, size, fam, k, S = args
    from exp45_c12 import target
    full = target("0:6572").astype(np.uint8); rows = []; dead = set(); _, yx = crop(full, a, 16, size)
    for T in (2, 3, 4, 6, 8, 12, 16):
        F, _ = crop(full, a, T, size, yx); H, W = F.shape[1:]; G, _ = crop(full, a, 16, size, yx); got = None
        for rn, rf, units in reps(H, W):
            if rn in dead or (rn == "zero" and k): continue
            r = solve(F, fam, k, S, rf, occam=20); luts, h0, seen = r.pop("_luts", None), r.pop("_h0", None), r.pop("_seen", None)
            row = dict(task=task, crop=size or "48x36", at=list(yx), family=fam, k=k, S=S, T=T, rep=rn, seed_bits=units * k if rn != "zero" else 0, **r)
            if r["status"] == "SAT":
                taps = family_taps(k, "self"); vis = lut_roll(G[0], taps, luts, 16, S, hidden_init=h0)[0]
                vz = lut_roll(F[0], taps, luts, T, S, hidden_init=0)[0]
                row.update(seed_bytes_zlib=seed_bytes(h0, k), clip_bits=int((T - 1) * H * W), err_16=float((vis[1:] != G[1:]).mean()), hold_16=float((G[1:] != G[0]).mean()),
                           err_after_window=float((vis[T:] != G[T:]).mean()) if T < 16 else None, hold_after_window=float((G[T:] != G[T - 1]).mean()) if T < 16 else None,
                           err_window_seed_zeroed=float((vz[1:] != F[1:]).mean()))
                got = row
            elif r["status"] == "UNSAT": dead.add(rn)
            rows.append(row); print(json.dumps(row), flush=True)
            if got: break
        if not got: break
    return rows


def a1():
    jobs = [(t, a, size, fam, k, S) for t, a in (("translate", 1221), ("contour", 4500)) for size in (8, 16, None) for fam in ("tot", "full") for k in (1, 2, 3) for S in (1, 2, 4)]
    jobs.sort(key=lambda j: (j[2] is None, j[2] or 0, j[5]))   # small crops first
    with Pool(20) as p: res = p.map(a1_ladder, jobs, chunksize=1)
    json.dump([r for rs in res for r in rs], open(os.path.join(OUT, "a1_badapple.json"), "w"), indent=1)


# ---------------- A2: search methods ----------------
from ca import rr_search
from rr_run import control, evaluate
NC = lambda p: sum(len(r["if"]) for r in p["rules"])


def roll(p, v, h, n, S): return rr_search.rollout(p, v.astype(np.uint8), h, n, S)[0]


def objectives(F, S, k=0, h0=None):
    """Objective functions on one training window F (frame 0 given). Hidden starts at h0 (or zero) for full rollouts;
    one-step and short rollouts restart from true frames with a zero hidden state."""
    H, W = F.shape[1:]; z = np.zeros((k, H, W), np.uint8); h0 = z if h0 is None else h0; n = len(F)
    full = lambda p: float((roll(p, F[0], h0, n - 1, S) != F[1:]).mean())
    one = lambda p: float(np.mean([(roll(p, F[t], z, 1, S)[0] != F[t + 1]).mean() for t in range(n - 1)]))
    short = lambda p: float(np.mean([(roll(p, F[t], z, min(4, n - 1 - t), S) != F[t + 1:t + 5]).mean() for t in range(0, n - 1, 4)]))
    return dict(full=full, one=one, short=short, mixed=lambda p: 0.5 * full(p) + 0.5 * short(p))


def climb(start, f, evals, rng, k=0, restarts=4, max_rules=8, max_conds=3):
    fit = lambda p: f(p) + 1e-4 * NC(p)
    best = start; bf = fit(best); cur, cf = best, bf; per = max(1, evals // restarts)
    for i in range(1, evals):
        if i % per == 0: cur, cf = best, bf
        c = rr_search.mutate(cur, rng, k, max_rules, max_conds); fc = fit(c)
        if fc <= cf:
            cur, cf = c, fc
            if fc < bf: best, bf = c, fc
    return best


def method(name, F, S, rng, E=4000, k=0):
    O = objectives(F, S, k); empty = {"rules": []}
    if name == "rollout_only": return climb(empty, O["full"], E, rng, k)
    cands = [climb(empty, O["one"], E // 8, rng, k, restarts=1) for _ in range(4)]   # stage 1: one-step proposals
    c = min(cands, key=lambda p: O["short"](p) + 1e-4 * NC(p))                          # chosen on the training window
    if name == "two_stage_full": return climb(c, O["full"], E // 2, rng, k)
    if name == "two_stage_mixed": return climb(c, O["mixed"], E // 2, rng, k)
    if name == "two_stage_staged": return climb(climb(c, O["short"], E // 4, rng, k), O["full"], E // 4, rng, k)
    raise ValueError(name)


A2_METHODS = ("rollout_only", "two_stage_full", "two_stage_staged", "two_stage_mixed")
A2_STARTS = (1221, 4500, 900, 2100, 3300, 5700)   # fixed before any A2 result: the two earlier clips + 4 spread over the song


def a2_one(args):
    task, n, meth, seed = args; rng = np.random.default_rng(seed); t0 = time.process_time()
    if task in ("life", "shift"):
        F, _ = control(task, seed=0); S = 1; p = method(meth, F, S, rng)
        held = [float((roll(p, G[0], np.zeros((0,) + G.shape[1:], np.uint8), len(G) - 1, S) != G[1:]).mean()) for G in (control(task, seed=s)[0] for s in (1, 2))]
        return dict(task=task, n=len(F), method=meth, seed=seed, train=objectives(F, S)["full"](p), one_step=objectives(F, S)["one"](p), held_out=held, conditions=NC(p),
                    cpu_s=round(time.process_time() - t0, 1), program=p)
    from exp45_c12 import target
    a = int(task); X = target(f"{a}:{3 * n}").astype(np.uint8); S = 2; F = X[:n]; p = method(meth, F, S, rng); row = dict(task=task, n=n, method=meth, seed=seed)
    for wn, W_ in (("train", X[:n]), ("select", X[n:2 * n]), ("final", X[2 * n:])):
        B = roll(p, W_[0], np.zeros((0,) + W_.shape[1:], np.uint8), n - 1, S); ev = evaluate(np.concatenate([W_[:1], B]), W_)
        row[wn] = dict(err=ev["err"], hold=ev["hold_err"], lag4_f1=ev["lag4_f1"], mass=ev["lag4_mass"])
    row.update(one_step=objectives(F, S)["one"](p), conditions=NC(p), cpu_s=round(time.process_time() - t0, 1), program=p)
    return row


def a2():
    jobs = [(t, 24, m, s) for t in ("life", "shift") for m in A2_METHODS for s in range(3)] + \
           [(str(a), n, m, s) for a in A2_STARTS for n in (16, 32) for m in A2_METHODS for s in range(3)]
    with Pool(24) as p: res = p.map(a2_one, jobs, chunksize=1)
    json.dump(res, open(os.path.join(OUT, "a2.json"), "w"), indent=1, default=float)


# ---------------- C1 / AC: phased rules, learned seed ----------------
def edits_n(base, p): return len(rr_search.diff(base, p))


def fit_clip(F, K, mode, S, rng, k=0, seed_block=None, E_base=4000, E_phase=1500, lam=0.002, eps=0.003, phases=None):
    """Returns (model, info). mode: indep | shared | reuse. seed_block: learn a k-bit seed at this block size (else zero).
    phases: explicit [{start, end}] boundaries (round 4, G3); default uniform K."""
    T, H, W = F.shape
    if phases is None: phases = [dict(start=1, end=T)] if K == 1 else __import__("ca.segment", fromlist=["x"]).uniform_k(T, K)
    K = len(phases)
    seed = None if seed_block is None else np.zeros((k, -(-H // seed_block), -(-W // seed_block)), np.uint8)
    up = lambda sd: np.zeros((k, H, W), np.uint8) if sd is None else sd.repeat(seed_block, 1).repeat(seed_block, 2)[:, :H, :W]
    def climb_seed(p, f):   # joint search over program and seed bits (seed flips are 20 % of the moves)
        nonlocal seed
        fit = lambda q, sd: f(q, sd) + 1e-4 * NC(q)
        best, bs = p, seed; bf = fit(best, bs); cur, cs, cf = best, bs, bf
        for i in range(1, E_base):
            if i % (E_base // 4) == 0: cur, cs, cf = best, bs, bf
            if bs is not None and rng.random() < 0.2:
                c, sd = cur, cs.copy(); j = tuple(rng.integers(0, m) for m in sd.shape); sd[j] ^= 1
            else: c, sd = rr_search.mutate(cur, rng, k, 8, 3), cs
            fc = fit(c, sd)
            if fc <= cf:
                cur, cs, cf = c, sd, fc
                if fc < bf: best, bs, bf = c, sd, fc
        seed = bs; return best
    # stage 1 (one-step proposals on the whole clip), then short rollouts, then the base rule by full rollout with the seed
    O = objectives(F, S, k)
    cands = [climb({"rules": []}, O["one"], E_base // 8, rng, k, restarts=1) for _ in range(4)]
    c0 = min(cands, key=lambda p: O["short"](p) + 1e-4 * NC(p))
    def clip_obj(q, sd): return float((roll(q, F[0], up(sd), T - 1, S) != F[1:]).mean())
    c0 = climb(c0, O["short"], E_base // 4, rng, k)   # A2 winner ('staged'): one-step -> short rollouts -> full rollout
    base = climb_seed(c0, clip_obj); h0 = up(seed)
    progs, sched, recs = [], [], []
    v, h = F[0].copy(), h0.copy()
    for pi, ph in enumerate(phases):
        a, b = ph["start"], ph["end"]; Fp = F[a - 1:b]   # Fp[0] = the frame before the phase (state given by the carry)
        def ph_err(q, v=v, h=h): return float((roll(q, v, h, b - a, S) != Fp[1:]).mean())
        if K == 1: q = base
        elif mode == "indep": q = climb({"rules": []}, ph_err, E_phase, rng, k)
        else: q = climb(base, lambda q: ph_err(q) + lam * edits_n(base, q), E_phase, rng, k)
        if mode == "reuse" and progs:
            own = ph_err(q); best_j = min(range(len(progs)), key=lambda j: ph_err(progs[j]))
            if ph_err(progs[best_j]) <= own + eps: q = progs[best_j]
        if q in progs: j = progs.index(q)
        else: progs.append(q); j = len(progs) - 1
        sched.append(dict(start=a, end=b, program=j)); recs.append(dict(phase=pi, err=ph_err(q), program=j, edits=edits_n(base, q)))
        Bp, v, h = rr_search.rollout(q, v, h, b - a, S)
    stored = [progs[0]] + [({"base": 0, "edits": rr_search.diff(progs[0], q)} if mode != "indep" else q) for q in progs[1:]]
    if mode != "indep" and K > 1:   # store every program as base + edits; the base itself is program 0
        stored = [base] + [{"base": 0, "edits": rr_search.diff(base, q)} for q in progs]; sched = [dict(x, program=x["program"] + 1) for x in sched]
    model = dict(k=k, S=S, T=T, programs=stored, schedule=sched, first=F[0].astype(np.uint8), h0=h0 if k else None)
    return model, dict(phases=recs, seed_bits=0 if seed is None else int(seed.size), seed_ones=0 if seed is None else int(seed.sum()))


def score_model(model, F, path=None, meta=None):
    if path: rules.export(path, model, meta=meta); model = rules.load(path)   # always score the export
    B, _ = rules.run(model); ev = evaluate(B, F); cx = rules.complexity(model)
    sb = len(json.dumps(model["schedule"], separators=(",", ":")))
    out = dict(err=ev["err"], hold=ev["hold_err"], worst16=ev["worst16"], lag4_f1=ev["lag4_f1"], mass=ev["lag4_mass"], blips=ev["spurious_blips"],
               programs=cx["programs"], distinct_used=cx["distinct_programs_used"], rules=cx["rules"], conditions=cx["conditions"], edits=cx["edits"],
               h0_bytes_zlib=len(zlib.compress(np.packbits(model["h0"].ravel()).tobytes(), 9)) if model.get("h0") is not None else 0,
               schedule_bytes=sb, file_bytes=cx["file_bytes"])
    if model.get("h0") is not None and model["h0"].any():
        Bz, _ = rules.run(dict(model, h0=np.zeros_like(model["h0"]))); out["err_seed_zeroed"] = float((Bz[1:] != F[1:]).mean())
    return out


C1_TASKS = ("1221:120", "4500:120", "1760:240")


def c1_one(args):
    task, K, mode, seed = args
    from exp45_c12 import target
    F = target(task).astype(np.uint8); t0 = time.process_time()
    model, info = fit_clip(F, K, mode, 2, np.random.default_rng(seed))
    d = os.path.join(OUT, "c1"); os.makedirs(d, exist_ok=True)
    r = score_model(model, F, os.path.join(d, f"{task.replace(':', '_')}_K{K}_{mode}_s{seed}.rule.json"), dict(task=task, K=K, mode=mode, seed=seed))
    return dict(task=task, K=K, mode=mode, seed=seed, **r, cpu_s=round(time.process_time() - t0, 1), phases=info["phases"])


def c1():
    jobs = [(t, K, m, s) for t in C1_TASKS for K in (1, 4, 8, 16) for m in (("indep",) if K == 1 else ("indep", "shared", "reuse")) for s in range(2)]
    with Pool(24) as p: res = p.map(c1_one, jobs, chunksize=1)
    json.dump(res, open(os.path.join(OUT, "c1.json"), "w"), indent=1, default=float)


def ac_one(args):
    task, K, learned, seed = args
    from exp45_c12 import target
    F = target(task).astype(np.uint8); t0 = time.process_time()
    model, info = fit_clip(F, K, "shared", 2, np.random.default_rng(seed), k=1, seed_block=6 if learned else None)
    d = os.path.join(OUT, "ac"); os.makedirs(d, exist_ok=True)
    r = score_model(model, F, os.path.join(d, f"{task.replace(':', '_')}_K{K}_{'learned' if learned else 'zero'}_s{seed}.rule.json"), dict(task=task, K=K, learned_seed=learned, seed=seed))
    return dict(task=task, K=K, learned_seed=learned, seed=seed, **r, **{k_: v for k_, v in info.items() if k_ != "phases"}, cpu_s=round(time.process_time() - t0, 1))


def ac():
    jobs = [(t, K, l, s) for t in ("1221:60", "4500:56", "1760:120") for K in (1, 8) for l in (False, True) for s in range(2)]
    with Pool(24) as p: res = p.map(ac_one, jobs, chunksize=1)
    json.dump(res, open(os.path.join(OUT, "ac.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    dict(a1ctl=a1ctl, a1=a1, a2=a2, c1=c1, ac=ac)[sys.argv[1]]()
