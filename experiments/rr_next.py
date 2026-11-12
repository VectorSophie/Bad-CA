"""Readable Rules, step 2: why rollout search fails, and whether SAT synthesis scales (CPU only).
  life      : objective landscape around the exact Life rule (B3/S23 as a 3-rule dlrule) on the blind-recovery control:
              every single edit of it, every prefix path from the empty program, random programs; scored by rollout error
              at horizons 1 (= teacher-forced one step), 4 and 23 frames. Then rollout search from (a) each 1-edit
              neighbour and (b) the empty program at 2k / 8k / 32k evaluations: budget vs objective structure.
  sat       : exact SAT synthesis on small Bad Apple crops with growing size and length. Families: 'tot' (outer-totalistic:
              own v, number of on Moore neighbours, own hidden bits -> at most 2*9*2^k table entries, readable as a short
              rule list) and 'full' (any 3x3 + own hidden function). Per (task, family, k, S, crop) the length grows until
              UNSAT / TIMEOUT / SIZE (encoding over the clause cap); UNSAT at length T implies UNSAT for every longer clip.
  lightcone : concrete witnesses for the light-cone proof (two cells with identical radius-R frame-0 patches whose frame-1
              values differ) + an empirical check with random radius-2 dlrule programs.
  stab      : one fixed rule (k=0, S=2, empty start) on 16/32/64 frames; objective 'full' (one rollout over the clip) vs
              'mixed' (0.5 full + 0.5 mean of 4-frame rollouts restarted from the original every 4 frames). Selection uses
              the training objective only; the per-frame teacher-forced vs rollout gap and the next unseen clip are reported.
usage: rr_next.py life|sat|lightcone|stab"""
import copy, json, os, sys, threading, time
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca import rules, rr_search
from rr_run import control, evaluate

OUT = os.path.join(ROOT, "runs", "readable_rules", "next"); os.makedirs(OUT, exist_ok=True)
LIFE = rr_search.LIBRARY["life_b3s23"]


def roll(prog, f0, n, S=1, k=0, h0=None):
    v, h = f0.astype(np.uint8), (np.zeros((k,) + f0.shape, np.uint8) if h0 is None else h0); out = []
    for _ in range(n):
        for _ in range(S): v, h, _ = rules.step(prog, v, h)
        out.append(v)
    return np.stack(out)


def life():
    F, _ = control("life", seed=0); T = len(F)
    def sc(p):
        B = roll(p, F[0], T - 1)
        one = np.mean([(roll(p, F[t], 1)[0] != F[t + 1]).mean() for t in range(T - 1)])
        return dict(h1=float(one), h4=float((B[:4] != F[1:5]).mean()), h23=float((B != F[1:]).mean()))
    res = dict(exact=sc(LIFE), hold=sc({"rules": []}))
    neigh = []
    for i, r in enumerate(LIFE["rules"]):
        for j, c in enumerate(r["if"]):
            for nv in (c[2] - 1, c[2] + 1):
                if 0 <= nv <= rr_search.frange(c[0]): p = copy.deepcopy(LIFE); p["rules"][i]["if"][j][2] = nv; neigh.append((f"R{i}.c{j} const {c[2]}->{nv}", p))
            for op in ("<=", ">=", "=="):
                if op != c[1]: p = copy.deepcopy(LIFE); p["rules"][i]["if"][j][1] = op; neigh.append((f"R{i}.c{j} op {c[1]}->{op}", p))
            p = copy.deepcopy(LIFE); del p["rules"][i]["if"][j]; neigh.append((f"R{i} drop c{j}", p))
        p = copy.deepcopy(LIFE); del p["rules"][i]; neigh.append((f"drop R{i}", p))
        p = copy.deepcopy(LIFE); p["rules"][i]["then"] = {"v": 1 - r["then"]["v"], "h": 0}; neigh.append((f"R{i} flip output", p))
    res["neighbours"] = [dict(edit=n, **sc(p)) for n, p in neigh]
    import itertools
    res["paths"] = []
    for order in itertools.permutations(range(3)):
        steps = [sc({"rules": [LIFE["rules"][i] for i in sorted(order[:m])]}) for m in range(4)]
        res["paths"].append(dict(order=list(order), h23=[s["h23"] for s in steps], h1=[s["h1"] for s in steps]))
    rng = np.random.default_rng(1); rand = []
    for _ in range(200):
        p = {"rules": []}
        for _ in range(int(rng.integers(1, 5))): p = rr_search.mutate(p, rng, 0, 8, 3)
        for _ in range(6): p = rr_search.mutate(p, rng, 0, 8, 3)
        rand.append(sc(p)["h23"])
    res["random_h23"] = dict(mean=float(np.mean(rand)), min=float(np.min(rand)), p10=float(np.percentile(rand, 10)))
    # search from each 1-edit neighbour (2000 evals) and from empty at growing budgets (fitness = full rollout, as in the pilot)
    z = np.zeros((0,) + F[0].shape, np.uint8); back = []
    for n, p in neigh:
        b, info = rr_search.search(p, F[0], z, F[1:], 1, 0, np.random.default_rng(0), evals=2000)
        back.append(dict(edit=n, start_h23=sc(p)["h23"], end_err=info["err"], exact=info["err"] == 0))
    res["search_from_neighbours"] = back; res["search_from_empty"] = []
    for ev in (2000, 8000, 32000):
        for s in range(3):
            t0 = time.process_time(); b, info = rr_search.search({"rules": []}, F[0], z, F[1:], 1, 0, np.random.default_rng(s), evals=ev)
            res["search_from_empty"].append(dict(evals=ev, seed=s, err=info["err"], cpu_s=round(time.process_time() - t0, 1), conditions=sum(len(r["if"]) for r in b["rules"])))
    json.dump(res, open(os.path.join(OUT, "life_landscape.json"), "w"), indent=1)
    print("exact", res["exact"], "hold", res["hold"], "random h23", res["random_h23"])
    for x in sorted(res["neighbours"], key=lambda x: x["h23"]): print(f"  {x['edit']:24s} h1 {x['h1']:.4f} h4 {x['h4']:.4f} h23 {x['h23']:.4f}")
    for x in res["paths"]: print("  path", x["order"], "h23", [round(v, 3) for v in x["h23"]], "h1", [round(v, 3) for v in x["h1"]])
    print("  back to exact from 1-edit neighbours:", sum(x["exact"] for x in back), "/", len(back))
    for x in res["search_from_empty"]: print("  empty start", x)


# ---------------- SAT ladder ----------------
class TooBig(Exception): pass


def sat_one(F, fam, k, S, timeout, cap=6_000_000):
    from pysat.solvers import Cadical195
    from ca.rule import family_taps, rollout as lut_roll
    from ca.sat import encode, luts_from_model
    taps = family_taps(k, "self"); cl = []
    def sink(c):
        cl.append(c)
        if len(cl) > cap: raise TooBig()
    t0 = time.process_time()
    try: n, L, nv = encode(F, taps, k + 1, S, sink=sink)
    except TooBig: return dict(status="SIZE", clauses=f">{cap}")
    if fam == "tot":   # tie table entries with the same (own v, #on neighbours, own hidden) for every output channel
        groups = {}
        for (c, e), var in L.items():
            key = (c, (e >> 4) & 1, bin(e & 0b111101111).count("1"), e >> 9); groups.setdefault(key, []).append(var)
        for vs in groups.values():
            for a in vs[1:]: cl.append([-vs[0], a]); cl.append([vs[0], -a])
    enc_s = round(time.process_time() - t0, 1); w0 = time.time()
    with Cadical195(bootstrap_with=cl) as s:
        tm = threading.Timer(timeout, s.interrupt); tm.start(); ok = s.solve_limited(expect_interrupt=True); tm.cancel()
        model = s.get_model() if ok else None
    out = dict(status="SAT" if ok else ("UNSAT" if ok is False else "TIMEOUT"), clauses=len(cl), encode_cpu_s=enc_s, solve_wall_s=round(time.time() - w0, 1))
    if ok:
        luts = luts_from_model(model, L, k + 1, len(taps)); vis, st, seen = lut_roll(F[0], taps, luts, len(F), S)
        out.update(exact=bool((vis == F).all()), entries_used=len(seen))
        if fam == "tot": out["table_entries"] = len({(c, (e >> 4) & 1, bin(e & 0b111101111).count("1"), e >> 9) for c in range(k + 1) for e in seen})
        out["_luts"] = luts; out["_taps"] = taps
    return out


def crop_at(full, a, n, size):
    """size x size crop (or the full frame) centred on the most-changing area of frames a..a+n."""
    if size is None: return full[a:a + n], (0, 0)
    ch = (full[a + 1:a + n] != full[a:a + n - 1]).sum(0).astype(float); H, W = ch.shape
    best, yx = -1, (0, 0)
    for y in range(0, H - size + 1, 2):
        for x in range(0, W - size + 1, 2):
            s = ch[y:y + size, x:x + size].sum()
            if s > best: best, yx = s, (y, x)
    y, x = yx; return full[a:a + n, y:y + size, x:x + size], yx


def sat_ladder(args):
    task, a, size, fam, k, S = args
    from exp45_c12 import target
    from ca.rule import rollout as lut_roll
    full = target("0:6572").astype(np.uint8); rows = []
    for T in (2, 3, 4, 6, 8, 12, 16):
        F, yx = crop_at(full, a, T, size)
        r = sat_one(F, fam, k, S, timeout=60); luts, taps = r.pop("_luts", None), r.pop("_taps", None)
        if r["status"] == "SAT":   # the found table, unchanged, run on to 16 frames of the same crop (reported, not used to select)
            G = full[a:a + 16, yx[0]:yx[0] + size, yx[1]:yx[1] + size] if size else full[a:a + 16]
            vis = lut_roll(G[0], taps, luts, 16, S)[0]; r["err_16_frames"] = float((vis[1:] != G[1:]).mean()); r["hold_16"] = float((G[1:] != G[0]).mean())
        rows.append(dict(task=task, crop=size or "48x36", at=list(yx), family=fam, k=k, S=S, T=T, **r)); print(json.dumps(rows[-1]), flush=True)
        if r["status"] != "SAT": break
    return rows


def sat():
    from multiprocessing import Pool
    jobs = [(t, a, size, fam, k, S) for t, a in (("translate", 1221), ("contour", 4500)) for size in (8, 16, None) for fam in ("tot", "full") for k in (0, 1, 2) for S in (1, 2)]
    with Pool(24) as p: res = p.map(sat_ladder, jobs)
    json.dump([r for rs in res for r in rs], open(os.path.join(OUT, "sat_ladder.json"), "w"), indent=1)


def lightcone():
    from exp45_c12 import target
    out = {}
    for task in ("1221:60", "4500:56", "1499:32", "1678:32"):
        F = target(task).astype(np.uint8); H, W = F[0].shape; row = {}
        for R in (1, 2, 3, 4):
            p = np.pad(F[0], R); pat = np.lib.stride_tricks.sliding_window_view(p, (2 * R + 1, 2 * R + 1)).reshape(H * W, -1)
            _, cls = np.unique(pat, axis=0, return_inverse=True); cls = cls.ravel(); v1 = F[1].ravel(); wit = None
            for c in np.unique(cls):
                idx = np.nonzero(cls == c)[0]
                if v1[idx].min() != v1[idx].max():
                    i0, i1 = idx[v1[idx] == 0][0], idx[v1[idx] == 1][0]; wit = dict(cell_off=[int(i0 // W), int(i0 % W)], cell_on=[int(i1 // W), int(i1 % W)],
                                                                                 patch="/".join("".join(map(str, r_)) for r_ in pat[i0].reshape(2 * R + 1, -1)), all_zero=bool(pat[i0].sum() == 0)); break
            row[R] = wit
        # empirical: random radius-2 dlrule programs (S=1, zero hidden) never separate the R=2 witness pair
        w = row[2]; viol = 0
        if w:
            rng = np.random.default_rng(0)
            for _ in range(300):
                k = int(rng.integers(0, 3)); p = {"rules": []}
                for _ in range(10): p = rr_search.mutate(p, rng, k, 8, 3)
                B = roll(p, F[0], 1, 1, k)[0]; viol += int(B[tuple(w["cell_off"])] != B[tuple(w["cell_on"])])
        row["random_r2_S1_programs_separating_witness"] = f"{viol}/300"; out[task] = row
        print(task, json.dumps(row))
    json.dump(out, open(os.path.join(OUT, "lightcone_witness.json"), "w"), indent=1)


def stab_one(args):
    task, n, obj, seed = args
    from exp45_c12 import target
    a = int(task); F = target(f"{a}:{n}").astype(np.uint8); S = 2; rng = np.random.default_rng(seed)
    def fit(p):
        B = roll(p, F[0], n - 1, S); full = float((B != F[1:]).mean()); nc = sum(len(r["if"]) for r in p["rules"])
        if obj == "full": return full + 1e-4 * nc, full
        sh = [float((roll(p, F[t], min(4, n - 1 - t), S) != F[t + 1:t + 5]).mean()) for t in range(0, n - 1, 4)]
        return 0.5 * full + 0.5 * float(np.mean(sh)) + 1e-4 * nc, full
    best = {"rules": []}; bf, _ = fit(best); cur, cf = best, bf; ev = 4000; t0 = time.process_time()
    for i in range(ev):
        if i % 1000 == 0: cur, cf = copy.deepcopy(best), bf
        c = rr_search.mutate(cur, rng, 0, 8, 3); f, _ = fit(c)
        if f <= cf:
            cur, cf = c, f
            if f < bf: best, bf = copy.deepcopy(c), f
    B = roll(best, F[0], n - 1, S); per_roll = (B != F[1:]).mean((1, 2)); per_one = np.array([(roll(best, F[t], 1, S)[0] != F[t + 1]).mean() for t in range(n - 1)])
    gap = per_roll - per_one; first = int(np.argmax(gap > 0.05)) + 1 if (gap > 0.05).any() else None
    G = target(f"{a + n}:{n}").astype(np.uint8); BG = roll(best, G[0], n - 1, S)
    ev_ = evaluate(np.concatenate([F[:1], B]), F)
    return dict(task=task, n=n, objective=obj, seed=seed, err=ev_["err"], hold=ev_["hold_err"], lag4_f1=ev_["lag4_f1"], mass=ev_["lag4_mass"], onestep_err=float(per_one.mean()),
                first_frame_gap_over_0_05=first, gap_by_quarter=[float(x.mean()) for x in np.array_split(gap, 4)], conditions=sum(len(r["if"]) for r in best["rules"]),
                next_clip_err=float((BG != G[1:]).mean()), next_clip_hold=float((G[1:] != G[0]).mean()), cpu_s=round(time.process_time() - t0, 1), program=best)


def stab():
    from multiprocessing import Pool
    jobs = [(t, n, o, s) for t in ("1221", "4500") for n in (16, 32, 64) for o in ("full", "mixed") for s in range(3)]
    with Pool(24) as p: res = p.map(stab_one, jobs)
    json.dump(res, open(os.path.join(OUT, "stability.json"), "w"), indent=1, default=float)
    for r in res: print(f"{r['task']}:{r['n']:2d} {r['objective']:5s} s{r['seed']} err {r['err']:.4f} (hold {r['hold']:.4f}) one-step {r['onestep_err']:.4f} F1 {r['lag4_f1']:.2f} gap>0.05 at {r['first_frame_gap_over_0_05']} "
                        f"gap/quarter {[round(x, 3) for x in r['gap_by_quarter']]} conds {r['conditions']} | next clip {r['next_clip_err']:.4f} (hold {r['next_clip_hold']:.4f}) cpu {r['cpu_s']}")


if __name__ == "__main__":
    dict(life=life, sat=sat, lightcone=lightcone, stab=stab)[sys.argv[1]]()
