"""Readable Rules mid-course diagnostics (CPU only).
  blind    : blind rule recovery on the positive controls. The generator rule is NOT given to any method. Train on the
             seed-0 run; test the recovered rule from two held-out initial states (seeds 1, 2), autonomously.
             Methods: search  = rollout search started from the empty program ('hold'), no library;
                      tree    = decision list fitted to observed one-step visible transitions (k = 0 only);
                      sat_k   = exact SAT synthesis of a LUT over 3x3 visible + k own hidden bits (pysat CaDiCaL), then
                                the LUT's own hidden trajectory is turned into a decision list (dlrule) and re-verified.
  collide  : Bad Apple tasks. (a) minimum one-step error of ANY deterministic rule whose inputs are the 3x3 (or 5x5) visible
             patch plus the cell's own last j visible values (j = 0..3, i.e. k = j hidden bits used as a shift register);
             this is exact for that representation (sum of minority counts per identical input). (b) light-cone proof
             (ca.sat.light_cone_conflicts): first frame at which NO radius-1 rule with ANY hidden state and constant hidden
             init can be exact, for S = 1..4. (c) heuristic: share of newly-on pixels farther than r*S from every on pixel
             of the previous frame (Chebyshev), r = 1 and 2.
  nca      : the pinned whole-song Neural CA's error on each task window (its own replay from frame 0).
usage: rr_diag.py blind|collide|nca"""
import json, os, sys, time
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca import rules, rr_search
from ca.rule import family_taps, rollout as lut_rollout
from rr_run import control, evaluate

OUT = os.path.join(ROOT, "runs", "readable_rules", "diag"); os.makedirs(OUT, exist_ok=True)
TASKS = {"translate": "1221:60", "contour": "4500:56", "wipe": "1499:32", "cut": "CUT"}   # CUT resolved in find_cut()


def run_prog(prog, f0, k, T, S=1):
    m = dict(k=k, S=S, T=T, programs=[prog], schedule=[dict(start=1, end=T, program=0)], first=f0, h0=np.zeros((k,) + f0.shape, np.uint8) if k else None)
    return rules.run(m)[0]


def ncond(prog): return sum(len(r["if"]) for r in prog["rules"])


def sat_recover(F, k, budget_s=600):
    from pysat.solvers import Cadical195
    from ca.sat import encode, luts_from_model
    taps = family_taps(k, "self"); t0 = time.process_time(); clauses, L, nv = encode(F, taps, k + 1, 1)
    with Cadical195(bootstrap_with=clauses) as s:
        ok = s.solve(); luts = luts_from_model(s.get_model(), L, k + 1, len(taps)) if ok else None
    if not ok: return dict(status="UNSAT", cpu_s=round(time.process_time() - t0, 1)), None, None
    vis, st, seen = lut_rollout(F[0], taps, luts, len(F))
    X, y, names = rules.dataset(st[:, 0], st[:, 1:], k)   # the LUT's own hidden trajectory -> readable decision list
    lv = rules.fit_tree(X, y, names, max_leaves=64, min_leaf=1); prog = rules.tree_program(lv, k)
    for r in prog["rules"]: r.pop("train_n", None); r.pop("train_purity", None)
    return dict(status="SAT", lut_exact=bool((vis == F).all()), lut_entries_used=len(seen), lut_entries_total=2 ** len(taps),
                cpu_s=round(time.process_time() - t0, 1)), prog, (taps, luts)


def blind():
    res = []
    for name, k in (("shift", 0), ("life", 0), ("blink", 1), ("excitable", 1)):
        F, gm = control(name, seed=0); held = [control(name, seed=s)[0] for s in (1, 2)]; T = len(F)
        truth = gm["programs"][0]; row = dict(control=name, generator_k=k, generator_conditions=ncond(truth), generator_rules=len(truth["rules"]), methods={})
        def score_prog(prog, kk, extra):
            tr = float((run_prog(prog, F[0], kk, T) != F).mean()); ho = [float((run_prog(prog, h[0], kk, T) != h).mean()) for h in held]
            return dict(extra, train_err=tr, holdout_err=ho, exact_train=tr == 0, exact_holdout=all(e == 0 for e in ho), rules=len(prog["rules"]), conditions=ncond(prog), program=prog)
        for kk in sorted({0, k}):   # search
            t0 = time.process_time(); prog, info = rr_search.search({"rules": []}, F[0], np.zeros((kk,) + F[0].shape, np.uint8), F[1:], 1, kk, np.random.default_rng(0), evals=8000, max_rules=8, max_conds=3)
            row["methods"][f"search_k{kk}"] = score_prog(prog, kk, dict(evals=info["evals"], cpu_s=round(time.process_time() - t0, 1)))
        t0 = time.process_time(); X, y, names = rules.dataset(F, np.zeros((T, 0) + F[0].shape, np.uint8), 0)
        prog = rules.tree_program(rules.fit_tree(X, y, names, max_leaves=64, min_leaf=1), 0)
        for r in prog["rules"]: r.pop("train_n", None); r.pop("train_purity", None)
        row["methods"]["tree_k0"] = score_prog(prog, 0, dict(evals=1, cpu_s=round(time.process_time() - t0, 1)))
        for kk in (0, 1):
            info, prog, lut = sat_recover(F, kk)
            if prog is None: row["methods"][f"sat_k{kk}"] = info; continue
            taps, luts = lut; info["lut_holdout_err"] = [float((lut_rollout(h[0], taps, luts, T)[0] != h).mean()) for h in held]
            row["methods"][f"sat_k{kk}"] = score_prog(prog, kk, info)
        res.append(row)
        for m, r in row["methods"].items():
            print(f"{name:9s} {m:9s} " + (f"train {r['train_err']:.4f} holdout {r['holdout_err'][0]:.4f}/{r['holdout_err'][1]:.4f} exact {r['exact_train']}/{r['exact_holdout']} rules {r['rules']} conds {r['conditions']} cpu {r['cpu_s']} s"
                                          + (f" lut_holdout {r['lut_holdout_err']}" if 'lut_holdout_err' in r else "") if "train_err" in r else str(r)), flush=True)
    json.dump(res, open(os.path.join(OUT, "blind.json"), "w"), indent=1, default=float)


def find_cut(F):
    d = (F[1:] != F[:-1]).mean((1, 2)); t = int(np.argmax(np.where(np.arange(len(d)) > 100, d, 0))) + 1   # largest one-frame change
    return t, float(d[t - 1])


def min_err(keys, nxt):
    _, cls = np.unique(keys, axis=0, return_inverse=True); cls = cls.ravel(); on = np.bincount(cls, nxt); tot = np.bincount(cls)
    return float(np.minimum(on, tot - on).sum() / len(nxt)), int(len(tot))


def collide():
    from exp45_c12 import target
    from ca.sat import light_cone_conflicts
    full = target("0:6572").astype(np.uint8); t, mag = find_cut(full); TASKS["cut"] = f"{t - 8}:32"
    out = dict(cut_frame=t, cut_change=mag, tasks={})
    for tn, spec in TASKS.items():
        a, n = map(int, spec.split(":")); F = full[a - 3:a + n]   # 3 frames of history before the window for own-history keys
        T, H, W = F.shape; row = dict(spec=spec, hold_err=float((full[a + 1:a + n] != full[a:a + 1]).mean()), min_onestep_err={})
        for r in (1, 2):
            p = np.pad(F, ((0, 0), (r, r), (r, r))); patch = np.lib.stride_tricks.sliding_window_view(p, (2 * r + 1, 2 * r + 1), axis=(1, 2)).reshape(T, H * W, -1)
            for j in range(4):
                keys = np.concatenate([np.concatenate([patch[tt]] + [F[tt - i].reshape(-1, 1) for i in range(1, j + 1)], 1) for tt in range(3, T - 1)])
                nxt = np.concatenate([F[tt + 1].ravel() for tt in range(3, T - 1)]).astype(np.int64)
                e, ncls = min_err(keys, nxt); row["min_onestep_err"][f"r{r}_k{j}"] = dict(err=e, distinct_inputs=ncls, samples=len(nxt))
        W_ = full[a:a + n]; row["light_cone_first_conflict"] = {}
        for S in (1, 2, 3, 4):
            lc = light_cone_conflicts(W_, S); row["light_cone_first_conflict"][S] = lc[0] if lc else None
        far = {}
        for r in (1, 2):
            for S in (1, 2, 4):
                R = r * S; tot = cnt = 0
                for tt in range(a + 1, a + n):
                    prev, new = full[tt - 1], (full[tt] == 1) & (full[tt - 1] == 0)
                    if not new.any(): continue
                    pp = np.pad(prev, R); near = np.lib.stride_tricks.sliding_window_view(pp, (2 * R + 1, 2 * R + 1)).max((2, 3)).astype(bool)
                    tot += int(new.sum()); cnt += int((new & ~near).sum())
                far[f"r{r}_S{S}"] = cnt / max(tot, 1)
        row["newly_on_beyond_reach"] = far; out["tasks"][tn] = row
        print(tn, spec, "hold", round(row["hold_err"], 4), {k: round(v["err"], 4) for k, v in row["min_onestep_err"].items()}, "lightcone", row["light_cone_first_conflict"],
              {k: round(v, 3) for k, v in far.items()}, flush=True)
    json.dump(out, open(os.path.join(OUT, "collide.json"), "w"), indent=1, default=float)


def nca():
    from exp45_c12 import target
    from rr_extract import TEACHER, TEACHER_SHA
    from ca import phase_nca
    import hashlib
    assert hashlib.sha256(open(TEACHER, "rb").read()).hexdigest() == TEACHER_SHA
    tasks = json.load(open(os.path.join(OUT, "collide.json")))["tasks"]; end = max(sum(map(int, t["spec"].split(":"))) for t in tasks.values())
    hd, E, sched, h0, f0 = phase_nca.load(TEACHER); B, _ = phase_nca.rollout_np(E, f0, sched[:end], hd["S"], h0); out = {}
    for tn, t in tasks.items():
        a, n = map(int, t["spec"].split(":")); F = target(t["spec"]); Bt = B[a:a + n].copy(); Bt[0] = F[0]
        out[tn] = dict(spec=t["spec"], err=float((Bt[1:] != F[1:]).mean()), eval=evaluate(Bt, F)); print(tn, t["spec"], "NCA err", round(out[tn]["err"], 4), flush=True)
    json.dump(out, open(os.path.join(OUT, "nca_windows.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    dict(blind=blind, collide=collide, nca=nca)[sys.argv[1]]()
