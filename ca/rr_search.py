"""Readable Rules: rollout-driven local search over dlrule programs (shared by the 'direct' and 'extend simple rules' paths).

A program is a dlrule list of rules {"if": [[feature, op, const], ...], "then": {"v": 0|1, "h": int}} (first match wins,
default keep). The search scores a program by AUTONOMOUS replay: starting from the carried state (visible + k hidden bits),
it runs the program S substeps per frame over the phase's frames and compares the visible frames with the original:
    fitness = pixel error + alpha * conditions + gamma * max(0, 0.5 - lag-4 change F1)   (gamma = 0 disables the motion term)
Moves (one per proposal): change a constant / operator / feature of a condition, add / remove a condition, add / delete a
rule, swap two adjacent rules (priority), change a rule's output. Accept if fitness does not get worse (plateau moves allowed),
with random restarts from the best. Caps on rules and conditions keep programs readable.
Phases are fitted one after another with the state carried; phase p starts from phase p-1's program, and its stored form is
the edit list against it (dlrule-v2 'base' + 'edits'), so sharing and per-phase change are measured, not assumed."""
import copy, time
import numpy as np
from ca import rules
from ca.motion9 import lag_terms

RANGES = {"v": 1, "u": 1, "d": 1, "l": 1, "r": 1, "N": 3, "S": 3, "E": 3, "W": 3, "n8": 8, "r2": 16}


def frange(name): return RANGES.get(name, 8 if name.startswith("m") else 1)


def rollout(prog, v, h, n, S):
    """Run prog for n frames from state (v, h); returns visible frames (n,H,W) and final state."""
    out = []
    for _ in range(n):
        for _ in range(S): v, h, _ = rules.step(prog, v, h)
        out.append(v)
    return np.stack(out), v, h


def score(prog, v0, h0, F, S, alpha, gamma):
    B, v, h = rollout(prog, v0, h0, len(F), S); err = float((B != F).mean())
    nc = sum(len(r["if"]) for r in prog["rules"]); f = err + alpha * nc
    m = None
    if gamma:
        prev = np.concatenate([v0[None], B]); orig = None
        lt = lag_terms(prev, np.concatenate([v0[None], F]), 4) if len(F) >= 5 else None
        m = lt["f1_tol"] if lt else 1.0; f += gamma * max(0.0, 0.5 - m)
    return f, err, m, (v, h)


def mutate(prog, rng, k, max_rules, max_conds):
    p = copy.deepcopy(prog); rs = p["rules"]; names = rules.feature_names(k); op = rng.integers(9)
    if (op in (0, 1, 2, 4) and not any(r["if"] for r in rs)) or (op in (5, 6, 7, 8) and not rs): op = 3
    if op in (0, 1, 2, 4):
        r = rs[rng.choice([i for i, x in enumerate(rs) if x["if"]])]; j = rng.integers(len(r["if"])); c = r["if"][j]
        if op == 0: c[2] = int(np.clip(c[2] + rng.choice([-1, 1]), 0, frange(c[0])))
        elif op == 1: c[1] = str(rng.choice(["<=", ">=", "=="]))
        elif op == 2: c[0] = str(rng.choice(names)); c[2] = int(rng.integers(0, frange(c[0]) + 1))
        else: del r["if"][j]
    elif op == 3:   # add condition (or a rule if none)
        if not rs or len(rs) < max_rules and rng.random() < 0.3:
            n = str(rng.choice(names)); rs.insert(int(rng.integers(len(rs) + 1)), {"if": [[n, str(rng.choice(["<=", ">=", "=="])), int(rng.integers(0, frange(n) + 1))]],
                                                                               "then": {"v": int(rng.integers(2)), "h": int(rng.integers(2 ** k))}})
        else:
            r = rs[rng.integers(len(rs))]
            if len(r["if"]) < max_conds: n = str(rng.choice(names)); r["if"].append([n, str(rng.choice(["<=", ">=", "=="])), int(rng.integers(0, frange(n) + 1))])
    elif op == 5: del rs[rng.integers(len(rs))]
    elif op == 6 and len(rs) > 1: i = int(rng.integers(len(rs) - 1)); rs[i], rs[i + 1] = rs[i + 1], rs[i]
    elif op == 7: r = rs[rng.integers(len(rs))]; r["then"] = {"v": int(rng.integers(2)), "h": int(rng.integers(2 ** k))}
    elif op == 8 and len(rs) < max_rules: rs.insert(int(rng.integers(len(rs) + 1)), copy.deepcopy(rs[rng.integers(len(rs))]))
    return p


def search(start, v0, h0, F, S, k, rng, evals=2000, alpha=1e-4, gamma=0.0, max_rules=8, max_conds=3, restarts=4):
    best = copy.deepcopy(start); bf, be, bm, _ = score(best, v0, h0, F, S, alpha, gamma); n = 1; hist = [(0, be)]
    cur, cf = copy.deepcopy(best), bf; per = max(1, evals // restarts)
    while n < evals:
        if n % per == 0: cur, cf = copy.deepcopy(best), bf
        cand = mutate(cur, rng, k, max_rules, max_conds); f, e, m, _ = score(cand, v0, h0, F, S, alpha, gamma); n += 1
        if f <= cf:
            cur, cf = cand, f
            if f < bf: best, bf, be, bm = copy.deepcopy(cand), f, e, m; hist.append((n, e))
    return best, dict(fitness=bf, err=be, motion=bm, evals=n, improvements=hist)


def diff(base, prog):
    """Edit list turning base into prog: positional 'set' for changed rules, then inserts / deletes at the end."""
    eds = []; a, b = base["rules"], prog["rules"]
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]: eds.append({"op": "set", "index": i, "rule": copy.deepcopy(b[i])})
    for i in range(len(a), len(b)): eds.append({"op": "insert", "index": i, "rule": copy.deepcopy(b[i])})
    for i in range(len(a) - 1, len(b) - 1, -1): eds.append({"op": "delete", "index": i})
    return eds


LIBRARY = {   # classic local rules used as starting points ('extend simple rules' path); names are descriptive only
    "hold": {"rules": []},
    "life_b3s23": {"rules": [{"if": [["v", "==", 1], ["n8", "<=", 1]], "then": {"v": 0, "h": 0}}, {"if": [["v", "==", 1], ["n8", ">=", 4]], "then": {"v": 0, "h": 0}},
                             {"if": [["v", "==", 0], ["n8", "==", 3]], "then": {"v": 1, "h": 0}}]},
    "majority": {"rules": [{"if": [["n8", ">=", 5]], "then": {"v": 1, "h": 0}}, {"if": [["n8", "<=", 3]], "then": {"v": 0, "h": 0}}]},
    "grow": {"rules": [{"if": [["v", "==", 0], ["n8", ">=", 3]], "then": {"v": 1, "h": 0}}]},
    "erode": {"rules": [{"if": [["v", "==", 1], ["n8", "<=", 5]], "then": {"v": 0, "h": 0}}]},
    **{f"shift_from_{d}": {"rules": [{"if": [[d, "==", 1]], "then": {"v": 1, "h": 0}}, {"if": [[d, "==", 0]], "then": {"v": 0, "h": 0}}]} for d in ("u", "d", "l", "r")},
}


def fit_phases(F, phases, k, S, rng, start_mode="library", evals=2000, gamma=0.0, max_rules=8, max_conds=3, tree_leaves=32, log=print):
    """Greedy over phases with carried state. F: target frames (T,H,W), frame 0 given. Returns model dict (dlrule) + records."""
    T, H, W = F.shape; v = F[0].astype(np.uint8); h = np.zeros((k, H, W), np.uint8); progs, recs, prev = [], [], None
    for pi, p in enumerate(phases):
        a, b = p["start"], p["end"]; Fp = F[a:b].astype(np.uint8); t0 = time.time()
        if prev is not None: cands = {"previous": prev}
        elif start_mode == "library": cands = dict(LIBRARY)
        elif start_mode == "hold": cands = {"hold": LIBRARY["hold"]}   # discovery test: no classic rule given
        else:   # 'tree': teacher-forced decision list fitted on the phase's true frame pairs (k=0), then rollout search
            X, y, names = rules.dataset(F[a - 1:b].astype(np.uint8), np.zeros((b - a + 1, 0, H, W), np.uint8), 0)
            cands = {"tree": rules.tree_program(rules.fit_tree(X, y, names, max_leaves=tree_leaves, min_leaf=5), 0)}
            for r in cands["tree"]["rules"]: r.pop("train_n", None); r.pop("train_purity", None)
        scored = {n: score(q, v, h, Fp, S, 1e-4, gamma)[0] for n, q in cands.items()}; sn = min(scored, key=scored.get)
        best, info = search(cands[sn], v, h, Fp, S, k, rng, evals, gamma=gamma, max_rules=max_rules, max_conds=max_conds)
        _, err, mot, (v, h) = score(best, v, h, Fp, S, 1e-4, gamma)   # carry the state with the chosen program
        progs.append(best if prev is None else {"base": len(progs) - 1, "edits": diff(prev, best)}); prev = best
        recs.append(dict(phase=pi, start=a, end=b, start_from=sn, start_scores=scored, err=err, motion=mot, evals=info["evals"], seconds=round(time.time() - t0, 1),
                         rules=len(best["rules"]), conditions=sum(len(r["if"]) for r in best["rules"])))
        log(f"phase {pi} [{a},{b}) from {sn}: err {err:.4f} rules {len(best['rules'])} conds {recs[-1]['conditions']} ({recs[-1]['seconds']} s)")
    model = dict(k=k, S=S, T=T, programs=progs, schedule=[dict(start=p["start"], end=p["end"], program=i) for i, p in enumerate(phases)], first=F[0].astype(np.uint8),
                 h0=np.zeros((k, H, W), np.uint8) if k else None)
    return model, recs
