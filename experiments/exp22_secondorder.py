"""Exp22 (Phase 07, candidate C3): second-order CA with a decision-tree rule, learned by supervised
induction (CART on pattern histograms) with optional DAgger-style re-labelling on its own rollouts.

Model (autonomous, F_shared k=1 = 18 taps): visible 3x3 now + hidden 3x3; hidden' = visible centre
(so hidden holds the previous frame; hidden init 0 = an all-black "previous frame"), visible' = tree(18 bits).
S = 1 generation per frame, boundary 0. Because hidden is a KNOWN function of past frames, every training
example is observed: (frame t, frame t-1) -> frame t+1. DAgger rounds add examples (rollout state at t,
rollout state at t-1) -> TRUE frame t+1, then refit.
Rule complexity: internal nodes (conditions), leaves; rule-native size = preorder bits; the official
size is the container file of the tree compiled to a 2-input-gate circuit (mux per node, constants folded).
usage: exp22_secondorder.py TARGET [--depths 1-12] [--dagger 3]
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c7_common import C7, baselines_bytes, evaluate, export_verify, media, target
from ca.rule import family_taps, index

TAPS = family_taps(1, "moore"); N = 18; CENTRE = 4
FEAT = [f"{'now' if c == 0 else 'prev'}{'NW N NE W C E SW S SE'.split()[(dy + 1) * 3 + dx + 1]}" for c, dy, dx in TAPS]


def patterns(vis, prev):
    s = np.stack([vis, prev]).astype(np.uint8)
    return index(s, TAPS, 0).ravel()


def hist(pairs):
    """pairs: list of (pattern ints, labels) -> counts (2^18, 2)."""
    h = np.zeros((2 ** N, 2), np.int64)
    for p, y in pairs:
        np.add.at(h, (p, y.ravel().astype(np.int64)), 1)
    return h


def grow(h, depth):
    """CART (Gini) on the pattern histogram. Node = (feature, left(=0), right(=1)) or leaf int."""
    pats = np.nonzero(h.sum(1))[0]

    def build(ps, d):
        c = h[ps].sum(0); leaf = int(c[1] > c[0])
        if d == 0 or c.min() == 0:
            return leaf
        best, bf = None, None
        tot = c.sum(); g0 = 1 - ((c / tot) ** 2).sum()
        for f in range(N):
            m = ((ps >> f) & 1).astype(bool)
            c1 = h[ps[m]].sum(0); c0 = c - c1
            if c1.sum() == 0 or c0.sum() == 0: continue
            g = sum(cc.sum() / tot * (1 - ((cc / cc.sum()) ** 2).sum()) for cc in (c0, c1))
            if best is None or g < best: best, bf = g, f
        if bf is None or best >= g0 - 1e-12:
            return leaf
        m = ((ps >> bf) & 1).astype(bool)
        l, r = build(ps[~m], d - 1), build(ps[m], d - 1)
        return leaf if (l == r and isinstance(l, int)) else (bf, l, r)
    return build(pats, depth)


def predict(tree, pats):
    out = np.empty(len(pats), np.uint8)
    def go(t, idx):
        if isinstance(t, int): out[idx] = t; return
        f, l, r = t; m = ((pats[idx] >> f) & 1).astype(bool)
        go(l, idx[~m]); go(r, idx[m])
    go(tree, np.arange(len(pats))); return out


def size(t):
    return (0, 1) if isinstance(t, int) else tuple(a + b for a, b in zip((1, 0), tuple(x + y for x, y in zip(size(t[1]), size(t[2])))))


def text(t, ind=""):
    if isinstance(t, int): return f"{ind}-> {t}\n"
    f, l, r = t
    return f"{ind}if {FEAT[f]}:\n{text(r, ind + '  ')}{ind}else:\n{text(l, ind + '  ')}"


def rollout(tree, F, T):
    vis, prev = F[0].copy(), np.zeros_like(F[0]); out = [vis]; states = []
    for _ in range(1, T):
        p = patterns(vis, prev); states.append((p, vis, prev))
        vis, prev = predict(tree, p).reshape(vis.shape), vis
        out.append(vis)
    return np.stack(out), states


def to_circuit(tree):
    """mux per internal node with constant folding; pool = 18 taps + gates. Returns gates, outputs."""
    gates = []
    def g(op, a, b):
        if a > b: a, b, op = b, a, ((op & 1) | ((op >> 2) & 1) << 1 | ((op >> 1) & 1) << 2 | (op & 8))  # swap inputs: permute tt bits 1<->2
        gates.append((op, a, b)); return N + len(gates) - 1
    def sig(t):  # -> ('c', bit) or ('s', pool index)
        if isinstance(t, int): return ("c", t)
        f, l, r = t; L, R = sig(l), sig(r)
        if L == R: return L
        if L[0] == "c" and R[0] == "c": return ("s", f) if R[1] == 1 else ("s", g(3, f, 0 if f else 1))  # x / not x
        x = f   # tt codes, a = x first input: 8 a&b, 2 ~a&b, 14 a|b, 11 ~a|b
        if L[0] == "c":   # x ? R : L-const
            return ("s", g(8, x, R[1])) if L[1] == 0 else ("s", g(11, x, R[1]))
        if R[0] == "c":   # x ? R-const : L
            return ("s", g(14, x, L[1])) if R[1] == 1 else ("s", g(2, x, L[1]))
        a = g(8, x, R[1]); b = g(2, x, L[1]); return ("s", g(14, a, b))
    s = sig(tree)
    if s[0] == "c":
        s = ("s", g(15 if s[1] else 0, 0, 1))
    return gates, [s[1], CENTRE]


if __name__ == "__main__":
    from ca import costs
    tname = sys.argv[1]; depths = range(1, 13); rounds = 3
    if "--depths" in sys.argv: a, b = sys.argv[sys.argv.index("--depths") + 1].split("-"); depths = range(int(a), int(b) + 1)
    if "--dagger" in sys.argv: rounds = int(sys.argv[sys.argv.index("--dagger") + 1])
    F = target(tname); T, H, W = F.shape
    OUT = os.path.join(C7, "c3_secondorder"); os.makedirs(OUT, exist_ok=True); stem0 = os.path.join(OUT, tname.replace(":", "_"))
    base = [(patterns(F[t], F[t - 1] if t else np.zeros_like(F[0])), F[t + 1]) for t in range(T - 1)]
    rows = []; t0 = time.process_time(); best = None
    for d in depths:
        pairs = list(base)
        for rd in range(rounds + 1):
            tree = grow(hist(pairs), d)
            P, states = rollout(tree, F, T)
            ev = evaluate(P, F); ni, nl = size(tree)
            rows.append(dict(depth=d, dagger_round=rd, internal_nodes=ni, leaves=nl, mean_err=ev["mean_err"], ratio=ev["mean_ratio_to_hold"],
                             change_iou=ev["change_iou"], horizon=ev["horizon"], exact=ev["exact"], tol_space=ev["mean_tol_space"]))
            print(json.dumps(rows[-1]), flush=True)
            key = (ev["mean_err"], ni)
            if best is None or key < best[0]: best = (key, d, rd, tree, P, ev)
            pairs += [(p, F[t + 1]) for t, (p, _, _) in enumerate(states)]   # DAgger: own states -> true next frame
    search_cpu = time.process_time() - t0
    # Pareto front (size vs error) over all fitted trees; the best-error tree is exported
    _, d, rd, tree, P, ev = best
    gates, outs = to_circuit(tree)
    L = costs.eval_circuit(gates, outs, N)
    from ca.rule import rollout as lut_rollout
    v, _, _ = lut_rollout(F[0], TAPS, L, T, 1); assert (v == P).all(), "circuit compile disagrees with tree rollout"
    ex = export_verify(P, F, dict(gates=gates, outputs=outs, n_in=N, family="F_shared", k=1, S=1), "ca_circuit", stem0 + ".best.cab")
    ni, nl = size(tree)
    res = dict(method="C3 second-order CA + decision tree (CART, DAgger)", target=tname, shape=[T, H, W], S=1, structure="F_shared k=1, hidden'=visible centre",
               search_cpu_s=round(search_cpu, 1), best_depth=d, best_dagger_round=rd, internal_nodes=ni, leaves=nl,
               rule_native_bits=ni * (1 + 5) + nl * 2, circuit_gates=len(gates), rule_text=text(tree), eval=ev, export=ex,
               baselines_bytes=baselines_bytes(F), all_fits=rows, media=media(P, F, stem0 + ".best"))
    json.dump(res, open(stem0 + ".result.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("eval", "all_fits", "rule_text")} | {"mean_err": ev["mean_err"], "hold": ev["mean_hold"], "change_iou": ev["change_iou"]}))
