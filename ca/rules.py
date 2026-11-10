"""Explicit, human-readable, executable CA rules ("ca-universal/dlrule-v1"; renamed from "rule-v1", which legacy C1-C4 LUT exports also use).

State per cell: v in {0,1} (visible) and k hidden bits h0..h{k-1} (k <= 3). Boundary: cells outside the grid are 0.
Features of a cell (all small integers, computed from the CURRENT state of the whole grid, synchronous update):
  v, h0..h{k-1}                      own state
  n8                                 visible neighbours in the 3x3 Moore neighbourhood (0..8, centre excluded)
  r2                                 visible cells on the radius-2 ring of the 5x5 square (0..16)
  N, S, E, W                         visible cells in the 3 neighbours on that side (0..3), e.g. N = up-left + up + up-right
  m0..m{k-1}                         neighbours with hidden bit i set (0..8)
Program: ordered list of rules {"if": [[feature, op, const], ...], "then": {"v": 0|1, "h": int}}; op in "<=", ">=", "==".
A rule fires when ALL its conditions hold (empty "if" = always). The FIRST firing rule sets the next state; if none
fires, the cell keeps its state (default). Every cell updates simultaneously; S substeps make one video frame.
Phases: a schedule [{start, end, program}] chooses the program for frames start..end-1 (external timeline); state
is carried across boundaries. Complexity is reported as rules, conditions, constants, state bits and bytes.
Learner: CART-style decision tree (Gini, integer thresholds) predicting the next full state (v, h) from the features;
each leaf becomes one rule whose conditions are the leaf's path (merged per feature into lo <= f <= hi)."""
import json, zlib
import numpy as np

OPS = {"<=": np.less_equal, ">=": np.greater_equal, "==": np.equal}


def feature_names(k):
    return ["v"] + [f"h{i}" for i in range(k)] + ["n8", "r2", "N", "S", "E", "W"] + [f"m{i}" for i in range(k)]


def features(v, h):
    """v: (H,W) {0,1}; h: (k,H,W) {0,1}. Returns dict name -> (H,W) int array."""
    H, W = v.shape; k = len(h); p = np.pad(v.astype(np.int16), 2); sh = lambda a, dy, dx: a[2 + dy:2 + dy + H, 2 + dx:2 + dx + W]
    f = {"v": v.astype(np.int16)}
    for i in range(k): f[f"h{i}"] = h[i].astype(np.int16)
    f["n8"] = sum(sh(p, dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)
    f["r2"] = sum(sh(p, dy, dx) for dy in range(-2, 3) for dx in range(-2, 3) if max(abs(dy), abs(dx)) == 2)
    f["N"] = sh(p, -1, -1) + sh(p, -1, 0) + sh(p, -1, 1); f["S"] = sh(p, 1, -1) + sh(p, 1, 0) + sh(p, 1, 1)
    f["W"] = sh(p, -1, -1) + sh(p, 0, -1) + sh(p, 1, -1); f["E"] = sh(p, -1, 1) + sh(p, 0, 1) + sh(p, 1, 1)
    for i in range(k):
        q = np.pad(h[i].astype(np.int16), 2); f[f"m{i}"] = sum(sh(q, dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)
    return f


def step(prog, v, h, trace=False):
    """One synchronous substep. Returns (v', h', fired) with fired = index of the firing rule per cell (-1 = default keep)."""
    f = features(v, h); fired = np.full(v.shape, -1, np.int32)
    for i, r in enumerate(prog["rules"]):
        m = fired < 0
        for name, op, c in r["if"]: m &= OPS[op](f[name], c)
        fired[m] = i
    v2, h2 = v.copy(), h.copy()
    for i, r in enumerate(prog["rules"]):
        m = fired == i
        if not m.any(): continue
        v2[m] = r["then"]["v"]
        for b in range(len(h)): h2[b][m] = (r["then"]["h"] >> b) & 1
    return (v2, h2, fired) if trace else (v2, h2, None)


def run(model, T=None):
    """model: {'k', 'S', 'programs': [...], 'schedule': [...], 'first': (H,W), 'h0': (k,H,W) or None}. Returns (T,H,W) binary
    visible frames and per-frame rule-fire counts per program."""
    k, S = model["k"], model["S"]; v = model["first"].astype(np.uint8).copy(); T = T or model["T"]
    h = model["h0"].astype(np.uint8).copy() if model.get("h0") is not None else np.zeros((k,) + v.shape, np.uint8)
    sched = np.zeros(T, np.int32)
    for p in model["schedule"]: sched[p["start"]:p["end"]] = p["program"]
    out = [v.copy()]; fires = [np.zeros(len(pr["rules"]) + 1, np.int64) for pr in model["programs"]]
    for t in range(1, T):
        prog = model["programs"][sched[t]]
        for _ in range(S):
            v, h, fired = step(prog, v, h, trace=True); fires[sched[t]] += np.bincount(fired.ravel() + 1, minlength=len(prog["rules"]) + 1)
        out.append(v.copy())
    return np.stack(out), [f.tolist() for f in fires]   # fires[p][0] = default (no rule fired), fires[p][i+1] = rule i


# ---------------- learner ----------------
def _gini(counts):
    n = counts.sum(-1, keepdims=True); p = counts / np.maximum(n, 1); return 1 - (p ** 2).sum(-1)


def fit_tree(X, y, names, max_leaves=64, min_leaf=20, w=None):
    """X: (n, d) small ints, y: (n,) class ids, w: optional sample weights (cost-sensitive learning). Best-first CART;
    returns list of leaves [(conds, class, n, purity)] (class = weighted majority, purity unweighted)."""
    ncls = int(y.max()) + 1; w = np.ones(len(y)) if w is None else np.asarray(w, float)
    def best_split(idx):
        best = None; cnt = np.bincount(y[idx], w[idx], minlength=ncls); base = _gini(cnt[None])[0] * cnt.sum()
        for j in range(X.shape[1]):
            xs = X[idx, j]; vals = np.unique(xs)
            if len(vals) < 2: continue
            tab = np.zeros((vals.max() + 1 - vals.min(), ncls)); np.add.at(tab, (xs - vals.min(), y[idx]), w[idx]); cum = np.cumsum(tab, 0); tot = cum[-1]
            cnt_n = np.zeros(len(tab)); np.add.at(cnt_n, xs - vals.min(), 1); cn = np.cumsum(cnt_n)
            for t in range(len(tab) - 1):
                L, R = cum[t], tot - cum[t]
                if cn[t] < min_leaf or cn[-1] - cn[t] < min_leaf: continue
                g = _gini(L[None])[0] * L.sum() + _gini(R[None])[0] * R.sum()
                if best is None or g < best[0]: best = (g, j, vals.min() + t)
        return None if best is None or best[0] >= base - 1e-9 else (base - best[0], best[1], best[2])
    leaves = [(np.arange(len(y)), [])]; heap = []
    def push(i):
        s = best_split(leaves[i][0]); heap.append((s[0], i, s) if s else (-1, i, None))
    push(0)
    while len(leaves) < max_leaves:
        cand = [c for c in heap if c[2] is not None]
        if not cand: break
        g, i, (gain, j, t) = max(cand, key=lambda c: c[0]); heap[:] = [c for c in heap if c[1] != i]
        idx, conds = leaves[i]; m = X[idx, j] <= t
        leaves[i] = (idx[m], conds + [(names[j], "<=", int(t))]); leaves.append((idx[~m], conds + [(names[j], ">=", int(t) + 1)]))
        push(i); push(len(leaves) - 1)
    out = []
    for idx, conds in leaves:
        cw = np.bincount(y[idx], w[idx], minlength=ncls); cnt = np.bincount(y[idx], minlength=ncls)
        out.append((conds, int(cw.argmax()), int(len(idx)), float(cnt[cw.argmax()] / max(len(idx), 1))))
    return out


def _merge(conds):
    lo, hi = {}, {}
    for f, op, c in conds:
        if op == "<=": hi[f] = min(hi.get(f, 99), c)
        else: lo[f] = max(lo.get(f, -99), c)
    for f in list(lo) + list(hi):           # own state bits are binary: v <= 0 -> v == 0, v >= 1 -> v == 1
        if f == "v" or (f[0] == "h" and f[1:].isdigit()):
            if hi.get(f) == 0: lo[f] = 0
            if lo.get(f) == 1: hi[f] = 1
    out = []
    for f in sorted(set(lo) | set(hi)):
        if f in lo and f in hi and lo[f] == hi[f]: out.append([f, "==", lo[f]])
        else:
            if f in lo: out.append([f, ">=", lo[f]])
            if f in hi: out.append([f, "<=", hi[f]])
    return out


def tree_program(leaves, k, drop_keep=True):
    """Leaves -> program. Leaves whose predicted next state equals the current state under ALL their own-state
    conditions are dropped when drop_keep (the default 'keep' covers them); rules are ordered by training count."""
    rules = []
    for conds, cls, n, pur in sorted(leaves, key=lambda l: -l[2]):
        c = _merge(conds); own = {f: val for f, op, val in c if op == "==" and (f == "v" or f.startswith("h"))}
        tv, th = cls & 1, cls >> 1
        if drop_keep and own.get("v") == tv and all(own.get(f"h{b}") == ((th >> b) & 1) for b in range(k)): continue
        rules.append({"if": c, "then": {"v": int(tv), "h": int(th)}, "train_n": n, "train_purity": round(pur, 4)})
    return {"rules": rules}


def dataset(V, Hb, k):
    """V: (n,H,W) visible, Hb: (n,k,H,W) hidden bits (states at consecutive substeps). Pairs (state_i -> state_{i+1})."""
    names = feature_names(k); Xs, ys = [], []
    for i in range(len(V) - 1):
        f = features(V[i], Hb[i]); Xs.append(np.stack([f[n].ravel() for n in names], 1))
        y = V[i + 1].astype(np.int64).ravel()
        for b in range(k): y = y | (Hb[i + 1][b].astype(np.int64).ravel() << (b + 1))
        ys.append(y)
    return np.concatenate(Xs), np.concatenate(ys), names


def complexity(model):
    progs = model["programs"]; nr = sum(len(p["rules"]) for p in progs); nc = sum(len(r["if"]) for p in progs for r in p["rules"])
    blob = json.dumps({"k": model["k"], "S": model["S"], "programs": [{"rules": [{"if": r["if"], "then": r["then"]} for r in p["rules"]]} for p in progs],
                       "schedule": model["schedule"]}, separators=(",", ":")).encode()
    first = np.packbits(model["first"].ravel()).tobytes(); h0 = b"" if model.get("h0") is None else np.packbits(model["h0"].ravel()).tobytes()
    return dict(programs=len(progs), rules=nr, conditions=nc, constants=nc + 2 * nr, hidden_bits=model["k"], rules_json_bytes=len(blob),
                file_bytes=len(zlib.compress(blob + first + h0, 9)), first_frame_bytes=len(first), h0_bytes=len(h0))


def export(path, model, meta=None):
    obj = {"format": "ca-universal/dlrule-v1", "k": model["k"], "S": model["S"], "T": int(model["T"]), "H": int(model["first"].shape[0]), "W": int(model["first"].shape[1]),
           "features": feature_names(model["k"]), "semantics": "first firing rule wins; default keep; synchronous; zero boundary; S substeps per frame; schedule = external timeline",
           "programs": [{"rules": [{"if": r["if"], "then": r["then"]} for r in p["rules"]]} for p in model["programs"]], "schedule": model["schedule"],
           "first": np.packbits(model["first"].astype(np.uint8).ravel(), bitorder="little").tobytes().hex(),
           "h0": None if model.get("h0") is None else np.packbits(model["h0"].astype(np.uint8).ravel(), bitorder="little").tobytes().hex(), "meta": meta or {}}
    json.dump(obj, open(path, "w"), separators=(",", ":")); return obj


def load(path):
    o = json.load(open(path)); H, W, k = o["H"], o["W"], o["k"]
    assert o["format"] == "ca-universal/dlrule-v1" or (o["format"] == "ca-universal/rule-v1" and "programs" in o), "not dlrule-v1"
    first = np.unpackbits(np.frombuffer(bytes.fromhex(o["first"]), np.uint8), bitorder="little")[:H * W].reshape(H, W)
    h0 = None if o["h0"] is None else np.unpackbits(np.frombuffer(bytes.fromhex(o["h0"]), np.uint8), bitorder="little")[:k * H * W].reshape(k, H, W)
    return dict(k=k, S=o["S"], T=o["T"], programs=o["programs"], schedule=o["schedule"], first=first, h0=h0, meta=o["meta"])


def to_text(prog, k):
    lines = []
    for i, r in enumerate(prog["rules"]):
        cond = " and ".join(f"{f} {op} {c}" for f, op, c in r["if"]) or "always"
        lines.append(f"R{i}: if {cond} -> v={r['then']['v']}" + (f", h={r['then']['h']:0{k}b}" if k else ""))
    return "\n".join(lines + ["default: keep state"])


if __name__ == "__main__":   # positive controls: (1) Conway's Life (k=0) recovered exactly; (2) a hidden-clock mover (k=1)
    rng = np.random.default_rng(0); H = W = 24
    def life_prog():
        return {"rules": [{"if": [["v", "==", 1], ["n8", "<=", 1]], "then": {"v": 0, "h": 0}}, {"if": [["v", "==", 1], ["n8", ">=", 4]], "then": {"v": 0, "h": 0}},
                          {"if": [["v", "==", 0], ["n8", "==", 3]], "then": {"v": 1, "h": 0}}]}
    f0 = (rng.random((H, W)) < .35).astype(np.uint8)
    truth = dict(k=0, S=1, T=20, programs=[life_prog()], schedule=[dict(start=1, end=20, program=0)], first=f0)
    Vt, fires = run(truth); X, y, names = dataset(Vt, np.zeros((20, 0, H, W), np.uint8), 0)
    learned = dict(truth, programs=[tree_program(fit_tree(X, y, names, max_leaves=32, min_leaf=1), 0)])
    Vl, _ = run(learned); assert (Vl == Vt).all(), "life not recovered"
    # hidden-bit control (k=1, full state observed): h toggles each substep; on h=1 a cell turns on iff n8 >= 3, on h=0 it keeps v
    tog = {"rules": [{"if": [["h0", "==", 0], ["v", "==", 1]], "then": {"v": 1, "h": 1}}, {"if": [["h0", "==", 0], ["v", "==", 0]], "then": {"v": 0, "h": 1}},
                     {"if": [["h0", "==", 1], ["n8", ">=", 3]], "then": {"v": 1, "h": 0}}, {"if": [["h0", "==", 1]], "then": {"v": 0, "h": 0}}]}
    t2 = dict(k=1, S=1, T=16, programs=[tog], schedule=[dict(start=1, end=16, program=0)], first=f0, h0=np.zeros((1, H, W), np.uint8))
    hs = [t2["h0"]]; vs = [f0]; v, h = f0.copy(), t2["h0"].copy()
    for _ in range(15): v, h, _ = step(tog, v, h); vs.append(v); hs.append(h)
    X, y, names = dataset(np.stack(vs), np.stack(hs), 1); l2 = dict(t2, programs=[tree_program(fit_tree(X, y, names, max_leaves=32, min_leaf=1), 1)])
    assert (run(l2)[0] == run(t2)[0]).all(), "hidden-bit control not recovered"
    import tempfile, os
    p = os.path.join(tempfile.mkdtemp(), "life.rule.json"); export(p, learned); L = load(p); Vr, _ = run(dict(L, first=L["first"])); assert (Vr == Vt).all()
    print("rules self-check ok: Life recovered exactly;", complexity(learned)); print(to_text(learned["programs"][0], 0))
