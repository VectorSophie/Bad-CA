"""Readable Rules, path 'extract from the Neural CA': learn a dlrule program from the PINNED teacher's local transitions,
with the state abstraction as the object of study. The teacher (runs/featured/whole_song_s0.ncab, sha256 facb43d9...) is
replayed by numpy from frame 0 (its own autonomous replay); its full 16-channel state is recorded at every substep of the task
window [a, a+n). An abstraction maps each cell's continuous state to (v, c): v = s0 > 0.5, c = a small code stored in k hidden bits:
  pca  : signs of the top-k principal components of the 15 hidden channels (fitted on the window)
  kmeans: k-means of the full 16-channel state with 2^k clusters, fitted separately for v = 1 and v = 0 cells
Transitions (abstract state + neighbourhood features at substep i -> abstract state at i+1) are fitted by the decision-list
learner; the collision rate (identical feature vectors with different next states) is reported. The extracted program is
exported with the abstract initial state h0 (stored, counted) and replayed ON ITS OWN (no teacher, no hidden re-injection),
then scored against the original and against the teacher.
usage: rr_extract.py START:LEN OUTDIR --abs pca|kmeans --k 2 [--leaves 64] [--balance 0]"""
import hashlib, json, os, sys, time
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca import nca, phase_nca, rules
from exp45_c12 import target
from rr_run import evaluate

TEACHER = os.path.join(ROOT, "runs", "featured", "whole_song_s0.ncab"); TEACHER_SHA = "facb43d9911399eeffa0e04df010a82f713d54b2212d5d2e2e4cf7c72358e4f5"


def teacher_states(a, n):
    assert hashlib.sha256(open(TEACHER, "rb").read()).hexdigest() == TEACHER_SHA, "teacher changed"
    hd, E, sched, h0, f0 = phase_nca.load(TEACHER); S = hd["S"]; s = np.zeros((hd["C"],) + f0.shape, np.float32); s[0] = f0; s[1:] = h0
    states = []
    for t in range(1, a + n):
        p = {k: E[k][sched[t]] for k in phase_nca.PNAMES}
        if t > a: states_frame = []
        for _ in range(S):
            s = nca.step_np(p, s)
            if t > a: states.append(s.copy())
        if t == a: states.append(s.copy())        # state at frame a (start of the window)
    return np.stack(states), S                     # (1 + (n-1)*S, C, H, W)


def kmeans(X, K, rng, it=25):
    C = X[rng.choice(len(X), K, replace=False)]
    for _ in range(it):
        lab = ((X[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
        for j in range(K):
            if (lab == j).any(): C[j] = X[lab == j].mean(0)
    return C


def abstract(St, kind, k, rng):
    """St (N, C, H, W) -> v (N,H,W) uint8, codes (N,k,H,W) uint8, description."""
    N, C, H, W = St.shape; v = (St[:, 0] > 0.5).astype(np.uint8); X = St.transpose(0, 2, 3, 1).reshape(-1, C)
    if kind == "pca":
        Hd = X[:, 1:] - X[:, 1:].mean(0); _, _, Vt = np.linalg.svd(Hd[rng.choice(len(Hd), min(50000, len(Hd)), replace=False)], full_matrices=False)
        code = (Hd @ Vt[:k].T > 0).astype(np.uint8)
    else:
        code = np.zeros((len(X), k), np.uint8); vv = v.reshape(-1)
        for val in (0, 1):
            idx = np.nonzero(vv == val)[0]; sub = X[idx]; mu, sd = sub.mean(0), sub.std(0) + 1e-6
            Cn = kmeans(((sub - mu) / sd)[rng.choice(len(sub), min(20000, len(sub)), replace=False)], 2 ** k, rng)
            lab = np.concatenate([((((sub[i:i + 100000] - mu) / sd)[:, None, :] - Cn[None]) ** 2).sum(-1).argmin(1) for i in range(0, len(sub), 100000)])
            for b in range(k): code[idx, b] = (lab >> b) & 1
    return v, code.reshape(N, H, W, k).transpose(0, 3, 1, 2)


def collisions(X, y):
    keys = {}; conflict = 0
    rows = [tuple(r) for r in X]
    for r, t in zip(rows, y): keys.setdefault(r, set()).add(int(t))
    amb = {r for r, s in keys.items() if len(s) > 1}
    return float(np.mean([r in amb for r in rows])), len(keys)


if __name__ == "__main__":
    spec, out = sys.argv[1], os.path.join(ROOT, "runs", "readable_rules", sys.argv[2]); os.makedirs(out, exist_ok=True)
    opt = lambda k_, d, c=str: c(sys.argv[sys.argv.index(k_) + 1]) if k_ in sys.argv else d
    kind, k, leaves, bal = opt("--abs", "pca"), opt("--k", 2, int), opt("--leaves", 64, int), opt("--balance", 0.0, float)
    a, n = map(int, spec.split(":")); F = target(spec); rng = np.random.default_rng(0); t0 = time.process_time()
    St, S = teacher_states(a, n); Vq, Hq = abstract(St, kind, k, rng)
    X, y, names = rules.dataset(Vq, Hq, k)
    cur = X[:, 0].astype(np.int64)
    for b in range(k): cur |= X[:, 1 + b].astype(np.int64) << (b + 1)
    chg = y != cur; w = np.where(chg, max(1.0, bal * (~chg).sum() / max(chg.sum(), 1)), 1.0) if bal else None
    coll, nkeys = collisions(X, y)
    lv = rules.fit_tree(X, y, names, max_leaves=leaves, min_leaf=5, w=w); prog = rules.tree_program(lv, k)
    for r in prog["rules"]: r.pop("train_n", None); r.pop("train_purity", None)
    onestep = float(sum(nn * pur for _, _, nn, pur in lv) / len(y))
    model = dict(k=k, S=S, T=n, programs=[prog], schedule=[dict(start=1, end=n, program=0)], first=F[0].astype(np.uint8), h0=Hq[0])
    path = os.path.join(out, "rule.rule.json"); rules.export(path, model, meta=dict(task=spec, method="extract", abstraction=kind, k=k, leaves=leaves, balance=bal, teacher=os.path.relpath(TEACHER, ROOT), teacher_sha256=TEACHER_SHA))
    L = rules.load(path); B, _ = rules.run(L); tB = (St[::S, 0] > 0.5).astype(np.uint8); tB[0] = F[0]
    ev = evaluate(B, F, tB); cx = rules.complexity(L); open(path[:-10] + ".txt", "w").write(rules.to_text(rules.resolve(L["programs"])[0], k) + "\n")
    teacher_vs_orig = float((tB[1:] != F[1:]).mean()); abstract_visible_err = float((Vq[S::S] != F[1:]).mean())
    res = dict(task=spec, abstraction=kind, k=k, leaves=leaves, balance=bal, collision_rate=coll, distinct_inputs=nkeys, onestep_match=onestep, eval=ev, complexity=cx,
               teacher_vs_original_err=teacher_vs_orig, h0_from="abstracted teacher state at the window start (stored)", cpu_s=round(time.process_time() - t0, 1), file=os.path.relpath(path, ROOT))
    json.dump(res, open(os.path.join(out, "result.json"), "w"), indent=1, default=float)
    print(f"{spec} {kind} k{k} L{leaves} b{bal}: err {ev['err']:.4f} (hold {ev['hold_err']:.4f}; teacher {teacher_vs_orig:.4f}) vs teacher {ev['teacher_agreement_err']:.4f} lag4 F1 {ev['lag4_f1']:.2f} mass {ev['lag4_mass'] or 0:.2f} "
          f"| collisions {coll:.3f} one-step {onestep:.3f} | rules {cx['rules']} conds {cx['conditions']} bytes {cx['file_bytes']} | cpu {res['cpu_s']} s", flush=True)
