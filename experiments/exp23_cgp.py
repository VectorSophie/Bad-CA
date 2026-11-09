"""Exp23 (Phase 07, candidate C4): Cartesian Genetic Programming over small 2-input-gate circuits,
fitness = HARD autonomous rollout error (no relaxation, no soft/hard gap).

Genome: G nodes (tt 4 bits, inputs a, b in [0, n + j)), C outputs (any pool index). Active gates =
those reachable from outputs. (1 + 4)-ES (Miller's CGP standard): each child = parent with point
mutations until >= 1 active gene changed (Goldman & Punch 'single active mutation'); a child replaces
the parent if its fitness is <= parent's (neutral drift). Fitness = (mean pixel error over frames 1..T-1,
active gates) lexicographic. Every improvement is saved immediately. Stops on exact (error 0) or budget.
usage: exp23_cgp.py TARGET STRUCT G SEED BUDGET_CPU_S OUTDIR
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c7_common import C7, ROOT, baselines_bytes, evaluate, export_verify, media, target
from c5_targets import STRUCTS
from ca import costs
from ca.rule import family_taps, rollout

tname, st, G, seed, budget, outdir = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), sys.argv[6]
k, seen, S = STRUCTS[st]; taps = family_taps(k, seen); n = len(taps); C = k + 1
fam = "memoryless" if k == 0 else ("F_priv" if seen == "self" else "F_shared")
F = target(tname); T = len(F); rng = np.random.default_rng(seed)
OUT = os.path.join(ROOT, "runs", outdir); os.makedirs(OUT, exist_ok=True)
stem = os.path.join(OUT, f"{tname.replace(':', '_')}__{st}__G{G}__s{seed}")


def active(gates, outs):
    act, stack = set(), [o for o in outs if o >= n]
    while stack:
        p = stack.pop()
        if p in act: continue
        act.add(p); op, a, b = gates[p - n]
        stack += [x for x in (a, b) if x >= n]
    return act


def compact(gates, outs):
    """Active sub-circuit with renumbered pool (for export/size)."""
    act = sorted(active(gates, outs)); m = {i: i for i in range(n)}
    g2 = []
    for p in act:
        op, a, b = gates[p - n]; g2.append((op, m[a], m[b])); m[p] = n + len(g2) - 1
    return g2, [m[o] for o in outs]


def fitness(gates, outs):
    L = costs.eval_circuit(gates, outs, n)
    vis, _, _ = rollout(F[0], taps, L, T, S)
    return float((vis[1:] != F[1:]).mean()), len(active(gates, outs)), vis


def random_genome():
    return [[int(rng.integers(16)), int(rng.integers(n + j)), int(rng.integers(n + j))] for j in range(G)], [int(rng.integers(n + G)) for _ in range(C)]


def mutate(gates, outs):
    g = [list(x) for x in gates]; o = list(outs); act = active(g, o)
    while True:
        if rng.random() < C / (3 * G + C):
            c = int(rng.integers(C)); o[c] = int(rng.integers(n + G)); return g, o
        j = int(rng.integers(G)); f = int(rng.integers(3))
        g[j][f] = int(rng.integers(16)) if f == 0 else int(rng.integers(n + j))
        if n + j in act: return g, o


t0 = time.process_time(); par = random_genome(); pf = fitness(*par); evals = 1; trace = []
hold = float((F[1:] != F[:1]).mean())
while pf[0] > 0 and time.process_time() - t0 < budget:
    for _ in range(4):
        ch = mutate(*par); cf = fitness(*ch); evals += 1
        if (cf[0], cf[1]) <= (pf[0], pf[1]):
            if (cf[0], cf[1]) < (pf[0], pf[1]):
                trace.append(dict(cpu_s=round(time.process_time() - t0, 1), evals=evals, err=cf[0], active=cf[1]))
                json.dump(dict(gates=ch[0], outputs=ch[1], err=cf[0], evals=evals), open(stem + ".best_sofar.json", "w"))  # saved immediately
            par, pf = ch, cf
search_cpu = time.process_time() - t0
g2, o2 = compact(*par)
L = costs.eval_circuit(g2, o2, n); P, _, _ = rollout(F[0], taps, L, T, S); assert (P == pf[2]).all()
ev = evaluate(P, F)
ex = export_verify(P.astype(np.uint8), F, dict(gates=g2, outputs=o2, n_in=n, family=fam, k=k, S=S), "ca_circuit", stem + ".best.cab")
res = dict(method="C4 CGP (1+4)-ES, hard-rollout fitness", target=tname, struct=st, G_max=G, seed=seed, budget_cpu_s=budget, search_cpu_s=round(search_cpu, 1),
           evals=evals, active_gates=len(g2), gates=g2, outputs=o2, rule_bits=len(costs.rule_circuit(g2, o2, n, fam, k, S)),
           eval=ev, export=ex, hold_err=hold, trace=trace, baselines_bytes=baselines_bytes(F), media=media(P.astype(np.uint8), F, stem + ".best"))
json.dump(res, open(stem + ".result.json", "w"), indent=1)
print(json.dumps(dict(target=tname, struct=st, G=G, seed=seed, evals=evals, err=ev["mean_err"], hold=hold, exact=ev["exact"], change_iou=ev["change_iou"],
                      horizon=ev["horizon"], active=len(g2), bytes=ex["file_bytes"], node=ex["node_replay_equals_P"])))
