"""Exp27 (Campaign 08): CGP v2 = Campaign 07 CGP ((1+4)-ES, single-active-gene mutation, neutral drift) with
(i) direct gate-by-gate grid simulation (any taps, so F_shared/S=4 cost no 2^n LUT), (ii) a selectable objective:
  pix : (mean pixel error over frames 1..T-1, active gates)                    [C7 objective, arms A/C]
  mot : (J_lam = mean_seg[err/hold + lam(1 - F1tol)], active gates)          [motion objective, arms B/D]
Only the TARGET FRAMES are given to the search (no generator circuit, no hidden trajectory).
Saved immediately on every improvement: genome, fitness, evals, RNG state. Stops on exact replay or CPU budget.
usage: exp27_cgp2.py TARGET STRUCT G SEED BUDGET_CPU_S OBJ LAM OUTDIR
  OBJ motcur = mot on a prefix growing 20 -> 40 -> T frames in equal CPU slices (genome carried over); final evaluation on all T frames.
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c7_common import ROOT, baselines_bytes, export_verify, target as t7
from c8_targets import STRUCT, control, simulate, taps_of
from ca import costs
from ca.motion import objective, report

tname, st, G, seed, budget, obj, lam, outdir = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), sys.argv[6], float(sys.argv[7]), sys.argv[8]
k, seen, S = STRUCT[st]; n = len(taps_of(st)); C = k + 1; fam = "F_priv" if seen == "self" else "F_shared"
F = control(tname)[0] if tname.startswith("pc") else t7(tname)
T = len(F); rng = np.random.default_rng(seed)
OUT = os.path.join(ROOT, "runs", outdir); os.makedirs(OUT, exist_ok=True)
stem = os.path.join(OUT, f"{tname}__{st}__{obj}{lam:g}__G{G}__s{seed}")


def active(g, o):
    act, stack = set(), [x for x in o if x >= n]
    while stack:
        p = stack.pop()
        if p in act: continue
        act.add(p); _, a, b = g[p - n]; stack += [x for x in (a, b) if x >= n]
    return act


CUR = [20, 40, T] if obj == "motcur" else [T]   # curriculum: prefix length grows in equal CPU slices
Lcur = CUR[0]


def fit(g, o):
    Fp = F[:Lcur]; P = simulate(g, o, st, F[0], Lcur)
    f = float((P[1:] != Fp[1:]).mean()) if obj == "pix" else objective(P, Fp, lam)
    return f, len(active(g, o)), P


def mutate(g, o):
    g = [list(x) for x in g]; o = list(o); act = active(g, o)
    while True:
        if rng.random() < C / (3 * G + C):
            o[int(rng.integers(C))] = int(rng.integers(n + G)); return g, o
        j = int(rng.integers(G)); f = int(rng.integers(3))
        g[j][f] = int(rng.integers(16)) if f == 0 else int(rng.integers(n + j))
        if n + j in act: return g, o


def save(g, o, f, evals, why):
    json.dump(dict(gates=g, outputs=o, fit=f, evals=evals, why=why, cpu_s=round(time.process_time() - t0, 2), rng_state=rng.bit_generator.state,
                   args=sys.argv[1:]), open(stem + ".ckpt.json", "w"))


t0 = time.process_time()
par = ([[int(rng.integers(16)), int(rng.integers(n + j)), int(rng.integers(n + j))] for j in range(G)], [int(rng.integers(n + G)) for _ in range(C)])
pf = fit(*par); evals = 1; trace = []; last = time.process_time()
stage = 0
while time.process_time() - t0 < budget and not (len(pf[2]) == T and (pf[2] == F).all()):
    want = min(int((time.process_time() - t0) / (budget / len(CUR))), len(CUR) - 1)
    if want != stage:   # curriculum stage change: re-score the parent on the longer prefix
        stage = want; Lcur = CUR[stage]; pf = fit(*par); save(par[0], par[1], pf[0], evals, f"stage {stage} L={Lcur}")
    if len(pf[2]) == T and (pf[2] == F).all(): break
    for _ in range(4):
        ch = mutate(*par); cf = fit(*ch); evals += 1
        if (cf[0], cf[1]) <= (pf[0], pf[1]):
            if (cf[0], cf[1]) < (pf[0], pf[1]):
                trace.append(dict(cpu_s=round(time.process_time() - t0, 1), evals=evals, fit=cf[0], active=cf[1])); save(ch[0], ch[1], cf[0], evals, "improvement")
            par, pf = ch, cf
    if time.process_time() - last > 60: save(par[0], par[1], pf[0], evals, "periodic"); last = time.process_time()
search_cpu = time.process_time() - t0
# compact active sub-circuit, re-simulate, export, verify (Python + Node replay of the FILE == simulation)
act = sorted(active(*par)); m = {i: i for i in range(n)}; g2 = []
for p in act:
    tt, a, b = par[0][p - n]; g2.append((tt, m[a], m[b])); m[p] = n + len(g2) - 1
o2 = [m[x] for x in par[1]]
P = simulate(g2, o2, st, F[0], T)   # evaluation is ALWAYS the full T-frame autonomous replay
assert (P[:len(pf[2])] == pf[2]).all(), "compaction changed the replay"
ex = export_verify(P.astype(np.uint8), F, dict(gates=g2, outputs=o2, n_in=n, family=fam, k=k, S=S), "ca_circuit", stem + ".best.cab")
rep = report(P, F, lam=1.0)
res = dict(campaign=8, target=tname, struct=st, k=k, S=S, family=fam, objective=obj, lam=lam, G_max=G, seed=seed, budget_cpu_s=budget,
           search_cpu_s=round(search_cpu, 1), evals=evals, evals_per_cpu_s=round(evals / max(search_cpu, 1e-9), 2), active_gates=len(g2), gates=g2, outputs=o2,
           rule_bits=len(costs.rule_circuit(g2, o2, n, fam, k, S)), final_fit=pf[0], report=rep, export=ex, trace=trace,
           baselines_bytes=baselines_bytes(F) if not tname.startswith("pc") else None)
json.dump(res, open(stem + ".result.json", "w"), indent=1, default=float)
r = rep
print(json.dumps(dict(target=tname, st=st, obj=obj, lam=lam, G=G, seed=seed, evals=evals, active=len(g2), exact=rep["exact"], J1=round(rep["J"], 3),
                      f1=[round(r[s]["f1_tol"], 3) for s in ("early", "mid", "late")], err_hold=[round(r[s]["err"] / max(r[s]["hold"], 1e-9), 3) for s in ("early", "mid", "late")],
                      late_period=rep["late_period"], bytes=ex["file_bytes"], node=ex["node_replay_equals_P"])))
