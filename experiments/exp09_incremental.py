"""Exp09 (Campaign 04, part 1): incremental vs cold-start exact SAT across clip lengths. CPU only.

Same clip/start/family for every arm: frames [START, START+L) of ba32 (target name ba32_<START>_<L>),
condition COND (exp03 CONDS). Plain feasibility: the first exact rule, no size objective.

  cold: a fresh CaDiCaL instance, encode L frames, solve.  One process per length.
  incr: ONE CaDiCaL instance; for each length in SCHEDULE append the new generations' clauses
        (ca.sat.Encoder) and solve again. Learned clauses carry over.
PHASE (solver decision preference, never a hard constraint):
  keep: LUT vars prefer the 'keep' default (as exp03/exp03c).  New LUT vars get it as they appear.
  prev: (incr only) after each solved length, every variable prefers its value in that model;
        LUT vars first seen later prefer 'keep'.
Each solved length is logged immediately (one JSON line) so the outer `timeout -s KILL` loses
nothing; a length with no line is UNKNOWN_timeout. Times exclude video loading; `step_s` is
encode + solve for that length, `cum_s` the sum of all steps so far in this process.

usage: exp09_incremental.py cold START L COND PHASE OUTDIR
       exp09_incremental.py incr START MAXL COND PHASE SCHEDULE OUTDIR   (SCHEDULE: step1 | 5,10,20,40)
"""
import json, os, resource, sys, time
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pysat.solvers import Cadical195
from ca.rule import family_taps, rollout
from ca.sat import Encoder, luts_from_model
from exp03_sat_ablation import CONDS, target

REPORT = (5, 10, 20, 40)
mode, start = sys.argv[1], int(sys.argv[2])
if mode == "cold":
    maxl, cond, phase, outdir = int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6]
    lengths = [maxl]; arm = f"cold_{phase}_L{maxl}"
else:
    maxl, cond, phase, sched, outdir = int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7]
    lengths = list(range(2, maxl + 1)) if sched == "step1" else [int(x) for x in sched.split(",")]
    arm = f"incr_{phase}_{sched.replace(',', '-')}"
assert phase in ("keep", "prev") and not (mode == "cold" and phase == "prev")
k, S, seen_by = CONDS[cond]; C = k + 1
taps = family_taps(k, seen_by); n = len(taps)
own = {0: 4}; own.update({c: taps.index((c, 0, 0)) for c in range(1, C)})
F = target(f"ba32_{start}_{maxl}")
OUT = os.path.join(ROOT, "runs", outdir); os.makedirs(OUT, exist_ok=True)
stem = os.path.join(OUT, f"ba32_{start}_{cond}_{arm}")
log = open(stem + ".jsonl", "a")
def emit(d):
    log.write(json.dumps(d) + "\n"); log.flush(); print(json.dumps(d), flush=True)

emit(dict(event="start", mode=mode, arm=arm, start=start, cond=cond, phase=phase, lengths=lengths,
          code_snapshot=os.path.basename(os.path.realpath(ROOT)), solver="cadical195 (pysat), default options"))
cum = 0.0
with Cadical195() as s:
    enc = Encoder(taps, C, S, sink=s.add_clause)
    for L in lengths:
        t0 = time.perf_counter(); nl0 = len(enc.L)
        enc.extend(F[:L])
        new = list(enc.L.items())[nl0:]
        s.set_phases([v if (e >> own[c]) & 1 else -v for (c, e), v in new])
        t1 = time.perf_counter()
        ok = s.solve()
        t2 = time.perf_counter(); cum += t2 - t0
        rec = dict(event="solved", L=L, status="SAT" if ok else "UNSAT", encode_s=round(t1 - t0, 2), solve_s=round(t2 - t1, 2),
                   step_s=round(t2 - t0, 2), cum_s=round(cum, 2), n_clauses=enc.n_cl, n_vars=enc.nv,
                   maxrss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 2))
        if not ok:
            emit(rec); break  # proof for this family/encoding: no longer clip can be SAT either
        m = s.get_model()
        luts = luts_from_model(m, enc.L, C, n)
        vis, _, _ = rollout(F[0], taps, luts, L, S)
        rec["python_rollout_exact"] = bool((vis == F[:L]).all())
        assert rec["python_rollout_exact"], "SAT model does not replay -> encoder bug"
        if L in REPORT:
            np.save(f"{stem}.L{L}.luts.npy", luts)
        emit(rec)
        if phase == "prev":
            s.set_phases(m)
emit(dict(event="done", cum_s=round(cum, 2)))
