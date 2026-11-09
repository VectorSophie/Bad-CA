"""Exp03c: compactness objective for exact SAT synthesis (CPU; never via gpuq).

Default rule = 'keep': visible -> its own centre bit, hidden -> its own hidden bit.
An exception is a LUT entry (present in the CNF) whose output differs from that default.
Anytime descent: solve, count exceptions E, add totalizer bound <= E-1, repeat until UNSAT
(E proved minimal for this family/encoding). Every improvement is saved immediately; the run is
ended by the outer `timeout` (CaDiCaL ignored/crashed on thread interrupts in v1 of this script).
Then `verify` re-simulates the saved best, exports it and replays it in Node.
Sizes (verify): decode-checked bitstreams from ca/costs.py -- rule = header + 'keep' default +
gap-coded exception indices; recordings = change list / XOR-RLE / frame-RLE of the same frames.
(v1 used a rough E x ceil(log2(C*2^n)) estimate; superseded.)

With REF_LUTS (.npy, shape (C, 2^n)) the objective is Hamming distance to that rule instead of
'keep' (used for SAT repair of a hardened DiffLogic rule); the size encoding stays keep-based.

usage: exp03c_min_exceptions.py search TARGET COND [OUTDIR] [REF_LUTS]
       exp03c_min_exceptions.py verify TARGET COND [OUTDIR] [REF_LUTS]
"""
import json, math, os, subprocess, sys, time
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pysat.solvers import Cadical195
from pysat.card import ITotalizer
from ca.rule import family_taps, rollout
from ca.sat import encode, luts_from_model
from ca import costs
from ca.export import export_rule
from exp03_sat_ablation import CONDS, target

mode, tname, cond = sys.argv[1:4]
k, S, seen_by = CONDS[cond]; C = k + 1
taps = family_taps(k, seen_by); n = len(taps)
own = {0: 4}; own.update({c: taps.index((c, 0, 0)) for c in range(1, C)})
F = target(tname); T = len(F)
OUT = os.path.join(ROOT, "runs", sys.argv[4] if len(sys.argv) > 4 else "exp03c"); os.makedirs(OUT, exist_ok=True)
SNAP = os.path.basename(os.path.realpath(ROOT))
stem = os.path.join(OUT, f"{tname}_{cond}")
idx_all = np.arange(2 ** n)
REF = np.load(sys.argv[5]) if len(sys.argv) > 5 else np.stack([(idx_all >> own[c]) & 1 for c in range(C)]).astype(np.uint8)


if mode == "search":
    t_start = time.perf_counter()
    s = Cadical195()
    n_clauses, L, nv = encode(F, taps, C, S, sink=s.add_clause)  # streamed: long clips don't fit as lists
    t_enc = round(time.perf_counter() - t_start, 1)
    print(json.dumps(dict(encoded_s=t_enc, n_clauses=n_clauses, n_vars=nv, code_snapshot=SNAP)), flush=True)
    exc = [(v if not REF[c, e] else -v) for (c, e), v in L.items()]  # true <=> entry deviates from REF
    keys = np.array(list(L.keys()))
    np.save(stem + ".lutkeys.npy", keys)
    trace, tot = [], None  # totalizer built after first model, bounded by its E (unbounded is O(n^2): memout)
    meta = dict(encode_s=t_enc, n_clauses=n_clauses, n_vars=nv, code_snapshot=SNAP)
    with s:
        s.set_phases([-x for x in exc])  # start near the default rule
        while True:
            if not s.solve():
                status = "OPTIMAL" if trace else "UNSAT"; break
            m = s.get_model(); pos = set(x for x in m if x > 0)
            E = sum((x in pos) if x > 0 else (-x not in pos) for x in exc)
            trace.append(dict(t=round(time.perf_counter() - t_start, 1), exceptions=E))
            np.save(stem + ".best_luts.npy", luts_from_model(m, L, C, n))
            json.dump(dict(trace=trace, status="running", **meta), open(stem + ".trace.json", "w"))
            print(json.dumps(trace[-1]), flush=True)
            if E == 0:
                status = "OPTIMAL"; break
            if tot is None:
                tot = ITotalizer(lits=exc, ubound=E, top_id=nv); s.append_formula(tot.cnf.clauses)
            s.add_clause([-tot.rhs[E - 1]])  # at most E-1 exceptions
    json.dump(dict(trace=trace, status=status, **meta), open(stem + ".trace.json", "w"))
    sys.exit(0)

# verify: status is OPTIMAL only if the search itself finished; otherwise best-so-far under the wall budget
if not os.path.exists(stem + ".trace.json"):  # trace.json appears with the first solution
    st = "UNKNOWN_timeout_no_solution" if os.path.exists(stem + ".lutkeys.npy") else "UNKNOWN_timeout_before_encoding_done"
    json.dump(dict(target=tname, cond=cond, status=st, changed_pixels=int((F[1:] != F[:-1]).sum()),
                   shared=costs.shared_info(F), recording_bits=costs.recording_costs(F)), open(stem + ".result.json", "w"), indent=1)
    sys.exit(0)
tr = json.load(open(stem + ".trace.json"))
status = tr["status"] if tr["status"] != "running" else "BEST_SO_FAR_budget"
if status == "running": status = "BEST_SO_FAR_budget"
if not tr["trace"] and status != "UNSAT": status = "UNKNOWN_timeout_no_solution"
fam = "memoryless" if k == 0 else ("F_priv" if seen_by == "self" else "F_shared")
res = dict(target=tname, cond=cond, hidden_bits=k, steps_per_frame=S, family=fam, trace=tr["trace"], status=status,
           search_code_snapshot=tr.get("code_snapshot"), encode_s=tr.get("encode_s"), n_clauses=tr.get("n_clauses"),
           changed_pixels=int((F[1:] != F[:-1]).sum()), shared=costs.shared_info(F), recording_bits=costs.recording_costs(F))
if tr["trace"]:
    luts = np.load(stem + ".best_luts.npy"); keys = np.load(stem + ".lutkeys.npy")
    idx = np.arange(2 ** n)
    for c in range(C):  # entries absent from the CNF are never consulted: set them to REF
        free = np.ones(2 ** n, bool); free[keys[keys[:, 0] == c, 1]] = False
        luts[c, free] = REF[c, free]
    E = int((luts != REF).sum())
    assert E == tr["trace"][-1]["exceptions"], "exception count mismatch -> bookkeeping bug"
    vis, _, seen = rollout(F[0], taps, luts, T, S)
    path = stem + ".rule.json"
    ex = export_rule(path, taps, luts, F[0], T, S, meta=dict(source=tname, method="SAT min-exceptions", cond=cond,
                                                           exceptions=E, status=status), expected=F)
    node = json.loads(subprocess.run(["node", os.path.join(ROOT, "player", "replay.mjs"), path], capture_output=True, text=True, check=True).stdout)
    rb = costs.rule_lut_exceptions(luts, taps, fam, k, S)
    d = costs.rule_lut_exceptions_decode(rb.b)
    vis2, _, _ = rollout(F[0], d["taps"], d["luts"], T, d["S"], d["boundary"], d["hidden_init"])
    res.update(objective="hamming to REF_LUTS" if len(sys.argv) > 5 else "exceptions vs keep",
               ref_luts=sys.argv[5] if len(sys.argv) > 5 else None,
               keep_exceptions=int((luts != np.stack([(idx >> own[c]) & 1 for c in range(C)])).sum()),
               best_exceptions=E, rule_bits=len(rb), rule_bitstream_decodes_to_exact=bool((vis2 == F).all()),
               python_rollout_exact=bool((vis == F).all()), node_hash_matches_target=node["sha256"] == ex["expected_sha256"],
               visited_entries=len(seen))
json.dump(res, open(stem + ".result.json", "w"), indent=1)
print(json.dumps({kk: v for kk, v in res.items() if kk != "trace"}))
