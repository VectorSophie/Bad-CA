"""Exp13 worker (Campaign 05): direct SAT synthesis of a G-gate circuit rule for one target/structure.
CPU only, 1 thread. Limits are applied by the caller (prlimit --as/--cpu, wall KILL).

On SAT the result is saved IMMEDIATELY (before any further work): gates, outputs, settings, target
SHA-256, solver stats, the SAT model's hidden/intermediate state bits. Then: Python autonomous rollout
of the decoded circuit (binary states), container file, independent Node decode of the file.
usage: exp13_csynth.py TARGET STRUCT G OUTDIR [--lex] [--entries]
"""
import json, os, subprocess, sys, time, resource
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pysat.solvers import Cadical195
from ca.circuit_sat import CircuitSAT
from ca.rule import family_taps, rollout
from ca import costs, container
from c5_targets import STRUCTS, target

tname, sname, G, outdir = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
lex = "--lex" in sys.argv; entries = "--entries" in sys.argv
k, seen, S = STRUCTS[sname]
F = target(tname); T, H, W = F.shape
taps = family_taps(k, seen); fam = "memoryless" if k == 0 else ("F_priv" if seen == "self" else "F_shared")
OUT = os.path.join(ROOT, "runs", outdir); os.makedirs(OUT, exist_ok=True)
stem = os.path.join(OUT, f"{tname.replace(':', '_')}__{sname}__G{G}{'_lex' if lex else ''}{'_ent' if entries else ''}")
res = dict(target=tname, target_sha256=container.sha(F), shape=[T, H, W], struct=sname, family=fam, k=k, S=S, boundary=0,
           hidden_init=0, n_inputs=len(taps), G=G, symmetry="lex+a<b+all used" if lex else "a<b+all used", encoding="v3 entries" if entries else "direct per cell-generation",
           solver="cadical195 (pysat) default", code_snapshot=os.path.basename(os.path.realpath(ROOT)))
t0 = time.process_time()
enc = CircuitSAT(F, taps, k + 1, G, S, lex=lex, entries=entries)
res.update(encode_cpu_s=round(time.process_time() - t0, 2), n_vars=enc.nv, n_clauses=len(enc.cl), circuit_copies=len(enc.cache))
print(json.dumps(dict(progress="encoded", **res)), flush=True)
with Cadical195(bootstrap_with=enc.cl) as s:
    enc.cl = None
    t0 = time.process_time(); ok = s.solve(); res["solve_cpu_s"] = round(time.process_time() - t0, 2)
    res["solver_stats"] = s.accum_stats()
    model = s.get_model() if ok else None
res["status"] = "SAT" if ok else "UNSAT"
res["maxrss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 2)
if ok:
    gates, outs = enc.decode(model)
    res.update(gates=[list(g) for g in gates], outputs=outs)
    json.dump(res, open(stem + ".sat.json", "w"))                       # saved immediately
    np.savez_compressed(stem + ".states.npz", **{"keys": np.array(list(enc.hidden_trajectory(model).keys())),
                                                 "vals": np.array(list(enc.hidden_trajectory(model).values()), np.uint8)})
    luts = costs.eval_circuit(gates, outs, len(taps))
    vis, _, seen_e = rollout(F[0], taps, luts, T, S)
    res["python_rollout_exact"] = bool((vis == F).all()); res["consulted_entries"] = len(seen_e)
    data = container.encode("ca_circuit", F, dict(gates=gates, outputs=outs, n_in=len(taps), family=fam, k=k, S=S))
    open(stem + ".cab", "wb").write(data)
    m, dec, info = container.decode(data)
    res["file_bytes"] = len(data); res["python_file_decode_exact"] = bool((dec == F).all())
    nd = json.loads(subprocess.run(["node", os.path.join(ROOT, "player", "decode_file.mjs"), stem + ".cab"],
                                   capture_output=True, text=True, check=True).stdout)
    res["node_file_decode_exact"] = nd["sha256"] == res["target_sha256"]; res["node_decode_ms"] = nd["decode_ms"]
    res["payload_bits"] = len(costs.rule_circuit(gates, outs, len(taps), fam, k, S))
    res["lut_exc_bits_same_rule"] = len(costs.rule_lut_exceptions(luts, taps, fam, k, S))
json.dump(res, open(stem + ".result.json", "w"), indent=1)
print(json.dumps({x: y for x, y in res.items() if x not in ("gates", "solver_stats")}), flush=True)
