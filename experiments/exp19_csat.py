"""Exp19 (Campaign 06) worker: encode one instance to a DIMACS file once; solve that SAME file with a
chosen solver in a separate process. CPU only, 1 thread; limits applied by the driver.

  encode TARGET STRUCT G VARIANT        -> runs/c6/cnf/<inst>.cnf (+ .vm.json, .meta.json with sha256)
  solve  INST SOLVER OUTDIR [CUBE]      -> runs/<OUTDIR>/<inst>__<solver>[__cube].result.json
VARIANT: '+'-joined subset of {base, lex, nodegen, chain, hidc, witness}
  lex      symmetry v2 (Campaign 05)                       [existence-preserving]
  nodegen  no buffers; constants/NOT feed outputs only      [existence-preserving for 'at most G']
  chain    gate j reads gate j-1                            [heuristic: restricted wiring]
  hidc     hidden output = visible centre tap (one-step delay) [heuristic: hidden type fixed]
  witness  every unobserved state bit fixed to the exp03c LUT witness trajectory
           [uses an EXISTING solution; never counted as video-only discovery]
CUBE: comma-separated literals of the selection variables, given as 'so0=<p>' etc. (see c6_cubes.py),
added as unit clauses (cube-and-conquer); recorded in the result.
On SAT the circuit is saved immediately, then replayed in Python, written as a container file and
decoded by the Node player.
"""
import hashlib, json, os, resource, subprocess, sys, time
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca.circuit_sat import CircuitSAT
from ca.rule import family_taps, index, rollout
from ca import costs, container
from c5_targets import STRUCTS, target

CNF_DIR = os.path.join(ROOT, "runs", "c6", "cnf")
WITNESS = os.path.join(ROOT, "runs", "exp03c", "ba32_3000_5_E.rule.json")


def inst_name(t, st, G, var):
    return f"{t.replace(':', '_')}__{st}__G{G}__{var}"


def witness_states(F, taps, S, C):
    """Every state bit (g, c, y, x) of the exp03c witness rollout, all generations."""
    import base64
    ex = json.load(open(WITNESS)); n = len(ex["taps"])
    L = np.stack([np.unpackbits(np.frombuffer(base64.b64decode(b), np.uint8), bitorder="little")[:2 ** n] for b in ex["luts_b64"]])
    s = np.zeros((C,) + F[0].shape, np.uint8); s[0] = F[0]; out = {}
    for g in range(1, (len(F) - 1) * S + 1):
        s = L[:, index(s, taps)].astype(np.uint8)
        for c in range(C):
            if c == 0 and g % S == 0: assert (s[0] == F[g // S]).all(); continue
            for (y, x), v in np.ndenumerate(s[c]): out[g, c, y, x] = int(v)
    return out


def encode(t, st, G, var):
    k, seen, S = STRUCTS[st]; F = target(t); taps = family_taps(k, seen); V = set(var.split("+"))
    t0 = time.process_time()
    e = CircuitSAT(F, taps, k + 1, G, S, lex="lex" in V, nodegen="nodegen" in V, chain="chain" in V,
                   fix_out={1: 4} if "hidc" in V else None, fix_states=witness_states(F, taps, S, k + 1) if "witness" in V else None)
    name = inst_name(t, st, G, var); os.makedirs(CNF_DIR, exist_ok=True); path = os.path.join(CNF_DIR, name + ".cnf")
    h = hashlib.sha256()
    with open(path, "w") as f:
        head = f"p cnf {e.nv} {len(e.cl)}\n"; f.write(head); h.update(head.encode())
        for c in e.cl:
            line = " ".join(map(str, c)) + " 0\n"; f.write(line); h.update(line.encode())
    meta = dict(inst=name, target=t, target_sha256=container.sha(F), struct=st, family="memoryless" if k == 0 else ("F_priv" if seen == "self" else "F_shared"),
                k=k, S=S, G=G, variant=var, n_inputs=len(taps), n_vars=e.nv, n_clauses=len(e.cl), cnf_sha256=h.hexdigest(),
                encode_cpu_s=round(time.process_time() - t0, 2), code_snapshot=os.path.basename(os.path.realpath(ROOT)))
    json.dump(e.varmap(), open(os.path.join(CNF_DIR, name + ".vm.json"), "w"))
    json.dump(meta, open(os.path.join(CNF_DIR, name + ".meta.json"), "w"), indent=1)
    print(json.dumps(meta))


def solve(name, solver, outdir, cube=""):
    from pysat.formula import CNF
    from pysat.solvers import Solver
    meta = json.load(open(os.path.join(CNF_DIR, name + ".meta.json"))); vm = json.load(open(os.path.join(CNF_DIR, name + ".vm.json")))
    OUT = os.path.join(ROOT, "runs", outdir); os.makedirs(OUT, exist_ok=True)
    stem = os.path.join(OUT, f"{name}__{solver}" + (f"__cube_{cube.replace(',', '_').replace('=', '')}" if cube else ""))
    t0 = time.process_time(); f = CNF(from_file=os.path.join(CNF_DIR, name + ".cnf")); load_s = time.process_time() - t0
    units = []
    for lit in filter(None, cube.split(",")):            # e.g. so0=13 or -so0=13 (selection var of output 0 = pool 13)
        neg = lit.startswith("-"); key, p = lit.lstrip("-").split("=")
        v = vm["so"][int(key[2:])][int(p)] if key.startswith("so") else None
        units.append([-v if neg else v])
    res = dict(meta, solver=solver, solver_impl="pysat " + __import__("pysat").__version__ + " bundled", seed="solver default (not exposed by pysat)",
               cube=cube or None, load_cpu_s=round(load_s, 2), proof="not produced/verified (no DRAT checker available)")
    t0 = time.process_time()
    with Solver(name=solver, bootstrap_with=f.clauses + units) as s:
        ok = s.solve(); model = s.get_model() if ok else None
    res.update(status="SAT" if ok else "UNSAT", solve_cpu_s=round(time.process_time() - t0, 2),
               maxrss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 2))
    if ok:
        gates, outs = CircuitSAT.decode_varmap(vm, model)
        res.update(gates=[list(g) for g in gates], outputs=outs)
        json.dump(res, open(stem + ".sat.json", "w"))                       # saved immediately
        F = target(meta["target"]); taps = family_taps(meta["k"], "moore" if meta["family"] == "F_shared" else "self")
        L = costs.eval_circuit(gates, outs, len(taps)); vis, _, _ = rollout(F[0], taps, L, len(F), meta["S"])
        res["python_rollout_exact"] = bool((vis == F).all())
        data = container.encode("ca_circuit", F, dict(gates=gates, outputs=outs, n_in=len(taps), family=meta["family"], k=meta["k"], S=meta["S"]))
        open(stem + ".cab", "wb").write(data)
        m, dec, info = container.decode(open(stem + ".cab", "rb").read())
        nd = json.loads(subprocess.run(["node", os.path.join(ROOT, "player", "decode_file.mjs"), stem + ".cab"], capture_output=True, text=True, check=True).stdout)
        res.update(file_bytes=len(data), rule_bits=len(costs.rule_circuit(gates, outs, len(taps), meta["family"], meta["k"], meta["S"])),
                   python_file_decode_exact=bool((dec == F).all()), node_file_decode_exact=nd["sha256"] == meta["target_sha256"], node_decode_ms=nd["decode_ms"])
    json.dump(res, open(stem + ".result.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("gates",)}))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "encode": encode(a[1], a[2], int(a[3]), a[4])
    else: solve(a[1], a[2], a[3], a[4] if len(a) > 4 else "")
