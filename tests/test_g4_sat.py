"""Radius-2 outer-totalistic SAT encoder (experiments/rr_gen.py, G4):
sound (a SAT model, exported as dlrule over {v, n8, r2}, replays the clip exactly with the shared executor),
complete (the generating rule's table satisfies the encoding), and separating (radius 1 is UNSAT on a radius-2 rule)."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
import rr_gen
from pysat.solvers import Glucose4

F, p = rr_gen.g4_control(0, T=4, n=12)
r = rr_gen.g4_solve(F, "tot2", 2); assert r["status"] == "SAT" and r["replay_exact_in_dlrule"], r
enc = rr_gen.Tot(2, [rr_gen.RING1, rr_gen.RING2]).extend(F); pins = []
for (v, a, b), var in enc.L.items():
    out = v
    for rule in p["rules"]:
        f = {"v": v, "n8": a, "r2": b}
        if all({"<=": f[n] <= c, ">=": f[n] >= c, "==": f[n] == c}[op] for n, op, c in rule["if"]): out = rule["then"]["v"]; break
    pins.append([var if out else -var])
with Glucose4(bootstrap_with=enc.cl + pins) as s: assert s.solve(), "generator table rejected: encoding over-constrains"
assert rr_gen.g4_solve(F, "tot1", 2)["status"] == "UNSAT"
print("ALL OK")
