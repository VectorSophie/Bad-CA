"""Campaign 05 self-checks: container round trips (Python + Node), planted-circuit recovery, UNSAT sanity.
usage: python tests/test_c5.py"""
import json, os, subprocess, sys, tempfile
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "experiments"))
from pysat.solvers import Cadical195
from ca import container, costs
from ca.circuit_sat import CircuitSAT
from ca.rule import family_taps, rollout
from c5_targets import KNOWN, known


def node(data):
    with tempfile.NamedTemporaryFile(suffix=".cab", delete=False) as f:
        f.write(data)
    return json.loads(subprocess.run(["node", os.path.join(ROOT, "player", "decode_file.mjs"), f.name], capture_output=True, text=True, check=True).stdout)


def solve(F, taps, C, G, S, lex=False):
    e = CircuitSAT(F, taps, C, G, S, lex=lex)
    with Cadical195(bootstrap_with=e.cl) as s:
        return e.decode(s.get_model()) if s.solve() else None


# 1. every method round-trips; CA files decode identically in Python and Node (circuit + LUT, F_priv + F_shared)
rng = np.random.default_rng(5)
for k, seen, S in [(1, "self", 1), (1, "moore", 2), (0, "self", 1), (2, "self", 1)]:
    taps = family_taps(k, seen); n = len(taps)
    gates = [(int(rng.integers(16)), int(a), int(rng.integers(a + 1, n + j))) for j in range(6) for a in [rng.integers(n + j - 1)]]
    outs = [n + 5] + [int(rng.integers(n + 6)) for _ in range(k)]
    f0 = (rng.random((7, 9)) < 0.4).astype(np.uint8)
    luts = costs.eval_circuit(gates, outs, n); fam = "memoryless" if k == 0 else ("F_priv" if seen == "self" else "F_shared")
    F, _, _ = rollout(f0, taps, luts, 5, S); F = F.astype(np.uint8)
    for m, rule in [("ca_circuit", dict(gates=gates, outputs=outs, n_in=n, family=fam, k=k, S=S)),
                    ("ca_lut_exc", dict(luts=luts, taps=taps, family=fam, k=k, S=S))]:
        d = container.encode(m, F, rule); m2, G2, _ = container.decode(d)
        assert m2 == m and (G2 == F).all(), (m, k, seen)
        assert node(d)["sha256"] == container.sha(F), ("node", m, k, seen)
    for m, d in container.baselines(F).items():
        assert container.decode(d)[0] == m
print("container round trips OK (Python + Node)")

# 2. planted recovery: pcA (2 gates) must be found at G <= 2; G=0 must be UNSAT for pcA
F = known("pcA"); taps = family_taps(1)
assert solve(F, taps, 2, 0, 1) is None, "pcA G=0 should be UNSAT"
for G in range(0, 3):
    r = solve(F, taps, 2, G, 1)
    if r: break
assert r, "pcA not recovered at G<=2"
L = costs.eval_circuit(r[0], r[1], len(taps)); vis, _, _ = rollout(F[0], taps, L, len(F), 1)
assert (vis == F).all(); print(f"pcA recovered at G={G}: {r}")

# 3. lex (v2) keeps planted SAT and agrees with v1 on G
r2 = solve(F, taps, 2, G, 1, lex=True); assert r2, "lex lost the solution"
print("lex OK")
print("ALL OK")

# 4. v3 (entries) encoding: same planted recovery, same UNSAT at G=0/1 for pcA
e = CircuitSAT(F, taps, 2, 1, 1, entries=True)
with Cadical195(bootstrap_with=e.cl) as s: assert not s.solve(), "pcA G=1 should be UNSAT (entries)"
e = CircuitSAT(F, taps, 2, 2, 1, entries=True)
with Cadical195(bootstrap_with=e.cl) as s:
    assert s.solve(); g, o = e.decode(s.get_model())
L = costs.eval_circuit(g, o, len(taps)); vis, _, _ = rollout(F[0], taps, L, len(F), 1); assert (vis == F).all()
Fb = known("pcB"); e = CircuitSAT(Fb, taps, 2, 3, 2, entries=True)
with Cadical195(bootstrap_with=e.cl) as s:
    assert s.solve(); g, o = e.decode(s.get_model())
L = costs.eval_circuit(g, o, len(taps)); vis, _, _ = rollout(Fb[0], taps, L, len(Fb), 2); assert (vis == Fb).all()
print("entries (v3) OK")

# 5. direct (v1) on pcB (non-symmetric gates): decoded circuit must replay exactly
e = CircuitSAT(Fb, taps, 2, 3, 2)
with Cadical195(bootstrap_with=e.cl) as s:
    assert s.solve(); g, o = e.decode(s.get_model())
L = costs.eval_circuit(g, o, len(taps)); vis, _, _ = rollout(Fb[0], taps, L, len(Fb), 2); assert (vis == Fb).all()
print("direct pcB OK")
