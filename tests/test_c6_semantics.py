"""Campaign 06: the SAT encoding, Python evaluator and Node player agree on every gate op, on direct
wiring (G=0) and constants; SAT is monotone in G (buffer padding); nodegen keeps the planted
minimal sizes. (Container round trips and planted recovery are already in tests/test_c5.py.)"""
import json, os, subprocess, sys, tempfile
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "experiments"))
from pysat.solvers import Cadical195
from ca import container, costs
from ca.circuit_sat import CircuitSAT
from ca.rule import family_taps, rollout
from c5_targets import known

taps = family_taps(1); n = len(taps)
f0 = (np.random.default_rng(3).random((8, 8)) < 0.5).astype(np.uint8)


def clip(gates, outs, S=1, T=4):
    L = costs.eval_circuit(gates, outs, n); v, _, _ = rollout(f0, taps, L, T, S); return v.astype(np.uint8)


def forced(F, gates, outs, S=1):
    """SAT with every structural variable fixed to the given circuit?"""
    e = CircuitSAT(F, taps, 2, len(gates), S)
    units = [[v if (tt >> q) & 1 else -v] for (tt, a, b), vs in zip(gates, e.tt) for q, v in enumerate(vs)]
    units += [[e.sa[j][a]] for j, (_, a, b) in enumerate(gates)] + [[e.sb[j][b]] for j, (_, a, b) in enumerate(gates)]
    units += [[e.so[c][o]] for c, o in enumerate(outs)]
    with Cadical195(bootstrap_with=e.cl + units) as s:
        return s.solve()


def node_ok(F, gates, outs, S=1):
    d = container.encode("ca_circuit", F, dict(gates=gates, outputs=outs, n_in=n, family="F_priv", k=1, S=S))
    with tempfile.NamedTemporaryFile(suffix=".cab", delete=False) as f: f.write(d)
    r = json.loads(subprocess.run(["node", os.path.join(ROOT, "player", "decode_file.mjs"), f.name], capture_output=True, text=True, check=True).stdout)
    return r["sha256"] == container.sha(F) and (container.decode(d)[1] == F).all()


# every op (incl. constants 0/15, buffers, NOTs) as gate 0 on inputs (W=3, E=5), output visible, hidden = centre
for op in range(16):
    g = [(op, 3, 5)]; F = clip(g, [n, 4])
    assert forced(F, g, [n, 4]), f"encoding rejects the true circuit, op {op}"
    assert node_ok(F, g, [n, 4]), f"node/python mismatch, op {op}"
    for op2 in range(16):            # any op with a different rollout must be rejected
        if op2 != op and not (clip([(op2, 3, 5)], [n, 4]) == F).all():
            assert not forced(F, [(op2, 3, 5)], [n, 4]), f"encoding accepts wrong op {op2} for {op}"
print("16 ops: encoding == Python == Node")
# direct wiring G=0 and a 2-gate chain feeding the hidden channel, S=2
for gates, outs in [([], [3, 4]), ([], [9, 0]), ([(6, 1, 9), (8, 4, 10)], [11, 10])]:
    F = clip(gates, outs, S=2)
    assert forced(F, gates, outs, S=2) and node_ok(F, gates, outs, S=2), (gates, outs)
print("G=0 wiring and S=2 chain OK")


def sat(F, G, S, **kw):
    e = CircuitSAT(F, taps, 2, G, S, **kw)
    with Cadical195(bootstrap_with=e.cl) as s: return s.solve()


A, B = known("pcA"), known("pcB")
assert all(sat(A, G, 1) for G in (2, 3, 4)) and all(sat(A, G, 1, lex=True) for G in (3, 4)), "padding monotonicity"
assert [sat(A, G, 1, nodegen=True) for G in range(4)] == [False, False, True, True]
assert [sat(B, G, 2, nodegen=True) for G in range(4)] == [False, False, False, True]
print("monotone in G; nodegen keeps planted minima (pcA 2, pcB 3)")
print("ALL OK")
