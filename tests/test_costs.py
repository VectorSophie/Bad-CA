"""Round-trip checks for every encoder in ca/costs.py."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ca import costs
from ca.rule import family_taps, rollout

rng = np.random.default_rng(0)
F = (rng.random((6, 7, 9)) < 0.3).astype(np.uint8)
F[3] = F[2]  # an empty frame diff
for enc, dec in [(costs.changelist, costs.changelist_decode), (costs.xor_rle, costs.xor_rle_decode),
                 (costs.frame_rle, costs.frame_rle_decode)]:
    w = enc(F); assert (dec(w.b, F[0], len(F)) == F).all(), enc.__name__

taps = family_taps(1); luts = (rng.random((2, 1024)) < 0.5).astype(np.uint8)
w = costs.rule_lut_exceptions(luts, taps, "F_priv", 1, 2)
d = costs.rule_lut_exceptions_decode(w.b)
assert (d["luts"] == luts).all() and d["S"] == 2 and d["k"] == 1

# circuit: x0 XOR x4 for channel 0, NOT x9 for channel 1 (truth-table op codes)
xor = costs.OP_TT[6]; nota = costs.OP_TT[12]
gates = [(xor, 0, 4), (nota, 9, 9)]
w = costs.rule_circuit(gates, [10, 11], 10, "F_priv", 1, 2)
d = costs.rule_circuit_decode(w.b)
L = costs.eval_circuit(d["gates"], d["outputs"], 10)
idx = np.arange(1024)
assert (L[0] == (((idx >> 0) ^ (idx >> 4)) & 1)).all() and (L[1] == 1 - ((idx >> 9) & 1)).all()
print("ok")

# OP_TT must match the DiffLogic relaxation at binary inputs, and extraction must equal hard_lut
import jax.numpy as jnp
from ca.difflogic import ops, make_wiring, init_logits, hard_lut, extract_circuit
o = np.asarray(ops(jnp.array([0., 0., 1., 1.]), jnp.array([0., 1., 0., 1.]))).round().astype(int)  # (4 inputs, 16 ops)
for i in range(16):
    assert sum(o[q, i] << q for q in range(4)) == costs.OP_TT[i], i
w = make_wiring(np.random.default_rng(3), 10, 2)
P = [l + jnp.asarray(np.random.default_rng(j).normal(0, 20, l.shape)) for j, l in enumerate(init_logits(w))]  # random hard ops
g, outs = extract_circuit(P, w, 10)
assert (costs.eval_circuit(g, outs, 10) == hard_lut(P, w, 10)).all()
print("extract ok, live gates", len(g))
