"""Path D (logic CA) checks: gate semantics, exact simplification, hard executors and exports.
1. dlca.OP_TT is the 16-gate relaxation evaluated at {0,1} and equals ca.costs.OP_TT.
2. simplify() preserves the local function of random netlists (exhaustive) and equivalent() agrees, and catches a change.
3. The committed exact Life circuit (D0, seed 1) equals B3/S23 on all 512 neighbourhoods, before and after exact
   simplification, and describe_totalistic() returns 'B3/S23'.
4. JAX hard circuit == NumPy layered circuit == flat netlist (random init, CPU).
5. Python and Node replays of a badca/logic-ca-v1 export are bit-identical."""
import hashlib, json, os, subprocess, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
os.environ.setdefault("JAX_PLATFORMS", "cpu")
from ca import dlca, logic_simplify as ls, costs

assert dlca.OP_TT == costs.OP_TT
rng = np.random.default_rng(0)
for trial in range(40):
    n_in = 9; gates = []
    for j in range(rng.integers(5, 60)): gates.append([int(rng.integers(16)), int(rng.integers(n_in + j)), int(rng.integers(n_in + j))])
    net = dict(n_in=n_in, gates=gates, outputs=[n_in + len(gates) - 1])
    s = ls.simplify(net); assert (ls.truth_table(net) == ls.truth_table(s)).all(), trial
    assert ls.equivalent(net, s)[0] is True
    if len(s["gates"]) and s["gates"][-1][0] not in (0, 15):
        t = ls.tie(s, s["n_in"] + len(s["gates"]) - 1, 0)
        if (ls.truth_table(t) != ls.truth_table(s)).any(): assert ls.equivalent(s, t)[0] is False

life = dlca.load(os.path.join(ROOT, "runs/path_d/d0/life_s1/circuit.logic.json"))["net"]
idx = np.arange(512); b = (idx[:, None] >> np.arange(9)) & 1; nb = b.sum(1) - b[:, 4]
ref = ((nb == 3) | ((b[:, 4] == 1) & (nb == 2))).astype(np.uint8)
for n in (life, ls.simplify(life)): assert (ls.truth_table(n)[:, 0] == ref).all()
assert ls.describe_totalistic(ls.simplify(life))["rule"] == "B3/S23"

import jax.numpy as jnp
hp = dict(channels=2, perceive=dict(n_kernels=4, layers=[9, 8, 4, 2], connections=["first_kernel", "unique", "unique", "unique"]),
          update=dict(layers=[18, 32, 16, 8, 4, 2], connections=["unique"] * 6))
params, wires = dlca.init(hp, 3)
params = {k: [p + jnp.asarray(rng.normal(0, 3, p.shape), p.dtype) for p in v] for k, v in params.items()}   # random gates, not pass-through
circ = dlca.harden(params, wires); net = dlca.to_netlist(circ, 2); P = rng.integers(0, 2, (2000, 9, 2)).astype(np.uint8)
J = np.asarray(dlca.circuit(params, wires, jnp.asarray(P, jnp.float32), True)).round().astype(np.uint8)
assert (J == dlca.hard_circuit(circ, P)).all() and (J == dlca.net_eval(net, P)).all()

f = os.path.join(ROOT, "runs/path_d/d0/checker_s2_e2000/circuit.logic.json"); d = dlca.load(f)
B = dlca.net_rollout(d["net"], d["init_state"], 30, d["boundary"] == "periodic")
out = subprocess.run(["node", os.path.join(ROOT, "player/logic_replay.mjs"), f, "30"], capture_output=True, text=True, check=True).stdout
assert json.loads(out)["sha256"] == hashlib.sha256(B.tobytes()).hexdigest()
print("ALL OK")
