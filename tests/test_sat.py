"""Encoder sanity: planted solutions must be SAT and replay exactly; proved-infeasible must be UNSAT."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pysat.solvers import Cadical195
from ca.core import life_like_lut
from ca.rule import family_taps, rollout
from ca.sat import Encoder, encode, luts_from_model, state_repeat_conflict
from ca.synthetic import TINY, translate


def solve(F, k, S):
    taps = family_taps(k)
    cl, L, _ = encode(F, taps, k + 1, S)
    with Cadical195(bootstrap_with=cl) as s:
        if not s.solve():
            return None
        luts = luts_from_model(s.get_model(), L, k + 1, len(taps))
    vis, _, _ = rollout(F[0], taps, luts, len(F), S)
    assert (vis == F).all(), "SAT model does not replay exactly -> encoder bug"
    return luts


rng = np.random.default_rng(1)
# 1. memoryless, 1 step: translation
assert solve(translate(16)[:6], 0, 1) is not None
# 2. Conway observed only every 2nd generation (intermediate visible bits unknown)
g0 = (rng.random((8, 8)) < 0.4).astype(np.uint8)
F, _, _ = rollout(g0, family_taps(0), life_like_lut({3}, {2, 3})[None], 5, steps_per_frame=2)
assert solve(F, 0, 2) is not None
# 3. planted random F_priv(1) rule, hidden init 0, 1 step/frame
taps1 = family_taps(1)
for trial in range(3):
    luts = (rng.random((2, 2 ** len(taps1))) < 0.5).astype(np.uint8)
    F, _, _ = rollout((rng.random((8, 8)) < 0.5).astype(np.uint8), taps1, luts, 5, 1)
    assert solve(F, 1, 1) is not None
# 4. proved infeasible
F = TINY["disc12"]()
assert state_repeat_conflict(F) is not None
assert solve(F, 0, 1) is None
# 5. incremental encoding == one cold encoding (same variables, same clauses), and an
#    incremental solve after appending frames still replays exactly
F, _, _ = rollout((rng.random((6, 6)) < 0.5).astype(np.uint8), taps1, luts, 6, 2)
cold, Lc, nvc = encode(F, taps1, 2, 2)
e = Encoder(taps1, 2, 2)
with Cadical195() as s:
    e.add = s.add_clause; e.extend(F[:3]); assert s.solve()
    e.extend(F); assert s.solve()
    inc = luts_from_model(s.get_model(), e.L, 2, len(taps1))
assert e.L == Lc and e.nv == nvc and e.n_cl == len(cold)
e2 = Encoder(taps1, 2, 2); e2.extend(F[:3]); e2.extend(F[:3]); e2.extend(F)
assert e2.clauses == cold, "incremental clauses differ from cold"
assert (rollout(F[0], taps1, inc, len(F), 2)[0] == F).all()
print("ok")
