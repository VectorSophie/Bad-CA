"""Exp08 (D1): discrete refinement of a hardened circuit by gate-op hill-climbing. CPU only.

Start: live sub-circuit extracted from a run's best_params.pkl (exp07). Move: replace one gate's
truth table (16 options), wiring fixed. Accept if hard autonomous rollout error does not
increase (plateau moves allowed). Fixed RNG seed 0, wall budget. Video-only: uses frames only.
usage: exp08_discrete_refine.py PARAMS_PKL OUTDIR [--budget_s 600]
"""
import json, os, pickle, subprocess, sys, time
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax.numpy as jnp
from ca.difflogic import extract_circuit
from ca.rule import family_taps, rollout
from ca.core import metrics
from ca.export import export_rule
from ca import costs
from exp03_sat_ablation import target

pkl, outdir = sys.argv[1], sys.argv[2]
budget = float(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[3] == "--budget_s" else 600.0
TNAME, K, S = "ba32_3000_5", 1, 2
taps = family_taps(K); n = len(taps); F = target(TNAME); T = len(F)
OUT = os.path.join(ROOT, "runs", outdir); os.makedirs(OUT, exist_ok=True)
d = pickle.load(open(pkl, "rb"))
gates, outs = extract_circuit([jnp.asarray(p) for p in d["Ps"]], d["Ws"], n)
gates = [list(g) for g in gates]


def err(gs):
    L = costs.eval_circuit(gs, outs, n); vis, _, _ = rollout(F[0], taps, L, T, S)
    return int((vis != F).sum()), L, vis


rng = np.random.default_rng(0)
t0 = time.perf_counter(); cur, L, vis = err(gates); trace = [dict(t=0.0, wrong_pixels=cur)]; evals = 1
start = cur
while cur > 0 and time.perf_counter() - t0 < budget and gates:
    j = int(rng.integers(len(gates))); old = gates[j][0]; new = int(rng.integers(16))
    if new == old: continue
    gates[j][0] = new; e, L2, v2 = err(gates); evals += 1
    if e <= cur:
        if e < cur: trace.append(dict(t=round(time.perf_counter() - t0, 1), wrong_pixels=e))
        cur, L, vis = e, L2, v2
    else:
        gates[j][0] = old
wall = time.perf_counter() - t0
cw = costs.rule_circuit([tuple(g) for g in gates], outs, n, "F_priv", K, S)
dc = costs.rule_circuit_decode(cw.b)
assert (costs.eval_circuit(dc["gates"], dc["outputs"], n) == L).all()
path = os.path.join(OUT, "best.rule.json")
ex = export_rule(path, taps, L, F[0], T, S, meta=dict(method="D1 gate-op hill-climb", start=pkl), expected=vis)
node = json.loads(subprocess.run(["node", os.path.join(ROOT, "player", "replay.mjs"), path], capture_output=True, text=True, check=True).stdout)
res = dict(method="D1", track="video only", start=pkl, code_snapshot=os.path.basename(os.path.realpath(ROOT)), budget_s=budget,
           wall_s=round(wall, 1), evals=evals, rng_seed=0, start_wrong_pixels=start, wrong_pixels=cur, exact=cur == 0,
           hard_pixel_err=metrics(vis, F)["pixel_err"], change_iou=metrics(vis, F)["change_iou"], live_gates=len(gates),
           circuit_bits=len(cw), lut_exception_bits=len(costs.rule_lut_exceptions(L, taps, "F_priv", K, S)),
           own_output_recording_bits=costs.recording_costs(vis), node_replay_matches_python=node["sha256"] == ex["expected_sha256"],
           trace=trace)
np.save(os.path.join(OUT, "best_luts.npy"), L)
json.dump(res, open(os.path.join(OUT, "result.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "trace"}))
