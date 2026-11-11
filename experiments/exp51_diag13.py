"""Exp51 (Campaign 13): numerical-stability diagnosis of an ncab-v2 export. Replays it from the file with
(a) the numpy reference (CPU, = the export replay used for every quality number) recording max |state| per frame, and
(b) the JAX path on the default device (GPU in a gpuq job = the training arithmetic), and compares them per frame:
binary mismatching pixels and max |visible difference|. Also lists 16-frame windows with error > 0.09 (C12 gate) against
the original. usage: exp51_diag13.py OUT_JSON EXPORT [EXPORT ...]"""
import json, os, sys, time
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax, jax.numpy as jnp
from ca import nca, phase_nca
from exp45_c12 import target

res = {}
for path in sys.argv[2:]:
    hd, E, sched, h0, f0 = phase_nca.load(path); T = hd["T"]; C = hd["C"]
    s = np.zeros((C,) + f0.shape, np.float32); s[0] = f0
    if h0 is not None: s[1:] = h0
    Vn, smax = [s[0].copy()], [float(np.abs(s).max())]; t0 = time.perf_counter()
    for t in range(1, T):
        p = {k: E[k][sched[t]] for k in phase_nca.PNAMES}
        for _ in range(hd["S"]): s = nca.step_np(p, s)
        Vn.append(s[0].copy()); smax.append(float(np.abs(s).max()))
    Vn = np.stack(Vn); np_s = time.perf_counter() - t0
    roll = phase_nca.jax_rollout(hd["S"], remat=False); t0 = time.perf_counter()
    v, _ = jax.jit(roll)({k: jnp.asarray(x) for k, x in E.items()}, jnp.asarray(f0, jnp.float32), jnp.asarray(sched[1:]), None if h0 is None else jnp.asarray(h0))
    Vj = np.concatenate([Vn[:1], np.asarray(v)]); jx_s = time.perf_counter() - t0
    Bn, Bj = Vn > 0.5, Vj > 0.5; Bn[0] = Bj[0] = f0.astype(bool)
    mism = (Bn != Bj).sum((1, 2)); dv = np.abs(Vn - Vj).reshape(T, -1).max(1)
    F = target(f"{hd['meta']['src_start']}:{T}").astype(bool); e = (Bn != F).mean((1, 2))
    w = np.array([e[a:a + 16].mean() for a in range(1, T - 15)]); bad = np.nonzero(w > 0.09)[0] + 1
    runs = []
    for a in bad:
        if runs and a <= runs[-1][1] + 1: runs[-1][1] = int(a)
        else: runs.append([int(a), int(a)])
    first = lambda c: int(np.argmax(c)) if c.any() else None
    res[os.path.relpath(path, ROOT)] = dict(T=T, device=str(jax.devices()[0]), numpy_s=round(np_s, 1), jax_s=round(jx_s, 1),
        first_binary_mismatch_frame=first(mism > 0), first_frame_mismatch_over_1pct=first(mism > 0.01 * f0.size), mismatch_px_per_frame=mism.tolist(),
        max_dv_per_frame=[float(x) for x in dv], max_abs_state_per_frame=smax, state_max_overall=float(max(smax)),
        bad16_windows=[dict(start=a, end=b + 15, peak=float(w[a - 1:b].max())) for a, b in runs])
    r = res[os.path.relpath(path, ROOT)]
    print(os.path.basename(os.path.dirname(path)) or path, "T", T, r["device"], "first mismatch", r["first_binary_mismatch_frame"], ">1%", r["first_frame_mismatch_over_1pct"],
          "state max", f"{r['state_max_overall']:.3g}", "bad16 windows", [(x["start"], x["end"], round(x["peak"], 3)) for x in r["bad16_windows"]][:12], flush=True)
json.dump(res, open(sys.argv[1], "w"))
