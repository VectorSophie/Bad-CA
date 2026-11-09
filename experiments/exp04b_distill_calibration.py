"""Exp04b: implementation calibration. Supervised hard accuracy of ca.difflogic (reference
architecture) on (a) Conway's full 512-entry LUT -- the reference paper learns it exactly, so
failing here means OUR implementation is wrong, not the CA problem; (b) a random partial
function with 10 inputs / 400 care entries (size of the SAT references).
v1 of this script (shallow random-wired net) gave Conway <= 83%: runs/exp04/distill_calibration_v1.json.
GPU via gpuq."""
import json, os, sys, time
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
import jax, jax.numpy as jnp
from ca.core import life_like_lut
from ca.difflogic import adamw, hard_lut, init_logits, make_wiring, net

SEEDS, STEPS = 8, int(sys.argv[1]) if len(sys.argv) > 1 else 3000
rng = np.random.default_rng(0)
tasks = {"conway": (9, np.arange(512), life_like_lut({3}, {2, 3})[None]),
         "random400": (10, np.sort(rng.choice(1024, 400, replace=False)), (rng.random((1, 1024)) < 0.5).astype(np.uint8))}
out = []
for tname, (n, care, lut) in tasks.items():
    X = jnp.asarray(((care[:, None] >> np.arange(n)) & 1).astype(np.float32)); Y = jnp.asarray(lut[:, care].T.astype(np.float32))
    for depth in (16,):
        for lr in (0.05,):
            wir = [make_wiring(np.random.default_rng(1000 + s), n, 1, 16, (128,) * depth + (128, 64, 32, 16, 8, 4)) for s in range(SEEDS)]
            W = [jnp.asarray(np.stack([w[l] for w in wir])) for l in range(len(wir[0]))]
            P = [jnp.stack([l] * SEEDS) for l in init_logits(wir[0])]
            init, upd = adamw(lr); st = jax.vmap(init)(P)
            vg = jax.vmap(jax.value_and_grad(lambda P, W: jnp.sum((net(P, W, X) - Y) ** 2)))
            @jax.jit
            def step(P, st):
                l, g = vg(P, W); P, st = jax.vmap(upd)(g, st, P); return P, st, l
            t0 = time.perf_counter(); traj = []
            for t in range(1, STEPS + 1):
                P, st, l = step(P, st)
                if t % 500 == 0 or t == STEPS:
                    accs = [float((hard_lut([p[s] for p in P], [np.asarray(w[s]) for w in W], n)[:, care] == lut[:, care]).mean()) for s in range(SEEDS)]
                    traj.append((t, max(accs), sum(a == 1 for a in accs)))
            r = dict(task=tname, depth=depth, lr=lr, best_hard_acc=traj[-1][1], seeds_exact=traj[-1][2], trajectory=traj,
                     final_soft_loss=[round(float(x), 3) for x in np.asarray(l)], seconds=round(time.perf_counter() - t0, 1))
            out.append(r); print(json.dumps(r), flush=True)
json.dump(out, open(os.path.join(ROOT, "runs", "exp04", "distill_calibration_v2.json"), "w"), indent=1)
