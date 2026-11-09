"""Exp33 (Campaign 09): train the Neural CA (ca/nca.py) on GPU (gpuq). 3 seeds vmapped.
REGIME B: target = window frames, rollout from the true window frame 0 with hidden 0.
REGIME C: target = clip frames 0..t0+15, rollout from the clip's first frame with hidden 0.
TARGET 'control': the known_mover NCA's own 30-frame video (24x80, 3 objects), regime C.
Loss: L2 on continuous visible at every observed frame (BPTT, per-step checkpointing). Adam lr 2e-3 with per-tensor
gradient normalisation. Every 30 s each seed is rolled out (continuous, same as deployment) and scored by BINARY pixel
error over the regime's frames; the best checkpoint per seed is saved immediately (npz + RNG/config).
usage: exp33_nca.py TASK REGIME BUDGET_S OUTDIR   (TASK in main|confirm|control)
"""
import json, os, sys, time
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax, jax.numpy as jnp
from ca import nca
from c7_common import target

TASKS = {"main": ("ba48_4500_60", 40), "confirm": ("ba48_1800_60", 44)}
task, regime, budget, outdir = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
C, Fh, S, SEEDS, LR = 16, 64, 4, 3, 2e-3
TEACHER = None   # reduction (distillation): --small C Fh --teacher EXPORT -> train a smaller NCA on the teacher's OWN binary replay
if "--small" in sys.argv:
    i = sys.argv.index("--small"); C, Fh = int(sys.argv[i + 1]), int(sys.argv[i + 2]); TEACHER = sys.argv[sys.argv.index("--teacher") + 1]
if task == "control":
    f0 = np.zeros((24, 80), np.uint8); f0[3:9, 2:8] = 1; f0[11:14, 1:10] = 1; f0[16:22, 3:6] = 1
    F, _ = nca.rollout_np(nca.known_mover(C, Fh), f0[None].repeat(30, 0), 30, S); Fr = F
else:
    clip, t0 = TASKS[task]; full = target(clip)
    Fr = full[t0:t0 + 16] if regime == "B" else full[:t0 + 16]
if TEACHER:
    _, Pt, _ = nca.load(TEACHER); Fr, _ = nca.rollout_np(Pt, Fr, len(Fr), S)   # teacher's own frames replace the original as targets
T = len(Fr); OUT = os.path.join(ROOT, "runs", outdir, f"nca_{task}_{regime}" + (f"_small{C}x{Fh}" if TEACHER else "")); os.makedirs(OUT, exist_ok=True)
_, roll = nca.jax_fns(S, remat=True)
f0j = jnp.asarray(Fr[0], jnp.float32); tgt = jnp.asarray(Fr[1:], jnp.float32)
P = [nca.init_params(np.random.default_rng(seed), C, Fh) for seed in range(SEEDS)]
P = {k: jnp.stack([p[k] for p in P]) for k in P[0]}


def loss(p):
    return jnp.mean((roll(p, f0j, T) - tgt) ** 2)


vg = jax.vmap(jax.value_and_grad(loss)); roll_v = jax.jit(jax.vmap(lambda p: roll(p, f0j, T)))
b1, b2, eps = 0.9, 0.999, 1e-8
m = jax.tree_util.tree_map(jnp.zeros_like, P); v = jax.tree_util.tree_map(jnp.zeros_like, P)


@jax.jit
def step(P, m, v, i):
    l, g = vg(P)
    g = jax.tree_util.tree_map(lambda x: x / (jnp.sqrt(jnp.sum(x ** 2, axis=tuple(range(1, x.ndim)), keepdims=True)) + 1e-8), g)  # per-seed per-tensor norm
    m = jax.tree_util.tree_map(lambda a, b: b1 * a + (1 - b1) * b, m, g); v = jax.tree_util.tree_map(lambda a, b: b2 * a + (1 - b2) * b * b, v, g)
    P = jax.tree_util.tree_map(lambda p, a, b: p - LR * (a / (1 - b1 ** i)) / (jnp.sqrt(b / (1 - b2 ** i)) + eps), P, m, v)
    return P, m, v, l


t0w = time.perf_counter(); P, m, v, l = step(P, m, v, 1.0); jax.block_until_ready(l); compile_s = time.perf_counter() - t0w
best = [dict(err=2.0) for _ in range(SEEDS)]; log = []; it = 1; t0w = time.perf_counter(); nxt = 0.0
while True:
    el = time.perf_counter() - t0w
    if el >= nxt or el >= budget:
        vis = np.asarray(roll_v(P)) > 0.5
        for s in range(SEEDS):
            e = float((vis[s] != Fr[1:].astype(bool)).mean()); log.append(dict(t=round(el, 1), it=it, seed=s, loss=float(np.asarray(l)[s]), bin_err=e))
            if e < best[s]["err"]:
                best[s] = dict(err=e, it=it, t=round(el, 1))
                np.savez(os.path.join(OUT, f"best_seed{s}.npz"), **{k: np.asarray(P[k][s]) for k in P})
                json.dump(dict(best[s], task=task, regime=regime, seed=s, C=C, Fh=Fh, S=S, lr=LR, init="numpy default_rng(seed)", teacher=TEACHER), open(os.path.join(OUT, f"best_seed{s}.json"), "w"))
        nxt = el + 30.0
        if el >= budget: break
    it += 1; P, m, v, l = step(P, m, v, float(it))
res = dict(task=task, regime=regime, teacher=TEACHER, T=T, C=C, Fh=Fh, S=S, n_params=nca.n_params(C, Fh), seeds=SEEDS, budget_s=budget, compile_s=round(compile_s, 1),
           train_s=round(time.perf_counter() - t0w, 1), iters=it, device=str(jax.devices()[0]), best=best)
json.dump(log, open(os.path.join(OUT, "log.json"), "w")); json.dump(res, open(os.path.join(OUT, "train.json"), "w"), indent=1)
print(json.dumps(res))
