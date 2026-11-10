"""Exp37 (Campaign 10): continue training the Campaign 9 C-trained NCAs (10,320 params, C=16, Fh=64, S=4, zero pad,
hidden 0, same preprocessing) under three arms, all models of one task vmapped in ONE gpuq job, so every model gets
exactly the same number of optimizer steps and wall time.

Regime C as in Campaign 9: target = clip frames 0..t0+15, autonomous from clip frame 0 with hidden 0 (BPTT over all).
v_t = continuous visible at frame t (t = 1..T-1), y_t = original binary frame, Y = y with frame 0 prepended.
  L0 = mean_{t,px} (v - y)^2                                                   (Campaign 9 loss; arm A uses only this)
  LB = sum_{t,px} band_t * relu(m - s*(v - 0.5))^2 / sum band_t,  s = 2y - 1, m = 0.25
       band_t = pixels whose 3x3 (edge-replicated) neighbourhood in the ORIGINAL y_t holds both values (2 px each side
       of the true contour). Pushes each contour-band pixel to the correct side of 0.5 with margin; thin parts and true
       gaps lie entirely in the band and are kept, nothing is smoothed.
  LC = 1/2 sum_{k in 1,4} sum_{t,px} (D_k u - D_k Y)^2 / sum_{t,px} dil3x3(D_k Y != 0),  u = clip(v,0,1), u_0 = y_0,
       D_k X_t = X_t - X_{t-k}. Matches the original's frame-to-frame (k=1) and accumulated (k=4) on/off changes; a
       replay that stalls or delays a change is penalised (D u = 0 where D Y != 0); unlike temporal smoothing the
       target is the original's change, not zero.
  loss = L0 + lamB*LB + lamC*LC (per model).
Optimizer: Adam lr 2e-3, per-tensor gradient normalisation (as Campaign 9). Campaign 9 saved no optimizer state, so
EVERY arm starts with fresh Adam moments. No randomness is used (deterministic BPTT); start params = C9 best_seed{s}.
Selection (same rule as C9): every 30 s each model's autonomous rollout is scored by BINARY pixel error over the
regime; the starting checkpoint is scored first (it=0), so 'best' can stay the C9 model. best_* and last_* saved.
usage: exp37_ft10.py TASK BUDGET_S OUTDIR CONFIG_JSON [--lr LR]   (config: [{"name","seed","lamB","lamC"}, ...])"""
import json, os, sys, time
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax, jax.numpy as jnp
from ca import nca
from ca.diag10 import band
from ca.motion import _dil
from c7_common import target

TASKS = {"main": ("ba48_4500_60", 40), "confirm": ("ba48_1800_60", 44)}
task, budget, outdir, cfg_path = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4]
S, MARGIN = 4, 0.25
LR = float(sys.argv[sys.argv.index("--lr") + 1]) if "--lr" in sys.argv else 2e-3   # pilot 1 (2e-3) knocked every model off its C9 optimum
cfg = json.load(open(cfg_path)); M = len(cfg)
clip, t0 = TASKS[task]; Fr = target(clip)[:t0 + 16]; T = len(Fr)
OUT = os.path.join(ROOT, "runs", outdir); os.makedirs(OUT, exist_ok=True); json.dump(cfg, open(os.path.join(OUT, "config.json"), "w"), indent=1)
_, roll = nca.jax_fns(S, remat=True)
f0j = jnp.asarray(Fr[0], jnp.float32); Yj = jnp.asarray(Fr, jnp.float32); y = Yj[1:]
bandj = jnp.asarray(band(Fr[1:]), jnp.float32); sg = 2 * y - 1
DY = {k: Yj[k:] - Yj[:-k] for k in (1, 4)}; NDY = {k: float(_dil(Fr[k:] != Fr[:-k]).sum()) for k in (1, 4)}


def terms(p):
    v = roll(p, f0j, T); L0 = jnp.mean((v - y) ** 2)
    LB = jnp.sum(bandj * jax.nn.relu(MARGIN - sg * (v - 0.5)) ** 2) / jnp.sum(bandj)
    u = jnp.concatenate([Yj[:1], jnp.clip(v, 0, 1)])
    LC = 0.5 * sum(jnp.sum((u[k:] - u[:-k] - DY[k]) ** 2) / NDY[k] for k in (1, 4))
    return L0, LB, LC


def loss(p, lb, lc):
    L0, LB, LC = terms(p); return L0 + lb * LB + lc * LC, (L0, LB, LC)


P0 = [dict(np.load(os.path.join(ROOT, "runs", "c9", "nca", f"nca_{task}_C", f"best_seed{c['seed']}.npz"))) for c in cfg]
P = {k: jnp.stack([jnp.asarray(p[k]) for p in P0]) for k in P0[0]}
lamB = jnp.asarray([c["lamB"] for c in cfg], jnp.float32); lamC = jnp.asarray([c["lamC"] for c in cfg], jnp.float32)
vg = jax.vmap(jax.value_and_grad(loss, has_aux=True)); roll_v = jax.jit(jax.vmap(lambda p: roll(p, f0j, T)))
b1, b2, eps = 0.9, 0.999, 1e-8
m = jax.tree_util.tree_map(jnp.zeros_like, P); v2 = jax.tree_util.tree_map(jnp.zeros_like, P)


@jax.jit
def step(P, m, v2, i):
    (l, aux), g = vg(P, lamB, lamC)
    g = jax.tree_util.tree_map(lambda x: x / (jnp.sqrt(jnp.sum(x ** 2, axis=tuple(range(1, x.ndim)), keepdims=True)) + 1e-8), g)
    m = jax.tree_util.tree_map(lambda a, b: b1 * a + (1 - b1) * b, m, g); v2 = jax.tree_util.tree_map(lambda a, b: b2 * a + (1 - b2) * b * b, v2, g)
    P = jax.tree_util.tree_map(lambda p, a, b: p - LR * (a / (1 - b1 ** i)) / (jnp.sqrt(b / (1 - b2 ** i)) + eps), P, m, v2)
    return P, m, v2, l, aux


def save(prefix, j, P, meta):
    np.savez(os.path.join(OUT, f"{prefix}_{cfg[j]['name']}.npz"), **{k: np.asarray(P[k][j]) for k in P})
    json.dump(dict(meta, **cfg[j], task=task, C=16, Fh=64, S=S, lr=LR, margin=MARGIN, start=f"runs/c9/nca/nca_{task}_C/best_seed{cfg[j]['seed']}.npz",
                   adam="fresh moments (C9 saved none)", rng="none used"), open(os.path.join(OUT, f"{prefix}_{cfg[j]['name']}.json"), "w"))


Fb = Fr[1:].astype(bool); best = [dict(err=2.0) for _ in range(M)]; log = []
init_terms = jax.jit(jax.vmap(terms))(P)


def score(P, it, el, l, aux):
    vis = np.asarray(roll_v(P)) > 0.5
    for j in range(M):
        e = float((vis[j] != Fb).mean())
        log.append(dict(t=round(el, 1), it=it, name=cfg[j]["name"], loss=None if l is None else float(l[j]), L0=float(aux[0][j]), LB=float(aux[1][j]), LC=float(aux[2][j]), bin_err=e))
        if e < best[j]["err"]:
            best[j] = dict(err=e, it=it, t=round(el, 1)); save("best", j, P, best[j])


score(P, 0, 0.0, None, init_terms)                                            # the unchanged C9 checkpoint
t0w = time.perf_counter(); P, m, v2, l, aux = step(P, m, v2, 1.0); jax.block_until_ready(l); compile_s = time.perf_counter() - t0w
it = 1; t0w = time.perf_counter(); nxt = 30.0
while True:
    el = time.perf_counter() - t0w
    if el >= nxt or el >= budget:
        score(P, it, el, l, aux); nxt = el + 30.0
        json.dump(log, open(os.path.join(OUT, "log.json"), "w"))
        if el >= budget: break
    it += 1; P, m, v2, l, aux = step(P, m, v2, float(it))
for j in range(M): save("last", j, P, dict(it=it, err=log[-M + j]["bin_err"]))
res = dict(task=task, T=T, models=M, budget_s=budget, compile_s=round(compile_s, 1), train_s=round(time.perf_counter() - t0w, 1), steps=it,
           device=str(jax.devices()[0]), best={c["name"]: b for c, b in zip(cfg, best)})
json.dump(log, open(os.path.join(OUT, "log.json"), "w")); json.dump(res, open(os.path.join(OUT, "train.json"), "w"), indent=1)
print(json.dumps(res))
