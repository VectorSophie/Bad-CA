"""Neural CA trainer shared by Campaign 11 experiments and the video pipeline (ca/pipeline.py).

Several models with the SAME architecture (C, Fh, S) are trained together (jax.vmap), each with its own seed, init
and loss weights, for a FIXED number of optimizer steps, so every model gets identical steps and target exposure.
Regime: autonomous rollout from frame 0 of the target, hidden 0, BPTT over the whole interval (per-step remat).
Loss per model (v = continuous visible of frames 1..T-1, y = target):
  L0 = mean (v - y)^2                                            (C9 loss)
  LB = sum band*relu(0.25 - s(v-0.5))^2 / sum band, s = 2y-1      (C10 contour hinge; band = 2-px band of the TRUE contour)
  LT = sum A*(v - clip(t,0,1))^2 / sum A                          (teacher; t = frozen teacher's continuous visible on the
       same interval, A = pixels where the teacher's binary output AGREES with the target -> teacher errors never taught)
  loss = L0 + lamB*LB + lamT*LT
Optimizer: Adam with per-tensor gradient normalisation (as C9/C10); lr schedule 'const' or 'cos' (cosine lr0 -> lr_end).
Every `eval_every` steps: binary pixel error of the full autonomous rollout; best params saved (best_<name>.npz) and a
resumable state (params + Adam moments + step + best) written atomically; progress.json for UIs.
No randomness besides init (numpy default_rng(seed))."""
import json, os, time
import numpy as np
from ca import nca
from ca.diag10 import band


def _atomic_json(path, obj):
    tmp = path + ".tmp"; json.dump(obj, open(tmp, "w"), default=float); os.replace(tmp, path)


def lr_at(sched, i, steps):
    if sched[0] == "const": return sched[1]
    lr0, lr1 = sched[1], sched[2]; return lr1 + 0.5 * (lr0 - lr1) * (1 + np.cos(np.pi * min(i, steps) / steps))


def train(F, cfg, out, C=16, Fh=64, S=4, steps=10000, sched=("cos", 2e-3, 2e-5), teacher_v=None, eval_every=500, resume=True, meta=None, log=print):
    """F: (T,H,W) uint8 target. cfg: [{name, seed, lamB, lamT, init(optional npz path)}]. Returns summary dict."""
    import jax, jax.numpy as jnp
    os.makedirs(out, exist_ok=True); T = len(F); M = len(cfg)
    prog_path = os.path.join(out, "progress.json"); state_path = os.path.join(out, "state.npz")
    _, roll = nca.jax_fns(S, remat=True)
    f0 = jnp.asarray(F[0], jnp.float32); y = jnp.asarray(F[1:], jnp.float32); sg = 2 * y - 1
    bnd = jnp.asarray(band(F[1:]), jnp.float32); nb = jnp.maximum(jnp.sum(bnd), 1.0)
    if teacher_v is not None:
        tv = jnp.asarray(np.clip(teacher_v[1:], 0, 1), jnp.float32); A = jnp.asarray((teacher_v[1:] > 0.5) == F[1:].astype(bool), jnp.float32)
    else:
        tv = jnp.zeros_like(y); A = jnp.zeros_like(y)
    na = jnp.maximum(jnp.sum(A), 1.0)

    def terms(p):
        v = roll(p, f0, T)
        return jnp.mean((v - y) ** 2), jnp.sum(bnd * jax.nn.relu(0.25 - sg * (v - 0.5)) ** 2) / nb, jnp.sum(A * (v - tv) ** 2) / na

    def loss(p, lb, lt):
        L0, LB, LT = terms(p); return L0 + lb * LB + lt * LT, (L0, LB, LT)

    P0 = [dict(np.load(c["init"])) if c.get("init") else nca.init_params(np.random.default_rng(c["seed"]), C, Fh) for c in cfg]
    P = {k: jnp.stack([jnp.asarray(p[k]) for p in P0]) for k in P0[0]}
    lamB = jnp.asarray([c.get("lamB", 0.0) for c in cfg], jnp.float32); lamT = jnp.asarray([c.get("lamT", 0.0) for c in cfg], jnp.float32)
    m = jax.tree_util.tree_map(jnp.zeros_like, P); v2 = jax.tree_util.tree_map(jnp.zeros_like, P)
    step0 = 0; best = [dict(err=2.0, step=-1) for _ in range(M)]; hist = []; train_s = 0.0
    if resume and os.path.exists(state_path):
        z = np.load(state_path, allow_pickle=False); step0 = int(z["step"]); st = json.loads(str(z["meta"]))
        best, hist, train_s = st["best"], st["hist"], st["train_s"]
        P = {k: jnp.asarray(z["P_" + k]) for k in P}; m = {k: jnp.asarray(z["m_" + k]) for k in P}; v2 = {k: jnp.asarray(z["v_" + k]) for k in P}
        log(f"resumed at step {step0}")
    vg = jax.vmap(jax.value_and_grad(loss, has_aux=True)); roll_v = jax.jit(jax.vmap(lambda p: roll(p, f0, T)))
    b1, b2, eps = 0.9, 0.999, 1e-8

    @jax.jit
    def step(P, m, v2, i, lr):
        (l, aux), g = vg(P, lamB, lamT)
        g = jax.tree_util.tree_map(lambda x: x / (jnp.sqrt(jnp.sum(x ** 2, axis=tuple(range(1, x.ndim)), keepdims=True)) + 1e-8), g)
        m0, v20 = m, v2
        m = m1 = jax.tree_util.tree_map(lambda a, b: b1 * a + (1 - b1) * b, m, g); v2 = v21 = jax.tree_util.tree_map(lambda a, b: b2 * a + (1 - b2) * b * b, v2, g)
        Pn = jax.tree_util.tree_map(lambda p, a, b: p - lr * (a / (1 - b1 ** i)) / (jnp.sqrt(b / (1 - b2 ** i)) + eps), P, m, v2)
        ok = jnp.isfinite(l)                     # per model: a non-finite loss/gradient step is skipped (counted in hist)
        keep = lambda new, old: jnp.where(ok.reshape((-1,) + (1,) * (new.ndim - 1)), new, old)
        return jax.tree_util.tree_map(keep, Pn, P), jax.tree_util.tree_map(keep, m1, m0), jax.tree_util.tree_map(keep, v21, v20), l, aux

    Fb = F[1:].astype(bool); skipped = np.zeros(M, int)

    def evaluate(P, i, aux):
        vis = np.asarray(roll_v(P)) > 0.5
        for j in range(M):
            e = float((vis[j] != Fb).mean())
            hist.append(dict(step=i, name=cfg[j]["name"], bin_err=e, L0=float(aux[0][j]), LB=float(aux[1][j]), LT=float(aux[2][j]), nonfinite_samples=int(skipped[j])))
            if e < best[j]["err"]:
                best[j] = dict(err=e, step=i)
                np.savez(os.path.join(out, f"best_{cfg[j]['name']}.npz"), **{k: np.asarray(P[k][j]) for k in P})
        tmp = state_path + ".tmp.npz"
        np.savez(tmp, step=i, meta=json.dumps(dict(best=best, hist=hist, train_s=train_s)), **{f"P_{k}": np.asarray(P[k]) for k in P},
                 **{f"m_{k}": np.asarray(m[k]) for k in P}, **{f"v_{k}": np.asarray(v2[k]) for k in P})
        os.replace(tmp, state_path)

    t0 = time.perf_counter(); i0 = float(step0 + 1)
    P1, m1, v21, l, aux = step(P, m, v2, i0, lr_at(sched, step0, steps)); jax.block_until_ready(l); compile_s = time.perf_counter() - t0
    if step0 == 0: evaluate(P, 0, jax.jit(jax.vmap(terms))(P))   # the initial weights are a candidate too
    P, m, v2 = P1, m1, v21; i = step0 + 1; tw = time.perf_counter()
    _atomic_json(prog_path, dict(status="running", step=i, steps=steps, train_s=train_s, compile_s=compile_s, best=best, names=[c["name"] for c in cfg]))
    while i < steps:
        P, m, v2, l, aux = step(P, m, v2, float(i + 1), lr_at(sched, i, steps)); i += 1
        if i % 50 == 0: skipped += ~np.isfinite(np.asarray(l))   # sampled every 50 steps (a host sync per step would slow training)
        if i % eval_every == 0 or i == steps:
            jax.block_until_ready(l); train_s += time.perf_counter() - tw
            evaluate(P, i, aux)
            _atomic_json(prog_path, dict(status="running" if i < steps else "finished", step=i, steps=steps, train_s=round(train_s, 1), compile_s=round(compile_s, 1),
                                         best=best, last=[h for h in hist[-M:]], names=[c["name"] for c in cfg]))
            log(f"step {i}/{steps} " + " ".join(f"{h['name']}={h['bin_err']:.4f}" for h in hist[-M:]))
            tw = time.perf_counter()
    for j in range(M): np.savez(os.path.join(out, f"last_{cfg[j]['name']}.npz"), **{k: np.asarray(P[k][j]) for k in P})
    res = dict(T=T, H=int(F.shape[1]), W=int(F.shape[2]), C=C, Fh=Fh, S=S, n_params=nca.n_params(C, Fh), steps=steps, sched=list(sched), models=cfg,
               compile_s=round(compile_s, 1), train_s=round(train_s, 1), device=str(jax.devices()[0]), best={c["name"]: b for c, b in zip(cfg, best)}, meta=meta or {})
    _atomic_json(os.path.join(out, "train.json"), res); json.dump(hist, open(os.path.join(out, "hist.json"), "w"))
    return res
