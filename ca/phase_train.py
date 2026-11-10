"""Training for phased NCAs (ca/phase_nca.py). Loss per model: L0 + lamB*LB (C10 contour hinge) over the frames trained.
Adam + per-tensor gradient normalisation, cosine lr, fixed steps, non-finite step guard, binary-error checkpoint
selection every `eval_every` steps on the trained span (the initial weights are a candidate), resumable state.

joint(...)  : all phases trained together with full BPTT over the whole interval from frame 0 (gradient crosses
              boundaries, so earlier phases can learn to prepare hidden state for later ones). M seeds vmapped.
greedy(...) : phases trained one after another. Phase p starts from the ACTUAL carried state produced by the
              already-trained phases 0..p-1 (autonomous, from frame 0, best checkpoints) and its rule is initialised
              from phase p-1's rule (hidden-meaning continuity); BPTT only inside phase p (truncation at the boundary,
              state values carried, no gradient into earlier phases). Phase 0 may also learn the low-res initial hidden.
              Final evaluation is always a full rollout from frame 0 (done by the evaluator, from the export)."""
import json, os, time
import numpy as np
from ca import nca
from ca.diag10 import band
from ca.phase_nca import PNAMES, effective, init_params, jax_rollout, upsample


def _atomic(path, obj):
    tmp = path + ".tmp"; json.dump(obj, open(tmp, "w"), default=float); os.replace(tmp, path)


def _lr(sched, i, steps):
    return sched[1] if sched[0] == "const" else sched[2] + 0.5 * (sched[1] - sched[2]) * (1 + np.cos(np.pi * min(i, steps) / steps))


def _fit(jax, jnp, P, loss_v, roll_bin, steps, sched, eval_every, out, tag, log, names, lamB_unused=None, resume=True):
    """Generic vmapped Adam loop. P: pytree with leading model axis M. loss_v(P)->(l[M], aux[M,..]); roll_bin(P)->binary err [M]."""
    M = len(names); sp = os.path.join(out, f"state_{tag}.npz"); m = jax.tree_util.tree_map(jnp.zeros_like, P); v = jax.tree_util.tree_map(jnp.zeros_like, P)
    best = [dict(err=2.0, step=-1) for _ in range(M)]; hist = []; i0 = 0; train_s = 0.0; skipped = np.zeros(M, int)
    if resume and os.path.exists(sp):
        z = np.load(sp); st = json.loads(str(z["meta"])); i0, best, hist, train_s = st["step"], st["best"], st["hist"], st["train_s"]
        P = {k: jnp.asarray(z["P_" + k]) for k in P}; m = {k: jnp.asarray(z["m_" + k]) for k in P}; v = {k: jnp.asarray(z["v_" + k]) for k in P}; log(f"{tag}: resumed at {i0}")
    b1, b2 = 0.9, 0.999

    @jax.jit
    def step(P, m, v, i, lr):
        (l, aux), g = loss_v(P, i)
        g = jax.tree_util.tree_map(lambda x: x / (jnp.sqrt(jnp.sum(x ** 2, axis=tuple(range(1, x.ndim)), keepdims=True)) + 1e-8), g)
        m1 = jax.tree_util.tree_map(lambda a, b: b1 * a + (1 - b1) * b, m, g); v1 = jax.tree_util.tree_map(lambda a, b: b2 * a + (1 - b2) * b * b, v, g)
        Pn = jax.tree_util.tree_map(lambda p, a, b: p - lr * (a / (1 - b1 ** i)) / (jnp.sqrt(b / (1 - b2 ** i)) + 1e-8), P, m1, v1)
        ok = jnp.isfinite(l); keep = lambda n, o: jnp.where(ok.reshape((-1,) + (1,) * (n.ndim - 1)), n, o)
        return jax.tree_util.tree_map(keep, Pn, P), jax.tree_util.tree_map(keep, m1, m), jax.tree_util.tree_map(keep, v1, v), l, aux

    def evaluate(P, i):
        e = np.asarray(roll_bin(P))
        for j in range(M):
            hist.append(dict(step=i, name=names[j], bin_err=float(e[j]), nonfinite_samples=int(skipped[j])))
            if np.isfinite(e[j]) and e[j] < best[j]["err"]:
                best[j] = dict(err=float(e[j]), step=i); np.savez(os.path.join(out, f"best_{tag}_{names[j]}.npz"), **{k: np.asarray(P[k][j]) for k in P})
        tmp = sp + ".tmp.npz"
        np.savez(tmp, meta=json.dumps(dict(step=i, best=best, hist=hist, train_s=train_s)), **{f"P_{k}": np.asarray(P[k]) for k in P}, **{f"m_{k}": np.asarray(m[k]) for k in P}, **{f"v_{k}": np.asarray(v[k]) for k in P})
        os.replace(tmp, sp)

    t0 = time.perf_counter(); out1 = step(P, m, v, float(i0 + 1), _lr(sched, i0, steps)); jax.block_until_ready(out1[3]); compile_s = time.perf_counter() - t0
    if i0 == 0: evaluate(P, 0)
    P, m, v = out1[:3]; i = i0 + 1; tw = time.perf_counter()
    while i < steps:
        P, m, v, l, aux = step(P, m, v, float(i + 1), _lr(sched, i, steps)); i += 1
        if i % 50 == 0: skipped += ~np.isfinite(np.asarray(l))
        if i % eval_every == 0 or i == steps:
            jax.block_until_ready(l); train_s += time.perf_counter() - tw; evaluate(P, i)
            _atomic(os.path.join(out, "progress.json"), dict(status="running", tag=tag, step=i, steps=steps, train_s=round(train_s, 1), best=best, names=names))
            log(f"{tag} step {i}/{steps} " + " ".join(f"{h['name']}={h['bin_err']:.4f}" for h in hist[-M:])); tw = time.perf_counter()
    return P, dict(best=best, hist=hist, train_s=round(train_s, 1), compile_s=round(compile_s, 1), steps=steps)


def _targets(jnp, F):
    y = jnp.asarray(F[1:], jnp.float32); return y, 2 * y - 1, jnp.asarray(band(F[1:]), jnp.float32)


def joint(F, phases, mode, out, seeds=(0, 1), C=16, Fh=64, rank=4, S=4, steps=15000, sched=("cos", 5e-4, 5e-6), lamB=2.0, init_factor=None, eval_every=500, log=print):
    import jax, jax.numpy as jnp
    os.makedirs(out, exist_ok=True); T, H, W = F.shape; K = len(phases)
    ph = np.zeros(T, np.int32)
    for k, p in enumerate(phases): ph[p["start"]:p["end"]] = k
    roll = jax_rollout(S); y, sg, bd = _targets(jnp, F); f0 = jnp.asarray(F[0], jnp.float32); phj = jnp.asarray(ph[1:]); nb = jnp.maximum(bd.sum(), 1)
    P0 = []
    for s in seeds:
        rng = np.random.default_rng(s); p = init_params(rng, mode, K, C, Fh, rank)
        if init_factor: p["init_g"] = np.zeros((C - 1, -(-H // init_factor), -(-W // init_factor)), np.float32)
        P0.append(p)
    P = {k: jnp.stack([jnp.asarray(p[k]) for p in P0]) for k in P0[0]}

    def run(p):
        E = effective({k: v for k, v in p.items() if k != "init_g"}, mode, jnp)
        h0 = upsample(p["init_g"], init_factor, H, W, jnp) if init_factor else None
        return roll(E, f0, phj, h0)[0]

    def loss(p, i=None):
        v = run(p); L0 = jnp.mean((v - y) ** 2); LB = jnp.sum(bd * jax.nn.relu(0.25 - sg * (v - 0.5)) ** 2) / nb; return L0 + lamB * LB, (L0, LB)

    Fb = jnp.asarray(F[1:].astype(bool))
    roll_bin = jax.jit(jax.vmap(lambda p: jnp.mean((run(p) > 0.5) != Fb)))
    names = [f"s{s}" for s in seeds]
    P, res = _fit(jax, jnp, P, jax.vmap(jax.value_and_grad(loss, has_aux=True), in_axes=(0, None)), roll_bin, steps, sched, eval_every, out, "joint", log, names)
    res.update(T=T, mode=mode, K=K, C=C, Fh=Fh, rank=rank, S=S, phases=phases, init_factor=init_factor, n_params_per_seed=int(sum(np.prod(v.shape[1:]) for v in P.values())), device=str(jax.devices()[0]))
    _atomic(os.path.join(out, "train.json"), res); return res


def greedy(F, phases, out, seed=0, C=16, Fh=64, S=4, steps_per_phase=4000, sched=("cos", 5e-4, 5e-6), lamB=2.0, init_factor=None, eval_every=500, log=print, warm=True):
    """mode 'indep' only. Writes best_p{k}.npz per phase (+ init_g with phase 0) and greedy.json."""
    import jax, jax.numpy as jnp
    os.makedirs(out, exist_ok=True); T, H, W = F.shape; roll = jax_rollout(S); done = []
    rec_path = os.path.join(out, "greedy.json"); rec = json.load(open(rec_path)) if os.path.exists(rec_path) else dict(phases=[], seed=seed)
    state = None; prev = None; g_best = None
    for k, p in enumerate(phases):
        a, b = p["start"], p["end"]; Fk = F[a - 1:b]                     # frame a-1 is the carried 'previous' frame
        ck = os.path.join(out, f"best_p{k}_s{seed}.npz")
        if k < len(rec["phases"]) and os.path.exists(ck):                  # resume: phase already finished
            prev = dict(np.load(ck))
            if k == 0 and init_factor: g_best = prev.pop("init_g")
        else:
            if prev is None: p0 = init_params(np.random.default_rng(seed), "fixed", 1, C, Fh)
            else: p0 = {kk: np.asarray(v)[None] for kk, v in prev.items() if kk in PNAMES} if warm else init_params(np.random.default_rng(seed + 1000 * k), "fixed", 1, C, Fh)
            p0 = {kk: np.asarray(v)[0] for kk, v in p0.items()}
            if k == 0 and init_factor: p0["init_g"] = np.zeros((C - 1, -(-H // init_factor), -(-W // init_factor)), np.float32)
            P = {kk: jnp.asarray(v)[None] for kk, v in p0.items()}
            y, sg, bd = _targets(jnp, Fk); nb = jnp.maximum(bd.sum(), 1); phj = jnp.zeros(len(Fk) - 1, jnp.int32)
            s_in = None if state is None else jnp.asarray(state)

            def run(pp, s_in=s_in):
                E = {n: pp[n][None] for n in PNAMES}
                if s_in is None:
                    f0 = jnp.asarray(Fk[0], jnp.float32); h0 = upsample(pp["init_g"], init_factor, H, W, jnp) if "init_g" in pp else None
                    return roll(E, f0, phj, h0)
                C_ = s_in.shape[0]
                def frame(s, _):
                    def sub(s, _):
                        q = jax.nn.relu(jax.lax.conv_general_dilated(s[None], E["Wp"][0], (1, 1), "SAME", precision=jax.lax.Precision.HIGHEST)[0] + E["bp"][0][:, None, None])
                        return s + jnp.einsum("cf,fhw->chw", E["Wu"][0], q, precision=jax.lax.Precision.HIGHEST) + E["bu"][0][:, None, None], None
                    s, _ = jax.lax.scan(jax.checkpoint(sub), s, None, length=S); return s, s[0]
                sT, v = jax.lax.scan(frame, s_in, None, length=len(Fk) - 1); return v, sT

            def loss(pp, i=None):
                v = run(pp)[0]; L0 = jnp.mean((v - y) ** 2); LB = jnp.sum(bd * jax.nn.relu(0.25 - sg * (v - 0.5)) ** 2) / nb; return L0 + lamB * LB, (L0, LB)
            Fb = jnp.asarray(Fk[1:].astype(bool)); roll_bin = jax.jit(jax.vmap(lambda pp: jnp.mean((run(pp)[0] > 0.5) != Fb)))
            P, res = _fit(jax, jnp, P, jax.vmap(jax.value_and_grad(loss, has_aux=True), in_axes=(0, None)), roll_bin, steps_per_phase, sched, eval_every, out, f"p{k}", log, [f"s{seed}"])
            os.replace(os.path.join(out, f"best_p{k}_s{seed}.npz"), ck) if os.path.exists(os.path.join(out, f"best_p{k}_s{seed}.npz")) else None
            prev = dict(np.load(ck)); g_k = prev.pop("init_g", None)
            if g_k is not None: g_best = g_k
            rec["phases"] = rec["phases"][:k] + [dict(p, best=res["best"][0], train_s=res["train_s"], compile_s=res["compile_s"], steps=steps_per_phase)]; _atomic(rec_path, rec)
        # carried state at the end of phase k with its BEST rule (what the export will replay)
        E = {n: jnp.asarray(prev[n])[None] for n in PNAMES}
        if state is None:
            h0 = upsample(jnp.asarray(g_best), init_factor, H, W, jnp) if (init_factor and g_best is not None) else None
            _, sT = roll(E, jnp.asarray(Fk[0], jnp.float32), jnp.zeros(len(Fk) - 1, jnp.int32), h0)
        else:
            _, sT = jax.jit(lambda E, s: _carry(jax, jnp, E, s, len(Fk) - 1, S))(E, jnp.asarray(state))
        state = np.asarray(sT); done.append(k)
    rec.update(T=T, C=C, Fh=Fh, S=S, init_factor=init_factor, steps_per_phase=steps_per_phase, sched=list(sched), warm=warm); _atomic(rec_path, rec)
    return rec


def _carry(jax, jnp, E, s, n, S):
    def frame(s, _):
        def sub(s, _):
            q = jax.nn.relu(jax.lax.conv_general_dilated(s[None], E["Wp"][0], (1, 1), "SAME", precision=jax.lax.Precision.HIGHEST)[0] + E["bp"][0][:, None, None])
            return s + jnp.einsum("cf,fhw->chw", E["Wu"][0], q, precision=jax.lax.Precision.HIGHEST) + E["bu"][0][:, None, None], None
        s, _ = jax.lax.scan(sub, s, None, length=S); return s, None
    sT, _ = jax.lax.scan(frame, s, None, length=n); return None, sT


def collect_greedy(out, phases, seed=0):
    """Stack greedy per-phase best rules into an 'indep' parameter set (+ init_g if learned)."""
    Ps = [dict(np.load(os.path.join(out, f"best_p{k}_s{seed}.npz"))) for k in range(len(phases))]
    P = {n: np.stack([p[n] for p in Ps]) for n in PNAMES}; g = Ps[0].get("init_g")
    return P, g


def chunked(F, phases, out, seed=0, C=16, Fh=64, S=4, steps=10000, sched=("cos", 5e-4, 5e-6), lamB=2.0, init_factor=None, eval_every=1000,
            chunk_min=200, chunk_max=280, log=print, noise=0.0):
    """Chunked-joint training for long sequences. Consecutive phases are grouped into chunks of ~chunk_min..chunk_max frames.
    Inside a chunk all its phase rules are trained JOINTLY with full BPTT (earlier phases can prepare hidden state for later
    ones); the chunk starts from the ACTUAL carried full state (visible + hidden) produced by the already-trained chunks
    (best checkpoints), so there is no reset; gradients do not cross chunk boundaries. Chunk 0 starts from the first frame
    with hidden 0 or the learned low-res init (init_factor). Resumable per chunk (chunks.json). Writes best_c{i}_s{seed}.npz."""
    import jax, jax.numpy as jnp
    os.makedirs(out, exist_ok=True); T, H, W = F.shape; roll = jax_rollout(S)
    chunks, cur = [], []
    for p in phases:
        cur.append(p)
        if cur[-1]["end"] - cur[0]["start"] >= chunk_min or (p is phases[-1]): chunks.append(cur); cur = []
    if cur: chunks.append(cur)
    rec_path = os.path.join(out, "chunks.json"); rec = json.load(open(rec_path)) if os.path.exists(rec_path) else dict(chunks=[])
    state = None; g_best = None; Pall = []
    for ci, ch in enumerate(chunks):
        a, b = ch[0]["start"], ch[-1]["end"]; Fk = F[a - 1:b]; Kc = len(ch); ck = os.path.join(out, f"best_c{ci}_s{seed}.npz")
        ph = np.zeros(len(Fk), np.int32)
        for k, p in enumerate(ch): ph[p["start"] - a + 1:p["end"] - a + 1] = k
        phj = jnp.asarray(ph[1:])
        if not (ci < len(rec["chunks"]) and os.path.exists(ck)):
            p0 = init_params(np.random.default_rng(seed + 7919 * ci), "indep", Kc, C, Fh)
            if ci == 0 and init_factor: p0["init_g"] = np.zeros((C - 1, -(-H // init_factor), -(-W // init_factor)), np.float32)
            P = {k: jnp.asarray(v)[None] for k, v in p0.items()}; y, sg, bd = _targets(jnp, Fk); nb = jnp.maximum(bd.sum(), 1)
            s_in = None if state is None else jnp.asarray(state)

            def run(pp, s_in=s_in, nz=None):
                E = {n: pp[n] for n in PNAMES}
                if s_in is None:
                    h0 = upsample(pp["init_g"], init_factor, H, W, jnp) if "init_g" in pp else None
                    return roll(E, jnp.asarray(Fk[0], jnp.float32), phj, h0, nz)[0]
                return roll(E, s_in[0], phj, s_in[1:], nz)[0]

            def loss(pp, i=None):
                nz = (jax.random.fold_in(jax.random.PRNGKey(seed * 1000 + ci), jnp.asarray(i, jnp.uint32)), noise) if (noise and i is not None) else None
                v = run(pp, nz=nz); L0 = jnp.mean((v - y) ** 2); LB = jnp.sum(bd * jax.nn.relu(0.25 - sg * (v - 0.5)) ** 2) / nb; return L0 + lamB * LB, (L0, LB)
            Fb = jnp.asarray(Fk[1:].astype(bool)); roll_bin = jax.jit(jax.vmap(lambda pp: jnp.mean((run(pp) > 0.5) != Fb)))
            _, res = _fit(jax, jnp, P, jax.vmap(jax.value_and_grad(loss, has_aux=True), in_axes=(0, None)), roll_bin, steps, sched, eval_every, out, f"c{ci}", log, [f"s{seed}"])
            rec["chunks"] = rec["chunks"][:ci] + [dict(start=a, end=b, phases=ch, best=res["best"][0], train_s=res["train_s"], compile_s=res["compile_s"], steps=steps)]
            _atomic(rec_path, rec)
        best = dict(np.load(ck)); g = best.pop("init_g", None)
        if g is not None: g_best = g
        Pall.append(best)
        E = {n: jnp.asarray(best[n]) for n in PNAMES}
        if state is None:
            h0 = upsample(jnp.asarray(g_best), init_factor, H, W, jnp) if (init_factor and g_best is not None) else None
            _, sT = roll(E, jnp.asarray(Fk[0], jnp.float32), phj, h0)
        else:
            _, sT = roll(E, jnp.asarray(state[0]), phj, jnp.asarray(state[1:]))
        state = np.asarray(sT)
        if not np.isfinite(state).all(): log(f"chunk {ci}: carried state not finite"); break
    P = {n: np.concatenate([p[n] for p in Pall]) for n in PNAMES}
    rec.update(T=T, C=C, Fh=Fh, S=S, init_factor=init_factor, steps=steps, sched=list(sched), n_chunks=len(chunks), done=len(Pall), noise=noise); _atomic(rec_path, rec)
    return P, g_best, [p for ch in chunks[:len(Pall)] for p in ch]
