"""Track G training (GPU via gpuq) and CPU evaluation. Datasets come from the registry (ca/datasets.py).

fit  (G2/G3): the C13 recipe on a contiguous clip, unchanged code: ca.phase_train.chunked (chunked-joint BPTT, state
     carried across phases and chunks with numpy carry, training-only state noise sigma, learned low-res initial hidden),
     C=16, Fh=64 (10,320 params per phase) unless overridden, S=4, cosine lr 5e-4 -> 5e-6, contour loss lamB 2.
     Export ncab-v2; evaluation (CPU, numpy replay of the export alone) reports, separately:
       recon     the fitted clip, continuous from frame 0 (the C13 setting: a fitted, scheduled trajectory)
       cont      UNSEEN continuation: the last phase's rule keeps running for up to 32 frames past the trained end
       fresh     per phase: start that phase's rule from the TRUE frame at its start with hidden 0 (no carried state)
win  (G4): ONE shared rule (no phases) trained on many short windows. Window = 18 frames: frame 0 is the start, frames
     1..2 are an observation prefix (after each of the first 2 frames the visible channel is overwritten by the true
     frame), frames 3..17 are generated freely and scored. Fixed steps and batch (equal compute across data budgets).
     Splits: blocks of 64 frames, pattern train, train, val, train, train, test (repeating), 16-frame guard gaps at
     every block edge (no window crosses a gap). Scenes can span several blocks, so test windows are also reported for
     the 'novel_scene' subset: no training window lies in the same scene (scenes = runs between ca.segment cut frames). Data budget f: 'spread' = a random f-subset of all train windows (more scenes);
     'contig' = windows from the first f of the train blocks only (fewer scenes, same window count per block).
     Model selection: none (the final step is used); val is reported, test is read once per run.
usage: gen_train.py fit  --ds D --start A --frames N [--stride k] --phases uniform:K|auto:K --steps S --out DIR [--Fh F]
       gen_train.py win  --ds D [--rep R] --frac f --mode spread|contig [--stride k] --steps S --seed s --out DIR
       gen_train.py eval_win RUN_DIR [--on DATASET[:REP]]       (CPU; numpy replay of the saved rule)"""
import argparse, json, os, sys, time
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path[:0] = [ROOT, os.path.dirname(os.path.abspath(__file__))]
from ca import datasets, nca
PRE, WL, BLOCK, GAP = 2, 18, 64, 16


# ---------------------------------------------------------------- fit (phased, C13 recipe) ----------------------------------------------------------------
def fit(a):
    from ca import phase_nca, segment
    from ca.phase_train import chunked
    F = datasets.frames(a.ds, a.rep)[a.start:a.start + a.frames * a.stride:a.stride]; os.makedirs(a.out, exist_ok=True)
    kind, K = a.phases.split(":"); K = int(K)
    ph = segment.uniform_k(len(F), K) if kind == "uniform" else segment.best_k(F, K, lmin=min(8, (len(F) - 1) // K))["phases"]
    json.dump(vars(a), open(os.path.join(a.out, "config.json"), "w"), indent=1); t0 = time.time()
    P, g, used = chunked(F, ph, os.path.join(a.out, "train"), seed=a.seed, C=a.C, Fh=a.Fh, S=4, steps=a.steps, sched=("cos", a.lr, a.lr / 100), lamB=2.0,
                         init_factor=4, eval_every=max(100, a.steps // 20), noise=a.noise, cpu_carry=True, max_hours=a.max_hours,
                         chunk_min=min(200, len(F) - 1), log=lambda s: print(s, flush=True))
    T = used[-1]["end"]; init = dict(kind="lowres", factor=4, g=np.asarray(g)) if g is not None else None
    sizes = phase_nca.export(os.path.join(a.out, "model.ncab"), P, "indep", used, F[0], T, 4, init=init,
                             meta=dict(dataset=a.ds, rep=a.rep or datasets.PRIMARY[a.ds], start=a.start, stride=a.stride, phases=a.phases, steps=a.steps, noise=a.noise))
    json.dump(dict(train_s=round(time.time() - t0, 1), sizes=sizes, frames=T, phases=len(used)), open(os.path.join(a.out, "train_done.json"), "w"), indent=1)


def eval_fit(run):
    """CPU: replay the export, then recon / continuation / fresh-start metrics. Also writes the binary replay (packed)."""
    import pv_v0
    from ca import phase_nca
    cfg = json.load(open(os.path.join(run, "config.json"))); full = datasets.frames(cfg["ds"], cfg["rep"])[cfg["start"]::cfg["stride"]]
    hd, E, sched, h0, f0 = phase_nca.load(os.path.join(run, "model.ncab")); T = hd["T"]; B, V = phase_nca.rollout_np(E, f0, sched, hd["S"], h0)
    F = full[:T]; r = dict(run=run, T=T, phases=int(sched.max()) + 1, recon=pv_v0.metrics(B[1:], F))
    blip = lambda X: float(((X[:-2] == X[2:]) & (X[1:-1] != X[:-2])).mean())
    r["recon"]["blip_pred"], r["recon"]["blip_true"] = blip(B.astype(np.int8)), blip(F.astype(np.int8))
    # continuation past the trained end with the last rule (state carried from the replay)
    n = min(32, len(full) - T)
    if n >= 4:
        last = int(sched[-1]); p = {k: E[k][last] for k in phase_nca.PNAMES}; s = _state_at(E, f0, sched, hd["S"], h0); out = []
        for _ in range(n):
            for _ in range(hd["S"]): s = nca.step_np(p, s)
            out.append((s[0] > 0.5).astype(np.uint8))
        r["cont"] = pv_v0.metrics(np.stack(out), full[T - 1:T + n])
    # fresh start per phase: true frame at phase start, hidden 0, that phase's rule, phase length (<= 32)
    fr = []
    bounds = [0] + [t for t in range(1, T) if sched[t] != sched[t - 1]] + [T]
    for i in range(len(bounds) - 1):
        a0, b0 = max(bounds[i], 1) - 1, min(bounds[i + 1], bounds[i] + 32)
        if b0 - a0 < 4: continue
        k = int(sched[min(a0 + 1, T - 1)]); p = {kk: E[kk][k] for kk in phase_nca.PNAMES}; s = np.zeros((E["Wu"].shape[1],) + f0.shape, np.float32); s[0] = F[a0]; out = []
        for _ in range(b0 - a0 - 1):
            for _ in range(hd["S"]): s = nca.step_np(p, s)
            out.append((s[0] > 0.5).astype(np.uint8))
        m = pv_v0.metrics(np.stack(out), F[a0:b0]); m.update(carried_err=float((B[a0 + 1:b0] != F[a0 + 1:b0]).mean())); fr.append(m)
    r["fresh"] = {k: float(np.mean([x[k] for x in fr])) for k in ("err", "hold", "f1", "iou", "carried_err")}; r["fresh"]["n"] = len(fr)
    r["sizes"] = dict(params=int(sum(E[k].size for k in phase_nca.PNAMES)), file_bytes=os.path.getsize(os.path.join(run, "model.ncab")),
                      frames_packed_bytes=int(np.ceil(F.size / 8)), frames_zlib_bytes=len(__import__("zlib").compress(np.packbits(F).tobytes(), 9)))
    np.savez_compressed(os.path.join(run, "replay_bits.npz"), bits=np.packbits(B), shape=B.shape)
    json.dump(r, open(os.path.join(run, "eval.json"), "w"), default=float, indent=0); return r


def _state_at(E, f0, sched, S, h0):
    from ca import phase_nca
    C = E["Wu"].shape[1]; s = np.zeros((C,) + f0.shape, np.float32); s[0] = f0
    if h0 is not None: s[1:] = h0
    for t in range(1, len(sched)):
        p = {k: E[k][sched[t]] for k in phase_nca.PNAMES}
        for _ in range(S): s = nca.step_np(p, s)
    return s


# ---------------------------------------------------------------- win (single shared rule, many windows) ----------------------------------------------------------------
def splits(T, stride=1):
    """Block pattern over the (strided) frame axis: train, train, val, train, train, test; guard gaps at block edges."""
    pat = ["train", "train", "val", "train", "train", "test"]; B = BLOCK // stride; G = GAP // stride; win = {"train": [], "val": [], "test": []}; blocks = []
    for i, b0 in enumerate(range(0, T, B)):
        role = pat[i % len(pat)]; lo, hi = b0 + G // 2, min(b0 + B, T) - G // 2; blocks.append((role, lo, hi))
        for s in range(lo, hi - WL + 1, WL // 2): win[role].append(s)
    return win, blocks


def pick_train(win, blocks, frac, mode, seed):
    tr = np.array(win["train"])
    if frac >= 1: return tr
    if mode == "spread": return np.sort(np.random.default_rng(seed).choice(tr, max(1, int(round(frac * len(tr)))), replace=False))
    tb = [b for b in blocks if b[0] == "train"]; keep = tb[:max(1, int(round(frac * len(tb))))]
    return np.array([s for s in tr if any(lo <= s < hi for _, lo, hi in keep)])


def win_train(a):
    import jax, jax.numpy as jnp
    from ca.diag10 import band
    full = datasets.frames(a.ds, a.rep)[::a.stride]; T, H, W = full.shape; win, blocks = splits(T, a.stride); tr = pick_train(win, blocks, a.frac, a.mode, a.seed)
    os.makedirs(a.out, exist_ok=True); json.dump(dict(vars(a), n_train_windows=len(tr), n_train_frames=int(len({t for s in tr for t in range(s, s + WL)})), shape=[T, H, W]), open(os.path.join(a.out, "config.json"), "w"), indent=1)
    Fj = jnp.asarray(full, jnp.float32); Bd = jnp.asarray(band(full), jnp.float32); trj = jnp.asarray(tr)
    step, _ = nca.jax_fns(4); P = {k: jnp.asarray(v) for k, v in nca.init_params(np.random.default_rng(a.seed), a.C, a.Fh).items()}

    def roll(P, s0, key):
        f = Fj[s0 + jnp.arange(WL)]; s = jnp.zeros((a.C, H, W)).at[0].set(f[0]); keys = jax.random.split(key, WL - 1)
        def frame(s, xs):
            t, k = xs
            s, _ = jax.lax.scan(lambda s, _: (jax.checkpoint(step)(P, s), None), s, None, length=4); v = s[0]
            s = jnp.where(t <= PRE, s.at[0].set(f[t]), s) + 0.02 * jax.random.normal(k, s.shape)
            return s, v
        _, v = jax.lax.scan(frame, s, (jnp.arange(1, WL), keys)); return v[PRE:], f[PRE + 1:], Bd[s0 + jnp.arange(PRE + 1, WL)]

    def loss(P, idx, key):
        v, y, bd = jax.vmap(lambda s0, k: roll(P, s0, k))(idx, jax.random.split(key, idx.shape[0]))
        sg = 2 * y - 1; L0 = jnp.mean((v - y) ** 2); LB = jnp.sum(bd * jax.nn.relu(0.25 - sg * (v - 0.5)) ** 2) / jnp.maximum(bd.sum(), 1)
        return L0 + 2.0 * LB

    b1, b2 = 0.9, 0.999; m = jax.tree_util.tree_map(jnp.zeros_like, P); vv = jax.tree_util.tree_map(jnp.zeros_like, P)

    @jax.jit
    def upd(P, m, vv, i, key):
        k1, k2 = jax.random.split(key); idx = trj[jax.random.randint(k1, (a.batch,), 0, trj.shape[0])]
        l, g = jax.value_and_grad(loss)(P, idx, k2)
        g = jax.tree_util.tree_map(lambda x: x / (jnp.sqrt(jnp.sum(x ** 2)) + 1e-8), g)
        lr = 2e-5 + 0.5 * (1e-3 - 2e-5) * (1 + jnp.cos(jnp.pi * i / a.steps))
        m = jax.tree_util.tree_map(lambda x, y: b1 * x + (1 - b1) * y, m, g); vv = jax.tree_util.tree_map(lambda x, y: b2 * x + (1 - b2) * y * y, vv, g)
        P = jax.tree_util.tree_map(lambda p, x, y: p - lr * (x / (1 - b1 ** (i + 1))) / (jnp.sqrt(y / (1 - b2 ** (i + 1))) + 1e-8), P, m, vv)
        return P, m, vv, l

    key = jax.random.PRNGKey(a.seed); t0 = time.time(); log = []; tc = None
    for i in range(a.steps):
        key, k = jax.random.split(key); P, m, vv, l = upd(P, m, vv, i, k)
        if i == 0: jax.block_until_ready(l); tc = time.time() - t0; t1 = time.time()
        if i % 500 == 0 or i == a.steps - 1: log.append(dict(step=i, loss=float(l), t=round(time.time() - t0, 1))); print(log[-1], flush=True)
    np.savez(os.path.join(a.out, "rule.npz"), **{k: np.asarray(v) for k, v in P.items()})
    peak = jax.local_devices()[0].memory_stats().get("peak_bytes_in_use") if jax.default_backend() == "gpu" else None
    json.dump(dict(log=log, compile_s=round(tc, 1), train_s_excl_compile=round(time.time() - t1, 1), peak_bytes=peak, backend=jax.default_backend()),
              open(os.path.join(a.out, "train_done.json"), "w"), indent=1)


def eval_win(run, on=None):
    """CPU numpy replay of the saved rule on val and test windows (2-frame prefix, 15 free frames), plus hold."""
    import pv_v0
    cfg = json.load(open(os.path.join(run, "config.json"))); P = dict(np.load(os.path.join(run, "rule.npz")))
    ds, rep = (on.split(":") + [None])[:2] if on else (cfg["ds"], cfg["rep"]); stride = cfg["stride"] if not on else 1
    full = datasets.frames(ds, rep)[::stride]; win, blocks = splits(len(full), stride); res = {}
    from ca import segment
    scene = np.cumsum(segment.signals(full)["cut"]); trw = pick_train(win, blocks, cfg["frac"], cfg["mode"], cfg["seed"]) if not on else np.array([], int)
    seen = {int(scene[t]) for s in trw for t in range(s, s + WL)}
    for split in ("val", "test"):
        rows = []
        for s in win[split]:
            f = full[s:s + WL]; st = np.zeros((P["Wu"].shape[0],) + f[0].shape, np.float32); st[0] = f[0]; out = []
            for t in range(1, WL):
                for _ in range(4): st = nca.step_np(P, st)
                if t <= PRE: st[0] = f[t]
                else: out.append((st[0] > 0.5).astype(np.uint8))
            m = pv_v0.metrics(np.stack(out), f[PRE:]); m["onestep_err"] = float((out[0] != f[PRE + 1]).mean()); m["onestep_hold"] = float((f[PRE] != f[PRE + 1]).mean())
            m.pop("err_t"); m.pop("iou_t"); m["start"] = int(s); m["novel_scene"] = not ({int(scene[t]) for t in range(s, s + WL)} & seen); rows.append(m)
        agg = {k: float(np.mean([r[k] for r in rows])) for k in ("err", "hold", "f1", "prec", "rec", "iou", "bf1", "onestep_err", "onestep_hold")}
        agg["beats_hold"] = float(np.mean([r["err"] < r["hold"] for r in rows])); agg["collapse_frac"] = float(np.mean([r["collapse"] is not None for r in rows])); agg["n"] = len(rows)
        nov = [r for r in rows if r["novel_scene"]]
        agg["novel_scene"] = {k: float(np.mean([r[k] for r in nov])) for k in ("err", "hold", "f1", "iou")} if nov else None; agg["novel_n"] = len(nov)
        res[split] = dict(agg=agg, rows=rows)
    name = "eval.json" if not on else f"eval_on_{ds}.json"; json.dump(res, open(os.path.join(run, name), "w"), default=float)
    return {k: v["agg"] for k, v in res.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("fit"); p.add_argument("--ds", required=True); p.add_argument("--rep"); p.add_argument("--start", type=int, default=0); p.add_argument("--frames", type=int, required=True)
    p.add_argument("--stride", type=int, default=1); p.add_argument("--phases", required=True); p.add_argument("--steps", type=int, default=8000); p.add_argument("--out", required=True)
    p.add_argument("--C", type=int, default=16); p.add_argument("--Fh", type=int, default=64); p.add_argument("--noise", type=float, default=0.02); p.add_argument("--seed", type=int, default=0); p.add_argument("--max-hours", type=float)
    p.add_argument("--lr", type=float, default=5e-4)
    p = sub.add_parser("win"); p.add_argument("--ds", required=True); p.add_argument("--rep"); p.add_argument("--frac", type=float, default=1.0); p.add_argument("--mode", default="spread")
    p.add_argument("--stride", type=int, default=1); p.add_argument("--steps", type=int, default=6000); p.add_argument("--batch", type=int, default=16); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--C", type=int, default=16); p.add_argument("--Fh", type=int, default=64); p.add_argument("--out", required=True)
    p = sub.add_parser("eval_fit"); p.add_argument("run")
    p = sub.add_parser("eval_win"); p.add_argument("run"); p.add_argument("--on")
    a = ap.parse_args()
    if a.cmd == "fit": fit(a)
    elif a.cmd == "win": win_train(a)
    elif a.cmd == "eval_fit": print(json.dumps({k: v for k, v in eval_fit(a.run).items() if k in ("recon", "cont", "fresh", "sizes")}, default=float)[:3000])
    else: print(json.dumps(eval_win(a.run, a.on)))
