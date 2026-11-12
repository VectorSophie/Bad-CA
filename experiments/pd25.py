"""Path D2.5 / D3: visible-only logic CA learning. The learner sees ONLY the visible channel; k latent channels must be
learned. Hidden initialisation (never from the generator's hidden state):
  zero    hidden = 0 at the window start
  burnin  observation prefix of P frames: start P frames earlier with hidden 0 and run the circuit with its visible
          channel clamped to the observed frames; its own logic fills the hidden channels; then it runs free.
          Cost: P extra visible frames (counted).
Objectives: soft rollout (softmax gates, real-valued states) or ste (argmax gates and rounded states forward,
straight-through gradients). Horizon curriculum 1, 2, 4, 8, 16 (equal step-evaluations per stage).
Tasks: gh (synthetic, Greenberg-Hastings; visible = excited) or badapple:<start> clips.
usage: pd25.py TASK SETTING SEED [BUDGET] where SETTING in zero | burnin | burnin_ste | zero_ste"""
import json, os, sys, time
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
OUT = os.path.join(ROOT, "runs", "path_d", "d25")
K_HIDDEN, P_BURNIN, HMAX = 2, 2, 16


# ---------------- data: lists of visible sequences (n, T, H, W) uint8, split train / val / test ----------------
def gh_data():
    """48x48 periodic Greenberg-Hastings from sparse starts (excited 0.004, refractory 0.004): fronts expand, then
    spiral waves persist with ~0.12 of cells changing per step (checked before any fit). Visible = excited only.
    test_dense: refractory-rich starts (0.08, 0.2), a different regime (dense period-3 cycling), out of family."""
    import pd_d1
    def seqs(seed, n, dens=(0.004, 0.004)):
        X = pd_d1.init_states("gh", n, np.random.default_rng(seed), size=48, density=dens); return pd_d1.trajectories("gh", X, 160)[..., 0]
    return dict(train=seqs(1, 16), val=seqs(2, 8), test=seqs(3, 16), test_dense=seqs(4, 8, (0.08, 0.2)), periodic=True,
                windows=dict(train=(P_BURNIN, 140), eval=[4, 40, 100]), cycle_baseline=True)


def hparams(C):
    K = 16; n_in = K * C * 2 + C; taper = []; w = 64
    while w > 2 * C: taper.append(w); w //= 2
    taper.append(w)                                                  # last width <= 2C so out*2 >= in holds for C
    layers = [n_in] + [128] * 8 + taper + [C]
    return dict(channels=C, perceive=dict(n_kernels=K, layers=[9, 8, 4, 2], connections=["first_kernel", "unique", "unique", "unique"]),
                update=dict(layers=layers, connections=["unique"] * len(layers)))


# ---------------- training ----------------
def train(data, setting, seed, budget, batch=8, k=K_HIDDEN):
    import jax, jax.numpy as jnp
    from ca import dlca
    from ca.difflogic import adamw
    C = 1 + k; periodic = data["periodic"]; burn = P_BURNIN if setting.startswith("burnin") else 0; ste = setting.endswith("_ste")
    V = jnp.asarray(data["train"], jnp.float32); n, T = V.shape[:2]; lo, hi = data["windows"]["train"]
    params, wires = dlca.init(hparams(C), seed); oi, ou = adamw(0.05); st = oi(params); key = jax.random.PRNGKey(seed)
    gate = "ste" if ste else False
    def rnd(s): return s + jax.lax.stop_gradient(jnp.round(s) - s) if ste else s

    def make(h):
        def loss(p, vis):                                          # vis: (B, burn + h + 1, H, W)
            s = jnp.zeros(vis.shape[:1] + vis.shape[2:] + (C,)).at[..., 0].set(vis[:, 0])
            for i in range(burn):                                   # observation prefix: visible clamped
                s = rnd(dlca.step(p, wires, s, gate, periodic)).at[..., 0].set(vis[:, i + 1])
            def body(g, _):
                g = rnd(dlca.step(p, wires, g, gate, periodic)); return g, g[..., 0]
            _, seq = jax.lax.scan(jax.checkpoint(body), s, None, length=h)
            return jnp.square(jnp.moveaxis(seq, 0, 1) - vis[:, burn + 1:]).sum() / h
        @jax.jit
        def f(p, s_, kk):
            k1, k2 = jax.random.split(kk); w = jax.random.randint(k1, (batch,), 0, n); t = jax.random.randint(k2, (batch,), lo - burn, hi - h)
            vis = jnp.stack([V[w, t + i] for i in range(burn + h + 1)], 1)
            l, g = jax.value_and_grad(loss)(p, vis); p, s_ = ou(g, s_, p); return p, s_, l
        return f

    log = []; t0 = time.time(); done = 0
    for h in (1, 2, 4, 8, 16):
        f = make(h); ep = budget // 5 // (batch * h)
        for e in range(ep):
            key, kk = jax.random.split(key); params, st, l = f(params, st, kk); done += h * batch
            if e % max(1, ep // 5) == 0 or e == ep - 1: log.append(dict(h=h, epoch=e, soft=float(l), t=round(time.time() - t0, 1)))
    jax.block_until_ready(params)
    return params, wires, log, time.time() - t0


# ---------------- evaluation (hard circuit, NumPy executor) ----------------
def hard_window(net, seq, t0, n, burn, periodic, C):
    """Fresh-start transfer: visible = seq[t0], hidden from the allowed mechanism; returns predicted visible (n, H, W)."""
    from ca import dlca
    s = np.zeros(seq.shape[1:] + (C,), np.uint8); s[..., 0] = seq[t0 - burn]
    for i in range(burn): s = dlca.net_eval(net, dlca.np_patches(s, periodic)); s[..., 0] = seq[t0 - burn + i + 1]
    out = []
    for _ in range(n): s = dlca.net_eval(net, dlca.np_patches(s, periodic)); out.append(s[..., 0].copy())
    return np.stack(out)


def lut_baseline(train, periodic):
    """Best memoryless visible rule (3x3 majority table fitted on all training transitions): the 'propagation' baseline."""
    from ca import dlca
    cnt = np.zeros((512, 2))
    for s in train:
        for t in range(len(s) - 1):
            P = dlca.np_patches(s[t][..., None], periodic)[..., 0]; key = (P.astype(np.int64) << np.arange(9)).sum(-1)
            np.add.at(cnt, (key.ravel(), s[t + 1].ravel()), 1)
    tab = (cnt[:, 1] > cnt[:, 0]).astype(np.uint8)
    def roll(v, n):
        out = []
        for _ in range(n):
            P = dlca.np_patches(v[..., None], periodic)[..., 0]; v = tab[(P.astype(np.int64) << np.arange(9)).sum(-1)]; out.append(v)
        return np.stack(out)
    return roll


def motion(B, F):
    from rr_run import evaluate
    ev = evaluate(np.concatenate([F[:1], B]), F); return ev["lag4_f1"], ev["lag4_mass"] or 0.0


def evaluate_net(net, data, setting, C, extra=None):
    periodic = data["periodic"]; burn = P_BURNIN if setting.startswith("burnin") else 0; lut = lut_baseline(data["train"], periodic); r = {}
    for split in [s for s in ("val", "test", "test_dense") if s in data]:
        rows, dead = [], []
        for seq in data[split]:
            for t0 in data["windows"]["eval"]:
                n = min(128, len(seq) - 1 - t0)
                if n < 16 or t0 < burn: continue
                if not (seq[t0:t0 + n + 1] != seq[t0]).any(): dead.append(1); continue   # no motion at all: trivially "exact", not scored
                B = hard_window(net, seq, t0, n, burn, periodic, C); F = seq[t0:t0 + n + 1]; e = (B != F[1:]).mean((1, 2))
                Lb = lut(seq[t0], n); f1, ms = motion(B[:min(n, 64)], F[:min(n, 64) + 1])
                cyc = {}
                if data.get("cycle_baseline"):   # 2-frame memory, NO neighbour dependence: e' = ~e & ~e_prev (the trivial period-3 cycle)
                    a, b = seq[t0 - 1], seq[t0]; cy = []
                    for _ in range(n): a, b = b, ((1 - b) & (1 - a)).astype(np.uint8); cy.append(b)
                    cy = np.stack(cy); cyc = {f"cycle_{h}": float((cy[:h] != F[1:h + 1]).mean()) for h in (16, 32, 64, 128) if h <= n}
                rows.append(dict({f"err_{h}": float(e[:h].mean()) for h in (16, 32, 64, 128) if h <= n},
                                 **{f"hold_{h}": float((F[1:h + 1] != F[0]).mean()) for h in (16, 32, 64, 128) if h <= n},
                                 **{f"lut_{h}": float((Lb[:h] != F[1:h + 1]).mean()) for h in (16, 32, 64, 128) if h <= n},
                                 **cyc, f1=f1, mass=ms, diverge=next((t + 1 for t in range(n) if e[t] > 0.05), None), exact=bool(e.max() == 0), n=n))
        agg = {k: float(np.mean([x[k] for x in rows if k in x])) for k in rows[0] if k not in ("diverge", "exact", "n")}
        agg["dead_windows_skipped"] = len(dead)
        if data.get("cycle_baseline"): agg["beats_all_baselines_16"] = f"{sum(x['err_16'] < min(x['hold_16'], x['lut_16'], x['cycle_16']) for x in rows)}/{len(rows)}"
        agg["exact_windows"] = f"{sum(x['exact'] for x in rows)}/{len(rows)}"; agg["diverge_median"] = float(np.median([x["diverge"] or 999 for x in rows]))
        agg["beats_hold_16"] = f"{sum(x['err_16'] < x['hold_16'] for x in rows)}/{len(rows)}"; agg["beats_lut_16"] = f"{sum(x['err_16'] < x['lut_16'] for x in rows)}/{len(rows)}"
        r[split] = agg
    return r


def main(task, setting, seed, budget):
    from ca import dlca, logic_simplify as ls
    import jax, jax.numpy as jnp
    data = gh_data() if task == "gh" else __import__("pd_d3").clip_data(task)
    d = os.path.join(OUT, f"{task.replace(':', '_')}_{setting}_s{seed}"); os.makedirs(d, exist_ok=True)
    C = 1 + K_HIDDEN; params, wires, log, secs = train(data, setting, seed, budget)
    circ = dlca.harden(params, wires); net = dlca.to_netlist(circ, C); simp = ls.simplify(net)
    r = dict(task=task, setting=setting, seed=seed, budget=budget, k_hidden=K_HIDDEN, burnin_frames=P_BURNIN if setting.startswith("burnin") else 0,
             train_seconds=round(secs, 1), backend=jax.default_backend(), final=log[-1], gates_raw=len(net["gates"]), simplified=ls.stats(simp))
    # soft-to-hard discrepancy on 4 validation windows of 16 steps (same init mechanism)
    burn = r["burnin_frames"]; periodic = data["periodic"]; sd = []
    for seq in data["val"][:4]:
        t0 = data["windows"]["eval"][0]; s = jnp.zeros(seq.shape[1:] + (C,)).at[..., 0].set(seq[t0 - burn])
        for i in range(burn): s = dlca.step(params, wires, s, False, periodic).at[..., 0].set(seq[t0 - burn + i + 1])
        soft = np.asarray(dlca.rollout(params, wires, s, 16, False, periodic, keep=True))[..., 0]
        hard = hard_window(simp, seq, t0, 16, burn, periodic, C); sd.append((float(np.abs(soft - seq[t0 + 1:t0 + 17]).mean()), float((hard != seq[t0 + 1:t0 + 17]).mean())))
    r["soft_vs_hard_16"] = dict(soft_err=float(np.mean([a for a, _ in sd])), hard_err=float(np.mean([b for _, b in sd])))
    r.update(evaluate_net(simp, data, setting, C))
    seq0 = data["test"][0]; t0 = data["windows"]["eval"][0]; init = np.zeros(seq0.shape[1:] + (C,), np.uint8); init[..., 0] = seq0[t0 - burn]
    dlca.export(os.path.join(d, "circuit.logic.json"), simp, C, periodic, init_state=init, steps=128 + burn, prefix=[seq0[t0 - burn + i + 1] for i in range(burn)],
                meta=dict(task=task, setting=setting, seed=seed))
    np.savez_compressed(os.path.join(d, "params.npz"), **{f"{n_}_{i}": np.asarray(g) for n_ in ("perceive", "update") for i, g in enumerate(params[n_])})
    json.dump(r, open(os.path.join(d, "result.json"), "w"), indent=1); open(os.path.join(d, "log.jsonl"), "w").write("\n".join(json.dumps(x) for x in log) + "\n")
    print(json.dumps(r)); return r


if __name__ == "__main__":
    task, setting, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]); budget = int(sys.argv[4]) if len(sys.argv) > 4 else 48_000
    if os.path.exists(os.path.join(OUT, f"{task.replace(':', '_')}_{setting}_s{seed}", "result.json")) and "--force" not in sys.argv: print("exists"); sys.exit(0)
    main(task, setting, seed, budget)
