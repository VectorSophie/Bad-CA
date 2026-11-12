"""Path D, stage D1: can a learned logic CA keep motion over long horizons? Synthetic tasks with known local generators;
objectives A one-step, B 16-step rollout, C horizon curriculum (1, 2, 4, 8, 16), all at the same budget of CA-step
evaluations, the same architecture, data and evaluation split. Full state is supervised (inputs of one-step training
need it). usage: pd_d1.py TASK OBJECTIVE SEED [BUDGET]   (GPU: gpuq -L pd-d1 -G 1 ...)"""
import json, os, sys, time
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax, jax.numpy as jnp
from ca import dlca
from ca.difflogic import adamw

OUT = os.path.join(ROOT, "runs", "path_d", "d1")
N, T_TRAIN, T_EVAL, H_ROLL = 32, 64, 128, 16


# ---------------- generators: (H, W, C) uint8 -> next state ----------------
def _sh(g, dy, dx, periodic):
    if periodic: return np.roll(np.roll(g, dy, 0), dx, 1)
    o = np.zeros_like(g); H, W = g.shape[:2]
    o[max(dy, 0):H + min(dy, 0), max(dx, 0):W + min(dx, 0)] = g[max(-dy, 0):H + min(-dy, 0), max(-dx, 0):W + min(-dx, 0)]; return o


def n8(v, periodic): return sum(_sh(v, dy, dx, periodic) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)


TASKS = {   # name: (channels, periodic, generator, init density for channel 0 / other channels)
    "translate": (1, True, lambda g: _sh(g, 0, 1, True), (0.3,)),
    "blink": (2, True, lambda g: np.stack([g[..., 0] ^ g[..., 1], g[..., 1]], -1), (0.5, 0.3)),
    "majority": (1, True, lambda g: ((n8(g[..., 0], True) + g[..., 0]) >= 5).astype(np.uint8)[..., None], (0.5,)),
    "gh": (2, True, lambda g: np.stack([((g[..., 0] == 0) & (g[..., 1] == 0) & (n8(g[..., 0], True) > 0)).astype(np.uint8), g[..., 0]], -1), (0.08, 0.0)),
    "propagate": (1, False, lambda g: (g[..., 0] | _sh(g[..., 0], 1, 0, False)).astype(np.uint8)[..., None], (0.03,)),
}


def init_states(task, n, rng, size=N, density=None):
    C, _, _, dens = TASKS[task]; dens = density or dens
    g = np.stack([(rng.random((n, size, size)) < dens[min(c, len(dens) - 1)]).astype(np.uint8) for c in range(C)], -1)
    if task == "gh": g[..., 1] &= 1 - g[..., 0]
    return g


def trajectories(task, X, T):
    f = TASKS[task][2]; out = [X]
    for _ in range(T): out.append(np.stack([f(x) for x in out[-1]]))
    return np.stack(out, 1)                                            # (n, T+1, H, W, C)


def hparams(C):
    K = 16; n_in = K * C * 2 + C; taper = [w for w in (64, 32, 16, 8, 4, 2) if w >= 2 * C]   # halve while out*2 >= in holds
    layers = [n_in] + [128] * 8 + taper + [C]
    return dict(channels=C, perceive=dict(n_kernels=K, layers=[9, 8, 4, 2], connections=["first_kernel", "unique", "unique", "unique"]),
                update=dict(layers=layers, connections=["unique"] * len(layers)))


def schedule(obj, budget, batch):
    """[(horizon, epochs)] with sum(horizon * epochs * batch) == budget (CA-step evaluations per grid)."""
    if obj == "onestep": return [(1, budget // batch)]
    if obj == "rollout": return [(H_ROLL, budget // (batch * H_ROLL))]
    hs = [1, 2, 4, 8, 16]; per = budget // len(hs)                    # curriculum: equal step-evaluations per stage
    return [(h, per // (batch * h)) for h in hs]


def train(task, obj, seed, budget, batch=8):
    C, periodic, _, _ = TASKS[task]; rng = np.random.default_rng(seed)
    TR = trajectories(task, init_states(task, 16, rng), T_TRAIN).astype(np.float32)   # 16 training worlds
    params, wires = dlca.init(hparams(C), seed); oi, ou = adamw(0.05); st = oi(params); key = jax.random.PRNGKey(seed)
    TRj = jnp.asarray(TR)

    def make_step(h):
        def loss(p, x, ys, hard):
            def body(g, _):
                g = dlca.step(p, wires, g, hard, periodic); return g, g
            _, seq = jax.lax.scan(jax.checkpoint(body), x, None, length=h)   # (h, B, H, W, C)
            return jnp.square(seq - ys).sum() / h                    # mean over the horizon, summed over cells
        @jax.jit
        def f(p, s, k):
            k1, k2 = jax.random.split(k); w = jax.random.randint(k1, (batch,), 0, TR.shape[0]); t = jax.random.randint(k2, (batch,), 0, T_TRAIN - h + 1)
            x = TRj[w, t]; ys = jnp.stack([TRj[w, t + i + 1] for i in range(h)])
            l, g = jax.value_and_grad(loss)(p, x, ys, False); p, s = ou(g, s, p)
            return p, s, l
        return f

    log = []; t0 = time.time(); done = 0
    for h, ep in schedule(obj, budget, batch):
        f = make_step(h)
        for e in range(ep):
            key, k = jax.random.split(key); params, st, l = f(params, st, k); done += h * batch
            if e % max(1, ep // 10) == 0 or e == ep - 1: log.append(dict(h=h, epoch=e, soft=float(l), step_evals=done, t=round(time.time() - t0, 1)))
    jax.block_until_ready(params)
    return params, wires, log, time.time() - t0


def motion(B, F):
    from rr_run import evaluate
    ev = evaluate(B, F); return dict(lag4_f1=ev["lag4_f1"], mass=ev["lag4_mass"])


def evaluate_run(task, params, wires, seed):
    C, periodic, gen, _ = TASKS[task]; circ = dlca.harden(params, wires); net = dlca.to_netlist(circ, C); r = dict(gates=dlca.gate_counts(circ))
    rng = np.random.default_rng(50_000 + seed)                      # unseen initial states, never used in training or selection
    # exact local function vs generator over every neighbourhood (9C <= 18 inputs)
    n = 9 * C; idx = np.arange(2 ** n); bits = ((idx[:, None] >> np.arange(n)) & 1).astype(np.uint8).reshape(-1, 9, C)
    pred = dlca.net_eval(net, bits); g3 = np.zeros((len(bits), 3, 3, C), np.uint8); g3[:] = bits.reshape(-1, 3, 3, C)
    if periodic: truth = np.stack([gen(np.tile(x, (3, 3, 1)))[4, 4] for x in g3])     # centre of a 3x3 tiling = the rule on this patch
    else: truth = np.stack([gen(np.pad(x, ((1, 1), (1, 1), (0, 0))))[2, 2] for x in g3])   # zero boundary: centre of the zero-padded patch
    wrong = (pred != truth).any(-1); r["table_entries_wrong"] = int(wrong.sum()); r["table_entries"] = int(len(bits))
    if task == "gh":   # the generator never makes a cell both excited and refractory: entries containing one are unreachable
        valid = ~((bits[..., 0] == 1) & (bits[..., 1] == 1)).any(-1); r["valid_entries"] = int(valid.sum()); r["valid_entries_wrong"] = int((wrong & valid).sum())
    X = init_states(task, 16, rng); TE = trajectories(task, X, T_EVAL)
    seen = set(); P = np.stack([dlca.np_patches(TE[i, t], periodic) for i in range(4) for t in range(0, T_EVAL, 4)]).reshape(-1, 9, C)
    keys = (P.reshape(len(P), -1).astype(np.int64) << np.arange(n)).sum(1); seen = np.unique(keys)
    r["reachable_entries_seen"] = int(len(seen)); r["reachable_entries_wrong"] = int(wrong[seen].sum())
    hard = np.stack([dlca.net_rollout(net, x, T_EVAL, periodic) for x in X])                        # (16, T+1, H, W, C)
    err = (hard != TE).mean((2, 3, 4))                                                               # (16, T+1)
    for h in (16, 32, 64, 128): r[f"err_{h}"] = float(err[:, 1:h + 1].mean())
    r["one_step_err"] = float(np.mean([(dlca.net_eval(net, dlca.np_patches(TE[i, t], periodic)) != TE[i, t + 1]).mean() for i in range(16) for t in range(0, T_EVAL, 8)]))
    r["diverge_step"] = [next((t for t in range(1, T_EVAL + 1) if err[i, t] > 0.01), None) for i in range(16)]
    r["exact_rollouts_128"] = int((err[:, 1:].max(1) == 0).sum())
    r["hold_err_128"] = float((TE[:, 1:] != TE[:, :1]).mean())
    m = [motion(hard[i, :, :, :, 0], TE[i, :, :, :, 0]) for i in range(4)]; r["lag4_f1"] = float(np.mean([x["lag4_f1"] for x in m])); r["mass"] = float(np.mean([x["mass"] or 0 for x in m]))
    soft = np.asarray(dlca.rollout(params, wires, jnp.asarray(X[:4], jnp.float32), 16, False, periodic, keep=True))   # (16, 4, H, W, C)
    r["soft_err_16"] = float((np.abs(np.moveaxis(soft, 0, 1) - TE[:4, 1:17])).mean()); r["hard_err_16_same"] = float((hard[:4, 1:17] != TE[:4, 1:17]).mean())
    X2 = init_states(task, 4, rng, size=64); T2 = trajectories(task, X2, 64)                         # larger grid
    r["err_64_on_64x64"] = float(np.mean([(dlca.net_rollout(net, x, 64, periodic) != t).mean() for x, t in zip(X2, T2)]))
    dens = TASKS[task][3]; X3 = init_states(task, 4, rng, density=tuple(min(0.9, 2 * d) for d in dens)); T3 = trajectories(task, X3, 64)   # other density
    r["err_64_density_x2"] = float(np.mean([(dlca.net_rollout(net, x, 64, periodic) != t).mean() for x, t in zip(X3, T3)]))
    return r, net, X[0]


if __name__ == "__main__":
    task, obj, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]); budget = int(sys.argv[4]) if len(sys.argv) > 4 else 32_000
    d = os.path.join(OUT, f"{task}_{obj}_s{seed}"); os.makedirs(d, exist_ok=True)
    if os.path.exists(os.path.join(d, "result.json")) and "--force" not in sys.argv: print("exists", d); sys.exit(0)
    params, wires, log, secs = train(task, obj, seed, budget)
    r, net, x0 = evaluate_run(task, params, wires, seed)
    dlca.export(os.path.join(d, "circuit.logic.json"), net, TASKS[task][0], TASKS[task][1], init_state=x0, steps=128, meta=dict(task=task, objective=obj, seed=seed, budget=budget))
    np.savez_compressed(os.path.join(d, "params.npz"), **{f"{n}_{i}": np.asarray(g) for n in ("perceive", "update") for i, g in enumerate(params[n])})
    res = dict(task=task, objective=obj, seed=seed, budget=budget, train_seconds=round(secs, 1), backend=jax.default_backend(), final=log[-1], **r)
    json.dump(res, open(os.path.join(d, "result.json"), "w"), indent=1); open(os.path.join(d, "log.jsonl"), "w").write("\n".join(json.dumps(x) for x in log) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "diverge_step"}))
