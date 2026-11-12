"""Path D, stage D0: faithful reproduction of the reference notebook's Game of Life and synchronous checkerboard.
usage: pd_d0.py life SEED [EPOCHS] | checker SEED [EPOCHS]   (GPU: gpuq -L pd-d0 -G 1 ...)
Writes runs/path_d/d0/<task>_s<seed>/{result.json, circuit.logic.json, log.jsonl}."""
import json, os, sys, time
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path.insert(0, ROOT)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax, jax.numpy as jnp
from ca import dlca
from ca.difflogic import adamw

OUT = os.path.join(ROOT, "runs", "path_d", "d0")


def life_np(g):   # true Game of Life, periodic, on (..., H, W)
    n = sum(np.roll(np.roll(g, dy, -2), dx, -1) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)
    return ((n == 3) | ((g == 1) & (n == 2))).astype(np.uint8)


def checkerboard(n, size=2):
    y, x = np.meshgrid(np.arange(n), np.arange(n), indexing="ij"); return (((x // size) + (y // size)) % 2).astype(np.uint8)


HP = {
    "life": dict(seed=23, lr=0.05, batch=20, epochs=3000, steps=1, channels=1, periodic=True,
                 perceive=dict(n_kernels=16, layers=[9, 8, 4, 2, 1], connections=["first_kernel", "unique", "unique", "unique"]),
                 update_hidden=[128] * 16 + [128, 64, 32, 16, 8, 4, 2]),
    "checker": dict(seed=23, lr=0.05, batch=2, epochs=500, steps=20, channels=8, periodic=False, size=16,
                    perceive=dict(n_kernels=16, layers=[9, 8, 4, 2], connections=["first_kernel", "unique", "unique", "unique"]),
                    update_hidden=[256] * 10 + [128, 64, 32, 16, 8]),
}


def hparams(task):
    h = HP[task]; C = h["channels"]; n_in = h["perceive"]["n_kernels"] * C * h["perceive"]["layers"][-1] + C
    layers = [n_in] + h["update_hidden"] + [C]
    return dict(channels=C, perceive=h["perceive"], update=dict(layers=layers, connections=["unique"] * len(layers)))


def train(task, seed, epochs):
    h = dict(HP[task]); hp = hparams(task); C = h["channels"]; params, wires = dlca.init(hp, seed)
    opt_init, opt_upd = adamw(h["lr"]); st = opt_init(params); key = jax.random.PRNGKey(seed)
    if task == "life":
        idx = np.arange(512); boards = ((idx[:, None] & (1 << np.arange(8, -1, -1))) > 0).astype(np.uint8).reshape(512, 3, 3)
        X_all, Y_all = jnp.asarray(boards[..., None], jnp.float32), jnp.asarray(life_np(boards)[..., None], jnp.float32)
    else:
        tgt = jnp.asarray(checkerboard(h["size"]), jnp.float32)

    def loss_fn(p, x, y, hard):
        out = dlca.rollout(p, wires, x, h["steps"], hard, h["periodic"]) if h["steps"] > 1 else dlca.step(p, wires, x, hard, h["periodic"])
        return jnp.square(out[..., 0] - y).sum()

    @jax.jit
    def tstep(p, st, x, y):
        l, g = jax.value_and_grad(loss_fn)(p, x, y, False); p, st = opt_upd(g, st, p)
        return p, st, l, loss_fn(p, x, y, True)

    log = []; t0 = time.time()
    for ep in range(epochs):
        key, k = jax.random.split(key)
        if task == "life":
            i = jax.random.randint(k, (h["batch"],), 0, 512); x, y = X_all[i], Y_all[i][..., 0]
        else:
            x = jax.random.randint(k, (h["batch"], h["size"], h["size"], C), 0, 2).astype(jnp.float32); y = jnp.broadcast_to(tgt, (h["batch"],) + tgt.shape)
        params, st, l, lh = tstep(params, st, x, y)
        if ep % 100 == 0 or ep == epochs - 1: log.append(dict(epoch=ep, soft=float(l), hard=float(lh), t=round(time.time() - t0, 1)))
    jax.block_until_ready(params)
    return params, wires, hp, log, time.time() - t0


def evaluate(task, params, wires, hp, seed):
    h = HP[task]; C = h["channels"]; circ = dlca.harden(params, wires); r = dict(gates=dlca.gate_counts(circ)); rng = np.random.default_rng(10_000 + seed)
    if task == "life":
        tt = dlca.truth_table(circ, 1)[:, 0]; idx = np.arange(512); bits = ((idx[:, None] >> np.arange(9)) & 1).reshape(512, 3, 3)
        ref = life_np(np.stack([np.tile(b, (3, 3)) for b in bits]))[:, 4, 4]   # centre of a 9x9 tiling = the 3x3 rule on that patch
        n = bits.reshape(512, 9).sum(1) - bits.reshape(512, 9)[:, 4]; ref2 = ((n == 3) | ((bits.reshape(512, 9)[:, 4] == 1) & (n == 2))).astype(np.uint8)
        assert (ref == ref2).all()
        r["truth_table_exact"] = int((tt == ref2).sum()); r["truth_table_wrong_entries"] = [int(i) for i in np.nonzero(tt != ref2)[0][:20]]
        g0 = rng.integers(0, 2, (128, 128, 1)).astype(np.uint8); B = dlca.np_rollout(circ, g0, 300, True)
        T = [g0[..., 0]]
        for _ in range(300): T.append(life_np(T[-1]))
        T = np.stack(T); r["rollout_128x128_300"] = dict(mismatched_cells_total=int((B[..., 0] != T).sum()), first_mismatch_step=next((t for t in range(301) if (B[t, ..., 0] != T[t]).any()), None))
        J = np.asarray(dlca.rollout(params, wires, jnp.asarray(g0, jnp.float32), 50, True, True, keep=True)).round().astype(np.uint8)
        r["jax_hard_vs_numpy_50_steps_equal"] = bool((J == B[1:51]).all())
        init = g0
    else:
        n = h["size"]; tgt = checkerboard(n); ok = []; ok64 = []
        for _ in range(100):
            g0 = rng.integers(0, 2, (n, n, C)).astype(np.uint8); B = dlca.np_rollout(circ, g0, h["steps"], False); ok.append(bool((B[-1, ..., 0] == tgt).all()))
        r["unseen_16x16_exact_at_step20"] = f"{sum(ok)}/100"
        for _ in range(10):
            g0 = rng.integers(0, 2, (4 * n, 4 * n, C)).astype(np.uint8); B = dlca.np_rollout(circ, g0, 4 * h["steps"], False)
            ok64.append(float((B[-1, ..., 0] != checkerboard(4 * n)).mean()))
        r["unseen_64x64_step80_pixel_err"] = ok64
        # does the pattern hold after it forms? steps 20..100 on 16x16
        g0 = rng.integers(0, 2, (n, n, C)).astype(np.uint8); B = dlca.np_rollout(circ, g0, 100, False)
        r["stays_exact_steps_20_to_100"] = bool(all((B[t, ..., 0] == tgt).all() for t in range(20, 101)))
        J = np.asarray(dlca.rollout(params, wires, jnp.asarray(g0, jnp.float32), 30, True, False, keep=True)).round().astype(np.uint8)
        r["jax_hard_vs_numpy_30_steps_equal"] = bool((J == B[1:31]).all())
        init = g0
    return r, circ, init


if __name__ == "__main__":
    task, seed = sys.argv[1], int(sys.argv[2]); epochs = int(sys.argv[3]) if len(sys.argv) > 3 else HP[task]["epochs"]
    d = os.path.join(OUT, f"{task}_s{seed}" + ("" if epochs == HP[task]["epochs"] else f"_e{epochs}")); os.makedirs(d, exist_ok=True)
    params, wires, hp, log, secs = train(task, seed, epochs)
    r, circ, init = evaluate(task, params, wires, hp, seed)
    net = dlca.to_netlist(circ, HP[task]["channels"]); P = np.random.default_rng(1).integers(0, 2, (4096, 9, HP[task]["channels"])).astype(np.uint8)
    assert (dlca.net_eval(net, P) == dlca.hard_circuit(circ, P)).all(), "netlist != layered circuit"
    dlca.export(os.path.join(d, "circuit.logic.json"), net, HP[task]["channels"], HP[task]["periodic"], init_state=init, steps=300 if task == "life" else 100,
                meta=dict(task=task, seed=seed, epochs=epochs, source="pd_d0.py (faithful notebook port)"))
    np.savez_compressed(os.path.join(d, "params.npz"), **{f"{n}_{i}": np.asarray(g) for n in ("perceive", "update") for i, g in enumerate(params[n])})
    res = dict(task=task, seed=seed, epochs=epochs, train_seconds=round(secs, 1), backend=jax.default_backend(), final=log[-1], **r)
    json.dump(res, open(os.path.join(d, "result.json"), "w"), indent=1); open(os.path.join(d, "log.jsonl"), "w").write("\n".join(json.dumps(x) for x in log) + "\n")
    print(json.dumps(res))
