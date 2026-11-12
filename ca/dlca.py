"""Path D: Differentiable Logic CA, a faithful JAX port of the reference notebook
(google-research/self-organising-systems notebooks/diffLogic_CA.ipynb @3d5547ca, Apache-2.0; Miotti et al. 2025,
arXiv:2506.04912), plus an independent NumPy hard-circuit executor and a portable export ("badca/logic-ca-v1").

Faithful parts (same order of PRNG splits, so a given seed gives the notebook's wiring):
  get_unique_connections / get_moore_connections, pass-through init (logit 10 on op 3), soft = softmax mixture of the
  16 real-valued gates, hard = argmax one-hot, perception = n_kernels small gate trees applied to every channel's 3x3
  patch with kernel-specific logits and shared wiring, update input = [centre cell (C)] + perception outputs ordered
  (channel, kernel-output, kernel), synchronous update (async: per-cell fire mask), loss = summed squared error on the
  supervised channels, optax.chain(clip(100), adamw(lr, 0.9, 0.99, wd 1e-2)) re-implemented in ca.difflogic.adamw.
Differences are listed in docs/path-d/REFERENCES.md.

The older ca/difflogic.py (Campaigns 02-07, SAT-family inputs) is left unchanged.
"""
import json
import numpy as np

N_OPS, PASS, PASS_VALUE = 16, 3, 10.0
OP_NAMES = ["0", "a&b", "a&~b", "a", "~a&b", "b", "a^b", "a|b", "~(a|b)", "~(a^b)", "~b", "a|~b", "~a", "~a|b", "~(a&b)", "1"]
# truth table of op i as a 4-bit integer indexed by (a << 1) | b, derived from the real-valued relaxation at {0,1}
OP_TT = []
for _i in range(16):
    _t = 0
    for _a in (0, 1):
        for _b in (0, 1):
            _r = [0, _a * _b, _a - _a * _b, _a, _b - _a * _b, _b, _a + _b - 2 * _a * _b, _a + _b - _a * _b, 1 - (_a + _b - _a * _b),
                  1 - (_a + _b - 2 * _a * _b), 1 - _b, 1 - _b + _a * _b, 1 - _a, 1 - _a + _a * _b, 1 - _a * _b, 1][_i]
            _t |= int(_r) << ((_a << 1) | _b)
    OP_TT.append(_t)


# ------------------------------------------------------------------ JAX model (training) ------------------------------------------------------------------
def _jax():
    import jax, jax.numpy as jnp
    return jax, jnp


def get_moore_connections(key):
    jax, jnp = _jax()
    nb = jnp.array([0, 1, 2, 3, 5, 6, 7, 8]); perm = jax.random.permutation(key, 8)
    return nb[perm], jnp.full_like(nb, 4)[perm]


def get_unique_connections(in_dim, out_dim, key):
    jax, jnp = _jax()
    assert out_dim * 2 >= in_dim
    x = jnp.arange(in_dim); a, b = x[::2], x[1::2]; m = min(a.shape[0], b.shape[0]); a, b = a[:m], b[:m]
    if a.shape[0] < out_dim:
        a_, b_ = x[1::2], x[2::2]; m = min(a_.shape[0], b_.shape[0]); a = jnp.concatenate([a, a_[:m]]); b = jnp.concatenate([b, b_[:m]])
    off = 2
    while out_dim > a.shape[0] and off < in_dim:
        a = jnp.concatenate([a, x[:-off]]); b = jnp.concatenate([b, x[off:]]); off += 1
    if a.shape[0] < out_dim: raise ValueError("not enough unique connections")
    a, b = a[:out_dim], b[:out_dim]; perm = jax.random.permutation(key, out_dim)
    return a[perm], b[perm]


def _layer(key, in_dim, out_dim, conn):
    jax, jnp = _jax()
    if conn == "random":
        k1, k2 = jax.random.split(key); c = jax.random.permutation(k2, 2 * out_dim) % in_dim; c = jax.random.permutation(k1, in_dim)[c].reshape(2, out_dim); w = (c[0], c[1])
    elif conn == "unique": w = get_unique_connections(in_dim, out_dim, key)
    elif conn == "first_kernel": w = get_moore_connections(key)
    else: raise ValueError(conn)
    return jnp.zeros((out_dim, N_OPS)).at[:, PASS].set(PASS_VALUE), w


def init(hp, seed):
    """hp: {'channels', 'perceive': {'n_kernels', 'layers', 'connections'}, 'update': {'layers', 'connections'}}.
    Mirrors init_state + init_diff_logic_ca of the notebook. Returns (params, wires) with integer numpy wires."""
    jax, jnp = _jax()
    key = jax.random.PRNGKey(seed); key, sub = jax.random.split(key, 2)
    params, wires = {"update": [], "perceive": []}, {"update": [], "perceive": []}
    k, s = jax.random.split(sub); ku = s; key2 = k   # init_diff_logic_ca: key, subkey = split(key) -> update uses subkey
    kk = ku
    L = hp["update"]["layers"]
    for i, (a, b) in enumerate(zip(L[:-1], L[1:])):
        kk, s2 = jax.random.split(kk); g, w = _layer(s2, int(a), int(b), hp["update"]["connections"][i]); params["update"].append(g); wires["update"].append(w)
    key2, s3 = jax.random.split(key2); kk = s3; P = hp["perceive"]["layers"]; K = hp["perceive"]["n_kernels"]
    for i, (a, b) in enumerate(zip(P[:-1], P[1:])):
        kk, s2 = jax.random.split(kk); g, w = _layer(s2, int(a), int(b), hp["perceive"]["connections"][i])
        params["perceive"].append(g.repeat(K, axis=0).reshape(K, int(b), N_OPS)); wires["perceive"].append(w)
    wires = {n: [(np.asarray(a), np.asarray(b)) for a, b in ws] for n, ws in wires.items()}
    return params, wires


def _ops(a, b):
    jax, jnp = _jax(); ab = a * b
    return jnp.stack([jnp.zeros_like(a), ab, a - ab, a, b - ab, b, a + b - 2 * ab, a + b - ab, 1 - (a + b - ab), 1 - (a + b - 2 * ab),
                      1 - b, 1 - b + ab, 1 - a, 1 - a + ab, 1 - ab, jnp.ones_like(a)], -1)


def _run_layer(logits, w, x, hard):
    jax, jnp = _jax()
    wts = jax.nn.one_hot(jnp.argmax(logits, -1), N_OPS) if hard else jax.nn.softmax(logits, -1)
    return (_ops(x[..., w[0]], x[..., w[1]]) * wts).sum(-1)


def circuit(params, wires, patch, hard):
    """patch: (..., 9, C) -> (..., C). Notebook run_circuit, batched over leading axes."""
    jax, jnp = _jax()
    x = jnp.swapaxes(patch, -1, -2)                                  # (..., C, 9)
    outs = []
    for k in range(params["perceive"][0].shape[0]):
        y = x
        for g, w in zip(params["perceive"], wires["perceive"]): y = _run_layer(g[k], w, y, hard)
        outs.append(y)                                               # (..., C, s)
    per = jnp.stack(outs, -1)                                        # (..., C, s, K)  -> 'k c s -> (c s k)'
    per = per.reshape(per.shape[:-3] + (-1,))
    y = jnp.concatenate([patch[..., 4, :], per], -1)
    for g, w in zip(params["update"], wires["update"]): y = _run_layer(g, w, y, hard)
    return y


def patches(grid, periodic):
    """grid (..., H, W, C) -> (..., H, W, 9, C), row-major 3x3, centre index 4."""
    jax, jnp = _jax()
    pad = [(0, 0)] * (grid.ndim - 3) + [(1, 1), (1, 1), (0, 0)]
    p = jnp.pad(grid, pad, mode="wrap") if periodic else jnp.pad(grid, pad)
    H, W = grid.shape[-3], grid.shape[-2]
    return jnp.stack([p[..., dy:dy + H, dx:dx + W, :] for dy in range(3) for dx in range(3)], -2)


def step(params, wires, grid, hard, periodic, key=None, fire_rate=None):
    jax, jnp = _jax()
    new = circuit(params, wires, patches(grid, periodic), hard)
    if fire_rate is None: return new
    m = (jax.random.uniform(key, new[..., :1].shape) <= fire_rate).astype(new.dtype)
    return grid * (1 - m) + new * m


def rollout(params, wires, grid, n, hard, periodic, key=None, fire_rate=None, keep=False):
    """n synchronous (or async) steps. keep=True returns every state (n, ...)."""
    jax, jnp = _jax()
    def body(c, _):
        g, k = c
        if fire_rate is None: g = step(params, wires, g, hard, periodic)
        else: k, s = jax.random.split(k); g = step(params, wires, g, hard, periodic, s, fire_rate)
        return (g, k), (g if keep else None)
    (g, _), seq = jax.lax.scan(body, (grid, key if key is not None else jax.random.PRNGKey(0)), None, length=n)
    return seq if keep else g


# ------------------------------------------------------------------ hard circuit: export + NumPy executor ------------------------------------------------------------------
def harden(params, wires):
    """Hard circuit as plain lists: perceive [{'wires': [[a, b]...], 'ops': [[op...] per kernel]}], update [{'wires', 'ops'}]."""
    per = [dict(wires=np.stack(w, 1).tolist(), ops=np.asarray(g).argmax(-1).tolist()) for g, w in zip(params["perceive"], wires["perceive"])]
    upd = [dict(wires=np.stack(w, 1).tolist(), ops=np.asarray(g).argmax(-1).tolist()) for g, w in zip(params["update"], wires["update"])]
    return dict(perceive=per, update=upd)


def to_netlist(circ, C):
    """Layered hard circuit -> flat netlist, no simplification: inputs are node p*C + c (patch position p row-major,
    centre p = 4, channel c); every gate is [op, a, b] reading earlier nodes; outputs[c] is the node of channel c."""
    gates = []; n_in = 9 * C
    def add(op, a, b): gates.append([int(op), int(a), int(b)]); return n_in + len(gates) - 1
    per = {}                                                          # (c, s, k) -> node
    K = len(circ["perceive"][0]["ops"])
    for c in range(C):
        for k in range(K):
            cur = [p * C + c for p in range(9)]
            for L in circ["perceive"]:
                cur = [add(op, cur[a], cur[b]) for op, (a, b) in zip(L["ops"][k], L["wires"])]
            for s_, nd in enumerate(cur): per[(c, s_, k)] = nd
    S = len(circ["perceive"][-1]["wires"])
    cur = [4 * C + c for c in range(C)] + [per[(c, s_, k)] for c in range(C) for s_ in range(S) for k in range(K)]
    for L in circ["update"]: cur = [add(op, cur[a], cur[b]) for op, (a, b) in zip(L["ops"], L["wires"])]
    return dict(n_in=n_in, gates=gates, outputs=cur)


def net_eval(net, P):
    """Flat netlist on patches P (..., 9, C) -> (..., C). Every gate is evaluated (used for exact checks)."""
    C = P.shape[-1]; vals = [P[..., i // C, i % C] for i in range(net["n_in"])]
    for op, a, b in net["gates"]: vals.append(_gate(op, vals[a], vals[b]))
    return np.stack([vals[o] for o in net["outputs"]], -1)


def net_rollout(net, g, n, periodic):
    out = [np.asarray(g, np.uint8)]
    for _ in range(n): out.append(net_eval(net, np_patches(out[-1], periodic)))
    return np.stack(out)


def export(path, net, channels, periodic, init_state=None, steps=None, meta=None):
    """badca/logic-ca-v1 (synchronous). Replays from the file alone: no training code, no target frames.
    init: hex of numpy.packbits (big-endian bit order) of the (H, W, C) uint8 state in C order."""
    d = dict(format="badca/logic-ca-v1", channels=channels, boundary="periodic" if periodic else "zero", update="sync",
             neighbourhood="3x3 Moore incl. centre; node p*C + c = patch position p (row-major, centre 4), channel c",
             ops=OP_NAMES, op_truth_tables=OP_TT, op_index="truth table bit (a << 1) | b",
             net=net, meta=meta or {})
    if init_state is not None:
        s = np.asarray(init_state, np.uint8); d["H"], d["W"] = int(s.shape[0]), int(s.shape[1]); d["init"] = np.packbits(s.ravel()).tobytes().hex()
    if steps is not None: d["steps"] = int(steps)
    json.dump(d, open(path, "w"), separators=(",", ":"))
    return d


def load(path):
    d = json.load(open(path)); assert d["format"] == "badca/logic-ca-v1", d["format"]
    if "init" in d:
        n = d["H"] * d["W"] * d["channels"]; d["init_state"] = np.unpackbits(np.frombuffer(bytes.fromhex(d["init"]), np.uint8))[:n].reshape(d["H"], d["W"], d["channels"])
    return d


def _gate(op, a, b):
    t = OP_TT[op]; return (((t >> ((a.astype(np.uint8) << 1) | b.astype(np.uint8))) & 1)).astype(np.uint8)


def hard_circuit(circ, P):
    """P: (..., 9, C) uint8 bits -> (..., C) uint8. Pure NumPy, independent of the JAX code."""
    x = np.swapaxes(P, -1, -2); outs = []                            # (..., C, 9)
    K = len(circ["perceive"][0]["ops"])
    for k in range(K):
        y = x
        for L in circ["perceive"]:
            w = np.asarray(L["wires"]); y = np.stack([_gate(op, y[..., a], y[..., b]) for op, (a, b) in zip(L["ops"][k], w)], -1)
        outs.append(y)
    per = np.stack(outs, -1); per = per.reshape(per.shape[:-3] + (-1,))
    y = np.concatenate([P[..., 4, :], per], -1)
    for L in circ["update"]:
        y = np.stack([_gate(op, y[..., a], y[..., b]) for op, (a, b) in zip(L["ops"], L["wires"])], -1)
    return y


def np_patches(g, periodic):
    p = np.pad(g, ((1, 1), (1, 1), (0, 0)), mode="wrap" if periodic else "constant"); H, W = g.shape[:2]
    return np.stack([p[dy:dy + H, dx:dx + W] for dy in range(3) for dx in range(3)], -2)


def np_rollout(circ, g, n, periodic):
    """Hard synchronous rollout, NumPy. Returns (n+1, H, W, C) including the start."""
    out = [np.asarray(g, np.uint8)]
    for _ in range(n): out.append(hard_circuit(circ, np_patches(out[-1], periodic)))
    return np.stack(out)


def truth_table(circ, C, care=None):
    """Exhaustive local function over all 2^(9C) neighbourhoods (C <= 2) -> (2^(9C), C) uint8. Bit i of the index is
    patch[i // C... ] -- index = sum over (pos p, channel c) of bit << (p * C + c)."""
    n = 9 * C; assert n <= 20, "exhaustive table only for 9*C <= 20"
    idx = np.arange(2 ** n, dtype=np.int64); bits = ((idx[:, None] >> np.arange(n)) & 1).astype(np.uint8)
    out = np.zeros((2 ** n, C), np.uint8); B = 1 << 15
    for s in range(0, 2 ** n, B): out[s:s + B] = hard_circuit(circ, bits[s:s + B].reshape(-1, 9, C))
    return out


def gate_counts(circ):
    """Active (non pass-through) gates as in the paper's count (ops other than 3 'a' and 5 'b'), and total gates."""
    ops = [o for L in circ["perceive"] for k in L["ops"] for o in k] + [o for L in circ["update"] for o in L["ops"]]
    return dict(active=int(sum(o not in (3, 5) for o in ops)), total=len(ops))
