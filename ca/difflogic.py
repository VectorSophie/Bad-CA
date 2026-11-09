"""Differentiable logic-gate CA, JAX. Architecture adapted from the DiffLogic CA reference
notebook (google-research/self-organising-systems notebooks/diffLogic_CA.ipynb @3d5547ca,
Apache-2.0; Miotti et al. 2025): centre-paired perception kernels + deep 'unique'-wired update
network, pass-through init (logit 10 on op 3), AdamW(b2=0.99, wd=1e-2), value clip 100.
Our own first attempt (shallow, random wiring) could not learn Conway; see CHECKPOINT_02.

Inputs = ca.rule.family_taps(k) order: visible 3x3 (0..8, centre 4) then hidden channels.
Perception kernels read only the visible 3x3; the update net additionally reads the visible
centre and every non-visible tap, so the represented family equals the SAT family.
Each layer's wiring indexes concat(raw inputs, previous layer) -> skip from raw inputs.
Hardening is exact: evaluate the hard circuit on all 2^n inputs -> LUT.
"""
import jax, jax.numpy as jnp
import numpy as np

PASS = 3


def ops(a, b):  # (..., 16) the 16 two-input Boolean functions, real-valued relaxation
    ab = a * b
    return jnp.stack([jnp.zeros_like(a), ab, a - ab, a, b - ab, b, a + b - 2 * ab, a + b - ab,
                      1 - (a + b - ab), 1 - (a + b - 2 * ab), 1 - b, 1 - b + ab, 1 - a, 1 - a + ab, 1 - ab,
                      jnp.ones_like(a)], -1)


def unique_pairs(in_dim, out_dim, rng):
    """Port of get_unique_connections (difflogic / reference notebook)."""
    x = np.arange(in_dim)
    a, b = list(x[::2][:in_dim // 2]), list(x[1::2][:in_dim // 2])
    if len(a) < out_dim:
        m = min(len(x[1::2]), len(x[2::2])); a += list(x[1::2][:m]); b += list(x[2::2][:m])
    off = 2
    while len(a) < out_dim and off < in_dim:
        a += list(x[:-off]); b += list(x[off:]); off += 1
    assert len(a) >= out_dim, "not enough unique connections"
    p = rng.permutation(out_dim)
    return np.stack([np.array(a[:out_dim])[p], np.array(b[:out_dim])[p]], 1)


def make_wiring(rng, n_in, n_out, n_kernels=16, update=(128,) * 16 + (128, 64, 32, 16, 8, 4)):
    """Returns list of (width,2) index arrays; index < n_in = raw input, else previous layer."""
    W = []
    # perception layer 1: each kernel pairs every neighbour with the centre (bit 4)
    l1 = []
    for _ in range(n_kernels):
        nb = rng.permutation([0, 1, 2, 3, 5, 6, 7, 8])
        l1 += [(j, 4) for j in nb]
    W.append(np.array(l1))
    width = 8
    while width > 1:  # within-kernel reduction 8 -> 4 -> 2 -> 1
        nxt = []
        for kk in range(n_kernels):
            base = n_in + kk * width
            nxt += [(base + 2 * i, base + 2 * i + 1) for i in range(width // 2)]
        W.append(np.array(nxt)); width //= 2
    # update layer 1 reads kernel outputs + visible centre + all non-visible taps
    feats = [n_in + kk for kk in range(n_kernels)] + [4] + list(range(9, n_in))
    widths = list(update) + [n_out]
    pr = unique_pairs(len(feats), widths[0], rng)
    W.append(np.array(feats)[pr])
    for prev, w in zip(widths[:-1], widths[1:]):
        W.append(unique_pairs(prev, w, rng) + n_in)
    return W


def init_logits(wiring, pass_value=10.0):
    return [jnp.zeros((len(w), 16)).at[:, PASS].set(pass_value) for w in wiring]


def net(logits, wiring, x, hard=False, ste=False):
    """x: (..., n_in) in [0,1] -> (..., n_out).
    hard: one-hot argmax gates. ste: straight-through -- hard one-hot forward, softmax gradient."""
    raw, prev = x, None
    for l, w in zip(logits, wiring):
        pool = raw if prev is None else jnp.concatenate([raw, prev], -1)
        o = ops(pool[..., w[:, 0]], pool[..., w[:, 1]])
        oh = jax.nn.one_hot(jnp.argmax(l, -1), 16)
        sm = jax.nn.softmax(l, -1)
        wts = oh if hard else (sm + jax.lax.stop_gradient(oh - sm) if ste else sm)
        prev = (o * wts).sum(-1)
    return prev


def ca_inputs(state, taps, boundary=0.0):
    C, H, W = state.shape
    p = jnp.pad(state, ((0, 0), (1, 1), (1, 1)), constant_values=boundary)
    return jnp.stack([p[c, 1 + dy:1 + dy + H, 1 + dx:1 + dx + W] for c, dy, dx in taps], -1)  # (H,W,n)


def _state(s, mode):
    """Per-generation state transform. soft: none. round/round_hidden: binarize (all / hidden
    channels; no gradient). ste/ste_hidden: binarize forward, identity gradient."""
    if mode == "soft": return s
    r = jnp.round(s)
    if mode == "round": return r
    if mode == "round_hidden": return s.at[1:].set(r[1:])
    st = s + jax.lax.stop_gradient(r - s)
    if mode == "ste": return st
    if mode == "ste_hidden": return s.at[1:].set(st[1:])
    raise ValueError(mode)


def soft_rollout(logits, wiring, taps, vis0, channels, T, S, hard=False, gate_ste=False, state="soft", full=False, gens=False, remat=False):
    """Returns visible at frames 1..T-1, or with full=True the whole state (T-1, C, H, W), or with
    gens=True every generation's whole state (T-1, S, C, H, W)."""
    s0 = jnp.zeros((channels,) + vis0.shape).at[0].set(vis0)
    def gen(s, _):
        s = _state(jnp.moveaxis(net(logits, wiring, ca_inputs(s, taps), hard, gate_ste), -1, 0), state)
        return s, s
    if remat:  # gradient checkpointing: store only states, recompute each generation's activations in backward
        gen = jax.checkpoint(gen)
    def frame(s, _):
        s, ss = jax.lax.scan(gen, s, None, length=S)
        return s, (ss if gens else (s if full else s[0]))
    _, vis = jax.lax.scan(frame, s0, None, length=T - 1)
    return vis  # frames 1..T-1


def hard_lut(logits, wiring, n_in):
    idx = np.arange(2 ** n_in)
    x = jnp.asarray(((idx[:, None] >> np.arange(n_in)) & 1).astype(np.float32))
    return np.asarray(net(logits, wiring, x, hard=True)).round().astype(np.uint8).T  # (C, 2^n)


def live_gates(logits, wiring, n_in):
    """Non-wire, non-constant gates reachable backwards from the outputs (pure wires cost 0)."""
    hard = [np.asarray(jnp.argmax(l, -1)) for l in logits]
    need = set(range(len(wiring[-1]))); count = 0
    for li in range(len(wiring) - 1, -1, -1):
        nxt = set()
        for g in need:
            op = hard[li][g]
            if op in (0, 15): continue
            count += op not in (3, 5)
            for src, used in ((wiring[li][g, 0], op not in (5, 10)), (wiring[li][g, 1], op not in (3, 12))):
                if used and src >= n_in: nxt.add(int(src - n_in))
        need = nxt
    return count


def adamw(lr, b1=0.9, b2=0.99, wd=1e-2, clip=100.0):
    """optax.chain(clip(100), adamw(lr, b1, b2, wd)) as used by the reference notebook."""
    def init(p):
        return (jax.tree.map(jnp.zeros_like, p), jax.tree.map(jnp.zeros_like, p), 0)
    def update(g, st, p):
        m, v, t = st; t += 1
        g = jax.tree.map(lambda g: jnp.clip(g, -clip, clip), g)
        m = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g, m, g)
        v = jax.tree.map(lambda v, g: b2 * v + (1 - b2) * g * g, v, g)
        p = jax.tree.map(lambda p, m, v: p - lr * ((m / (1 - b1 ** t)) / (jnp.sqrt(v / (1 - b2 ** t)) + 1e-8) + wd * p), p, m, v)
        return p, (m, v, t)
    return init, update


def extract_circuit(logits, wiring, n_in):
    """Hardened live sub-circuit in ca.costs form: gates [(truth-table op, a, b)] over the pool
    [inputs, gates...], outputs per channel. Pass-through ops (3, 5) are resolved to their source
    (wires cost nothing); identical (op, a, b) gates are merged (common subexpressions)."""
    from ca.costs import OP_TT
    hard = [np.asarray(jnp.argmax(l, -1)) for l in logits]
    wir = [np.asarray(w) for w in wiring]
    gates, memo, cse = [], {}, {}
    def src(l, i):  # pool index i of layer l -> node
        return ("in", i) if (l == 0 or i < n_in) else (l - 1, i - n_in)
    def resolve(node):
        if node[0] == "in": return node[1]
        if node in memo: return memo[node]
        l, g = node; op = int(hard[l][g]); a, b = wir[l][g]
        if op == 3: r = resolve(src(l, a))
        elif op == 5: r = resolve(src(l, b))
        else:
            ra = resolve(src(l, a)) if op not in (0, 5, 10, 15) else 0
            rb = resolve(src(l, b)) if op not in (0, 3, 12, 15) else 0
            key = (OP_TT[op], ra, rb)
            if key not in cse:
                gates.append(key); cse[key] = n_in + len(gates) - 1
            r = cse[key]
        memo[node] = r
        return r
    outputs = [resolve((len(wir) - 1, c)) for c in range(len(wir[-1]))]
    return gates, outputs
