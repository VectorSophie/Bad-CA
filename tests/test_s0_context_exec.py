"""Stage 1 executor sanity (CPU, hand-written, NOT learned): can the existing badca/logic-ca-v1 executors run one fixed
Boolean circuit whose behaviour is selected by context bits, with no engine or format change?
Encoding: context = 2 extra channels that the circuit copies through unchanged (ch1 = c0, ch2 = c1), so the code is part
of the state. Modes: 00 hold, 01 move east (v' = W.v), 10 grow (v | N | S | E | W), 11 shrink (v & N & S & E & W).
1. The 17-gate circuit equals the reference rule of each mode on every one of the 2^11 relevant inputs.
2. On 20 unseen random blob starts per mode, Python replay equals the reference rule for 24 steps.
3. Mid-trajectory switch (east 8 steps -> grow 4 -> shrink 4): v-state carries across switches, Python == Node (SHA-256).
   The switch itself is done by re-exporting the carried state with new context bits: logic-ca-v1 has no event
   mechanism (its prefix only overwrites channel 0), so scheduled switching needs a format extension."""
import hashlib, json, os, subprocess, sys, tempfile
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from ca import dlca

C = 3; n_in = 9 * C; v = lambda p: p * C; c0, c1 = 4 * C + 1, 4 * C + 2
N_, W_, Cv, E_, S_ = v(1), v(3), v(4), v(5), v(7)
AND, OR, NOTA_AND_B = 1, 7, 4
gates = []
def g(op, a, b): gates.append([op, a, b]); return n_in + len(gates) - 1
grow = g(OR, Cv, g(OR, g(OR, N_, S_), g(OR, E_, W_)))
shrink = g(AND, Cv, g(AND, g(AND, N_, S_), g(AND, E_, W_)))
lo = g(OR, g(AND, c0, W_), g(NOTA_AND_B, c0, Cv))                # c0 ? W : C
hi = g(OR, g(AND, c0, shrink), g(NOTA_AND_B, c0, grow))          # c0 ? shrink : grow
out = g(OR, g(AND, c1, hi), g(NOTA_AND_B, c1, lo))               # c1 ? hi : lo
net = dict(n_in=n_in, gates=gates, outputs=[out, c0, c1]); assert len(gates) == 17


def ref(f, code):
    P = dlca.np_patches(f[..., None], False)[..., 0]
    return [f, P[..., 3], P[..., [1, 3, 4, 5, 7]].max(-1), P[..., [1, 3, 4, 5, 7]].min(-1)][code].astype(np.uint8)


# 1. exhaustive over the 5 visible inputs that matter (plus 4 more visible corners) and the 2 context bits
idx = np.arange(2 ** 11); bits = ((idx[:, None] >> np.arange(11)) & 1).astype(np.uint8)
P = np.zeros((len(idx), 9, C), np.uint8); P[:, :, 0] = bits[:, :9]; P[:, 4, 1] = bits[:, 9]; P[:, 4, 2] = bits[:, 10]
o = dlca.net_eval(net, P)
code = bits[:, 9] + 2 * bits[:, 10]; vv = bits[:, :9]
want = np.select([code == 0, code == 1, code == 2], [vv[:, 4], vv[:, 3], vv[:, [1, 3, 4, 5, 7]].max(1)], vv[:, [1, 3, 4, 5, 7]].min(1))
assert (o[:, 0] == want).all() and (o[:, 1] == bits[:, 9]).all() and (o[:, 2] == bits[:, 10]).all()

# 2. unseen starts
rng = np.random.default_rng(1)
for code in range(4):
    for _ in range(20):
        f = (rng.random((36, 48)) < 0.1).astype(np.uint8)
        f = (dlca.np_patches(f[..., None], False)[..., 0].sum(-1) >= 2).astype(np.uint8)
        s = np.zeros((36, 48, C), np.uint8); s[..., 0] = f; s[..., 1] = code & 1; s[..., 2] = code >> 1
        B = dlca.net_rollout(net, s, 24, False); r = f
        for t in range(24): r = ref(r, code); assert (B[t + 1, ..., 0] == r).all(), (code, t)

# 3. switching with carried state, Python == Node
f = np.zeros((36, 48), np.uint8); f[12:24, 6:16] = 1; f[15:20, 16:20] = 1
state = np.zeros((36, 48, C), np.uint8); state[..., 0] = f; r = f.copy()
with tempfile.TemporaryDirectory() as td:
    for code, n in ((1, 8), (2, 4), (3, 4)):
        state[..., 1], state[..., 2] = code & 1, code >> 1; p = os.path.join(td, f"seg{code}.logic.json")
        dlca.export(p, net, C, False, init_state=state, steps=n, meta=dict(note="hand-written context mux, Stage 1 sanity"))
        B = dlca.replay(dlca.load(p))
        js = json.loads(subprocess.run(["node", os.path.join(ROOT, "player/logic_replay.mjs"), p], capture_output=True, text=True, check=True).stdout)
        assert js["sha256"] == hashlib.sha256(B.tobytes()).hexdigest(), code
        for t in range(n): r = ref(r, code)
        assert (B[-1, ..., 0] == r).all(); state = B[-1].copy()
print("test_s0_context_exec: all checks passed (17-gate hand-written context mux; Python == Node)")
