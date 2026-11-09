"""Campaign 09 Neural CA (candidate ② 'local model with easy capacity'), deterministic, continuous state.

State s: (C, H, W) float32; channel 0 = visible, 1..C-1 = hidden. Init: visible = binary first frame, hidden = 0.
One step (shared everywhere, boundary = zero padding for every channel):
    p = relu(conv3x3(s; Wp (Fh, C, 3, 3)) + bp)          # learned 3x3 perception, Fh features
    s = s + (Wu (C, Fh) @ p + bu)                        # 1x1 residual update
S steps per observed frame. Output frame = (s[0] > 0.5) (binary visible, for every metric); the continuous state is
carried on unchanged (the deployed model IS this continuous system; nothing is binarised inside the rollout).
Receptive field per frame = S cells. Params = Fh*C*9 + Fh + C*Fh + C.
Two independent implementations: JAX (training, `rollout_jax`) and numpy float32 (`rollout_np`, reference replay
from an export); a third in player/nca_replay.mjs.
Forcing modes for diagnostics: force='tf' sets the visible channel to the true frame at every observed frame
(hidden carried) -> one-step teacher-forced prediction; force='tf_reset' also zeroes hidden every frame.
"""
import base64, json
import numpy as np


def n_params(C, Fh):
    return Fh * C * 9 + Fh + C * Fh + C


def init_params(rng, C, Fh):
    return dict(Wp=(rng.standard_normal((Fh, C, 3, 3)) * np.sqrt(2.0 / (C * 9))).astype(np.float32), bp=np.zeros(Fh, np.float32),
                Wu=np.zeros((C, Fh), np.float32), bu=np.zeros(C, np.float32))   # zero update: starts as 'hold'


# ---------------- numpy float32 reference ----------------
def _conv3_np(s, Wp, bp):
    C, H, W = s.shape; p = np.pad(s, ((0, 0), (1, 1), (1, 1)))
    out = np.zeros((Wp.shape[0], H, W), np.float32)
    for dy in range(3):
        for dx in range(3):
            out += np.einsum("fc,chw->fhw", Wp[:, :, dy, dx], p[:, dy:dy + H, dx:dx + W], dtype=np.float32)
    return out + bp[:, None, None]


def step_np(P, s):
    q = np.maximum(_conv3_np(s, P["Wp"], P["bp"]), 0)
    return (s + np.einsum("cf,fhw->chw", P["Wu"], q, dtype=np.float32) + P["bu"][:, None, None]).astype(np.float32)


def rollout_np(P, F, T, S, force=None, start=0):
    """Binary visible frames (T, H, W) from true frame F[start] (hidden 0); with force, visible := F[start+t] each frame."""
    C = P["Wu"].shape[0]; s = np.zeros((C,) + F[start].shape, np.float32); s[0] = F[start]; out = [F[start].astype(np.uint8)]; prob = [s[0].copy()]
    for t in range(1, T):
        if force:
            s[0] = F[start + t - 1]
            if force == "tf_reset": s[1:] = 0
        for _ in range(S): s = step_np(P, s)
        out.append((s[0] > 0.5).astype(np.uint8)); prob.append(s[0].copy())
    return np.stack(out), np.stack(prob)


# ---------------- JAX ----------------
def jax_fns(S, remat=True):
    import jax, jax.numpy as jnp
    def step(P, s):
        q = jax.nn.relu(jax.lax.conv_general_dilated(s[None], P["Wp"], (1, 1), "SAME", precision=jax.lax.Precision.HIGHEST)[0] + P["bp"][:, None, None])
        return s + jnp.einsum("cf,fhw->chw", P["Wu"], q, precision=jax.lax.Precision.HIGHEST) + P["bu"][:, None, None]
    st = jax.checkpoint(step) if remat else step
    def rollout(P, f0, T, hidden0=None):
        s = jnp.zeros((P["Wu"].shape[0],) + f0.shape, jnp.float32).at[0].set(f0)
        if hidden0 is not None: s = s.at[1:].set(hidden0)
        def frame(s, _):
            s, _ = jax.lax.scan(lambda s, _: (st(P, s), None), s, None, length=S); return s, s[0]
        _, v = jax.lax.scan(frame, s, None, length=T - 1)
        return v                                      # continuous visible at frames 1..T-1
    return step, rollout


# ---------------- export ----------------
def export(path, P, f0, T, S, meta):
    b64 = lambda a: base64.b64encode(np.ascontiguousarray(a, np.float32).tobytes()).decode()
    ex = dict(format="ca-universal/nca-v1", dtype="float32 little-endian", C=int(P["Wu"].shape[0]), Fh=int(P["Wp"].shape[0]), S=S, n_frames=T,
              height=int(f0.shape[0]), width=int(f0.shape[1]), boundary="zero padding, all channels", hidden_init=0.0, threshold=0.5,
              first_frame_b64=base64.b64encode(np.packbits(f0.astype(np.uint8).ravel(), bitorder="little").tobytes()).decode(),
              params={k: dict(shape=list(v.shape), b64=b64(v)) for k, v in P.items()},
              step="p = relu(conv3x3(s,Wp)+bp) [zero pad, kernel index (f,c,dy,dx), dy,dx = 0..2 for offsets -1..1]; s += Wu@p + bu",
              n_params=int(sum(v.size for v in P.values())), meta=meta)
    json.dump(ex, open(path, "w"))
    return ex


def load(path):
    ex = json.load(open(path))
    P = {k: np.frombuffer(base64.b64decode(v["b64"]), np.float32).reshape(v["shape"]).copy() for k, v in ex["params"].items()}
    f0 = np.unpackbits(np.frombuffer(base64.b64decode(ex["first_frame_b64"]), np.uint8), bitorder="little")[:ex["height"] * ex["width"]].reshape(ex["height"], ex["width"])
    return ex, P, f0


def known_mover(C=16, Fh=64):
    """Hand-set NCA in this exact architecture: hidden 1 = global clock h' = 1 - h; visible' = (h ? W : C) for binary
    states (moves right by 1 cell on every other step). Exact in float32 for 0/1 states."""
    P = dict(Wp=np.zeros((Fh, C, 3, 3), np.float32), bp=np.zeros(Fh, np.float32), Wu=np.zeros((C, Fh), np.float32), bu=np.zeros(C, np.float32))
    P["Wp"][0, 0, 1, 0] = 1; P["Wp"][0, 1, 1, 1] = 1; P["bp"][0] = -1      # f0 = relu(v_W + h - 1) = v_W AND h
    P["Wp"][1, 0, 1, 1] = 1; P["Wp"][1, 1, 1, 1] = -1                      # f1 = relu(v_C - h)      = v_C AND NOT h
    P["Wp"][2, 1, 1, 1] = -1; P["bp"][2] = 1                               # f2 = relu(1 - h)
    P["Wp"][3, 1, 1, 1] = 1                                                # f3 = relu(h)
    P["Wp"][4, 0, 1, 1] = 1                                                # f4 = relu(v_C)
    P["Wu"][0, 0] = 1; P["Wu"][0, 1] = 1; P["Wu"][0, 4] = -1               # dv = f0 + f1 - v
    P["Wu"][1, 2] = 1; P["Wu"][1, 3] = -1                                  # dh = (1 - h) - h
    return P
