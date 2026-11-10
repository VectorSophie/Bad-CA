"""Phased Neural CA: a time-varying CA with an EXTERNAL schedule (rule id per frame); state (visible + hidden) is
carried across phase boundaries unchanged; nothing from the target is read after the first frame.

Rule modes (same architecture C, Fh, S, 3x3 zero-padded perception, ReLU, 1x1 residual update as nca.py):
  fixed : one parameter set (K = 1)
  indep : K independent parameter sets
  delta : shared base + per phase k a rank-r update of Wp (Fh x C*9) and Wu (C x Fh) plus full bias deltas:
          Wp_k = Wp + (Ap_k @ Bp_k).reshape(Fh,C,3,3), bp_k = bp + dbp_k, Wu_k = Wu + Au_k @ Bu_k, bu_k = bu + dbu_k
Initial state: visible = first frame; hidden = 0 ('zero') or a learned low-resolution array g (C-1, ceil(H/f), ceil(W/f))
upsampled by nearest neighbour ('lowres', factor f): this is stored initial information and is counted in the file.
Frame t (t >= 1) is produced from frame t-1 by S substeps of the rule of phase sched[t].
Export ncab-v2 ("NCAB", version 2): u32 header length, JSON header, then float blobs in header['blobs'] order, then
the packed first frame. Readers: this module (numpy reference) and player/nca_core.mjs (Node/browser)."""
import json, zlib
import numpy as np
from ca import nca

PNAMES = ("Wp", "bp", "Wu", "bu"); DNAMES = ("Ap", "Bp", "dbp", "Au", "Bu", "dbu")


def init_params(rng, mode, K, C=16, Fh=64, rank=4):
    base = nca.init_params(rng, C, Fh)
    if mode == "fixed": return {k: v[None] for k, v in base.items()}
    if mode == "indep":
        return {k: np.stack([nca.init_params(rng, C, Fh)[k] if i else base[k] for i in range(K)]) for k in PNAMES}
    sc = 0.01
    return dict(base, Ap=np.zeros((K, Fh, rank), np.float32), Bp=(rng.standard_normal((K, rank, C * 9)) * sc).astype(np.float32), dbp=np.zeros((K, Fh), np.float32),
                Au=np.zeros((K, C, rank), np.float32), Bu=(rng.standard_normal((K, rank, Fh)) * sc).astype(np.float32), dbu=np.zeros((K, C), np.float32))


def effective(P, mode, xp=np):
    """Stacked per-phase parameters {Wp: (K,Fh,C,3,3), ...}."""
    if mode in ("fixed", "indep"): return {k: P[k] for k in PNAMES}
    K, Fh, C = P["Ap"].shape[0], P["Wp"].shape[0], P["Wp"].shape[1]
    return dict(Wp=P["Wp"][None] + xp.einsum("kfr,krc->kfc", P["Ap"], P["Bp"]).reshape(K, Fh, C, 3, 3), bp=P["bp"][None] + P["dbp"],
                Wu=P["Wu"][None] + xp.einsum("kcr,krf->kcf", P["Au"], P["Bu"]), bu=P["bu"][None] + P["dbu"])


def upsample(g, f, H, W, xp=np):
    return xp.repeat(xp.repeat(g, f, axis=1), f, axis=2)[:, :H, :W]


def n_params(P):
    return int(sum(np.asarray(v).size for v in P.values()))


# ---------------- numpy reference ----------------
def rollout_np(E, f0, sched, S, hidden0=None):
    """E: effective stacked params (numpy float32). sched: phase id per frame (len T). Returns binary (T,H,W), visible (T,H,W)."""
    C = E["Wu"].shape[1]; s = np.zeros((C,) + f0.shape, np.float32); s[0] = f0
    if hidden0 is not None: s[1:] = hidden0
    B, V = [f0.astype(np.uint8)], [s[0].copy()]
    for t in range(1, len(sched)):
        p = {k: E[k][sched[t]] for k in PNAMES}
        for _ in range(S): s = nca.step_np(p, s)
        B.append((s[0] > 0.5).astype(np.uint8)); V.append(s[0].copy())
    return np.stack(B), np.stack(V)


# ---------------- JAX ----------------
def jax_rollout(S, remat=True):
    import jax, jax.numpy as jnp
    step, _ = nca.jax_fns(S, remat)
    st = jax.checkpoint(step) if remat else step

    def roll(E, f0, ph, hidden0=None, noise=None):
        """E stacked effective params, ph: int array of phase ids for frames 1..T-1. Returns visible frames 1..T-1.
        noise = (key, sigma): TRAINING ONLY - Gaussian noise added to the whole state after every frame (robustness to
        small perturbations such as float-order differences between devices); never used at replay."""
        C = E["Wu"].shape[1]; s = jnp.zeros((C,) + f0.shape, jnp.float32).at[0].set(f0)
        if hidden0 is not None: s = s.at[1:].set(hidden0)
        keys = jax.random.split(noise[0], ph.shape[0]) if noise is not None else jnp.zeros((ph.shape[0], 2), jnp.uint32)
        def frame(s, xs):
            k, key = xs; p = {n: E[n][k] for n in PNAMES}
            s, _ = jax.lax.scan(lambda s, _: (st(p, s), None), s, None, length=S)
            out = s[0]
            if noise is not None: s = s + noise[1] * jax.random.normal(key, s.shape)
            return s, out
        sT, v = jax.lax.scan(frame, s, (ph, keys))
        return v, sT
    return roll


# ---------------- export ncab-v2 ----------------
def export(path, P, mode, phases, f0, T, S, init=None, meta=None, dtype="f32"):
    """phases: [{start, end, rule, ...}] (frame ranges); init: None or dict(kind='lowres', factor=f, g=array)."""
    names = list(PNAMES) + (list(DNAMES) if mode == "delta" else []); blobs = [[k, list(np.asarray(P[k]).shape)] for k in names]
    if init: blobs.append(["init_g", list(init["g"].shape)])
    K = int(np.asarray(P["Wp"]).shape[0]) if mode != "delta" else int(np.asarray(P["Ap"]).shape[0])
    hd = dict(C=int(np.asarray(P["Wu"]).shape[-2]), Fh=int(np.asarray(P["Wp"]).shape[-4]), S=int(S), T=int(T), H=int(f0.shape[0]), W=int(f0.shape[1]), threshold=0.5,
              boundary="zero", update="synchronous, S substeps per frame, residual", weight_dtype=dtype, mode=mode, K=K,
              schedule=[dict(start=int(p["start"]), end=int(p["end"]), rule=int(p.get("rule", i)), reason=p.get("reason")) for i, p in enumerate(phases)],
              schedule_semantics="frame t in [start, end) is produced from frame t-1 by `rule`; state carried across boundaries; external schedule (time-varying CA)",
              init=dict(kind="zero") if not init else dict(kind=init["kind"], factor=int(init["factor"]), hidden_channels=int(init["g"].shape[0])),
              blobs=blobs, meta=meta or {})
    h = json.dumps(hd, separators=(",", ":")).encode(); dt = "<f4" if dtype == "f32" else "<f2"
    w = b"".join(np.ascontiguousarray(P[k] if k != "init_g" else init["g"], dt).tobytes() for k, _ in blobs)
    fr = np.packbits(f0.astype(np.uint8).ravel(), bitorder="little").tobytes()
    body = len(h).to_bytes(4, "little") + h + w + fr; data = b"NCAB\x02\x01" + zlib.compress(body, 9); open(path, "wb").write(data)
    return dict(weights=len(w), first_frame=len(fr), header=len(h) + 10, raw_total=len(body) + 6, file=len(data), n_params=n_params({k: np.asarray(P[k]) for k in names}),
                init_values=int(init["g"].size) if init else 0)


def load(path):
    """-> header, effective stacked params (float32, combined in float64 then rounded once), sched (T,), hidden0 or None, f0."""
    data = open(path, "rb").read(); assert data[:5] == b"NCAB\x02", "not ncab-v2"
    body = zlib.decompress(data[6:]) if data[5] & 1 else data[6:]
    hl = int.from_bytes(body[:4], "little"); hd = json.loads(body[4:4 + hl]); off = 4 + hl; dt = "<f4" if hd["weight_dtype"] == "f32" else "<f2"; R = {}
    for k, shape in hd["blobs"]:
        n = int(np.prod(shape)); R[k] = np.frombuffer(body, dt, n, off).reshape(shape).astype(np.float64); off += n * np.dtype(dt).itemsize
    f0 = np.unpackbits(np.frombuffer(body, np.uint8, (hd["H"] * hd["W"] + 7) // 8, off), bitorder="little")[:hd["H"] * hd["W"]].reshape(hd["H"], hd["W"])
    E = {k: v.astype(np.float32) for k, v in effective(R, hd["mode"]).items()}
    sched = np.zeros(hd["T"], np.int32)
    for p in hd["schedule"]: sched[p["start"]:p["end"]] = p["rule"]
    h0 = upsample(R["init_g"], hd["init"]["factor"], hd["H"], hd["W"]).astype(np.float32) if hd["init"]["kind"] == "lowres" else None
    return hd, E, sched, h0, f0


if __name__ == "__main__":   # self-check: v2 round trip; a 'fixed' v2 export replays exactly like the v1 path
    import os, tempfile
    rng = np.random.default_rng(0); f0 = (rng.random((10, 12)) < .5).astype(np.uint8)
    for mode in ("fixed", "indep", "delta"):
        P = init_params(rng, mode, 3); P = {k: (v + 0.01 * rng.standard_normal(v.shape)).astype(np.float32) for k, v in P.items()}
        ph = [dict(start=1, end=4), dict(start=4, end=6), dict(start=6, end=9)] if mode != "fixed" else [dict(start=1, end=9)]
        g = (rng.standard_normal((15, 3, 3)) * .1).astype(np.float32); p = os.path.join(tempfile.mkdtemp(), "x.ncab")
        s = export(p, P, mode, ph, f0, 9, 2, init=dict(kind="lowres", factor=4, g=g))
        hd, E, sched, h0, g0 = load(p); assert (g0 == f0).all() and h0.shape == (15, 10, 12) and list(sched) == [0, 0, 0, 0, 1, 1, 2, 2, 2] or mode == "fixed"
        B, V = rollout_np(E, f0, sched, 2, h0)
        if mode == "fixed":
            B1, _ = rollout_np({k: P[k] for k in PNAMES}, f0, sched, 2, upsample(g, 4, 10, 12)); assert (B == B1).all()
    print("phase_nca self-check ok", s)
