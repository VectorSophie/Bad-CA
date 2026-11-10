"""ncab-v1: compact self-contained binary Neural CA export (layout documented in player/nca_core.mjs).
Weights stored as float32 or float16 (storage only: they are widened to float32 on load and the state/compute stay
float32). The whole body after the 6-byte prefix is zlib-deflated. `write` returns a byte breakdown:
weights / first frame / header (JSON + length + prefix) before deflate, and the actual file size."""
import json, zlib
import numpy as np

ORDER = ("Wp", "bp", "Wu", "bu")


def write(path, P, f0, T, S, meta=None, dtype="f32", threshold=0.5):
    hd = dict(C=int(P["Wu"].shape[0]), Fh=int(P["Wp"].shape[0]), S=int(S), T=int(T), H=int(f0.shape[0]), W=int(f0.shape[1]), threshold=threshold,
              hidden_init=0.0, boundary="zero", weight_dtype=dtype, params=[[k, list(P[k].shape)] for k in ORDER], meta=meta or {})
    h = json.dumps(hd, separators=(",", ":")).encode()
    w = b"".join(np.ascontiguousarray(P[k], "<f4" if dtype == "f32" else "<f2").tobytes() for k in ORDER)
    fr = np.packbits(f0.astype(np.uint8).ravel(), bitorder="little").tobytes()
    body = len(h).to_bytes(4, "little") + h + w + fr; data = b"NCAB\x01\x01" + zlib.compress(body, 9)
    open(path, "wb").write(data)
    return dict(weights=len(w), first_frame=len(fr), header=len(h) + 4 + 6, raw_total=len(body) + 6, file=len(data), n_params=int(sum(P[k].size for k in ORDER)))


def read(path):
    data = open(path, "rb").read(); assert data[:5] == b"NCAB\x01", "not ncab-v1"
    body = zlib.decompress(data[6:]) if data[5] & 1 else data[6:]
    hl = int.from_bytes(body[:4], "little"); hd = json.loads(body[4:4 + hl]); off = 4 + hl; P = {}
    dt = "<f4" if hd["weight_dtype"] == "f32" else "<f2"
    for k, shape in hd["params"]:
        n = int(np.prod(shape)); P[k] = np.frombuffer(body, dt, n, off).reshape(shape).astype(np.float32); off += n * np.dtype(dt).itemsize
    f0 = np.unpackbits(np.frombuffer(body, np.uint8, (hd["H"] * hd["W"] + 7) // 8, off), bitorder="little")[:hd["H"] * hd["W"]].reshape(hd["H"], hd["W"])
    return hd, P, f0


if __name__ == "__main__":   # self-check: round trip f32 exact, f16 = float16-rounded weights
    import os, tempfile, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from ca import nca
    P = nca.init_params(np.random.default_rng(0), 8, 16); P["Wu"] += 0.01; f0 = (np.random.default_rng(1).random((9, 13)) < .5).astype(np.uint8)
    d = tempfile.mkdtemp()
    for dt in ("f32", "f16"):
        s = write(os.path.join(d, "x.ncab"), P, f0, 5, 4, dict(a=1), dt); hd, Q, g = read(os.path.join(d, "x.ncab"))
        assert (g == f0).all() and hd["T"] == 5 and s["weights"] == 1304 * (4 if dt == "f32" else 2)
        for k in P: assert np.array_equal(Q[k], P[k] if dt == "f32" else P[k].astype(np.float16).astype(np.float32))
    print("ncab self-check ok", s)
