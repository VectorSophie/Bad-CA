"""Campaign 05 file container: one decodable file per (clip, method); cost = file bytes.

Bitstream (MSB-first, zero-padded to whole bytes), Elias-gamma integers (ca.costs.BitW):
  gamma(method id) gamma(W) gamma(H) gamma(T)
  methods 1..5: first frame = 1 flag bit (0 raw W*H bits, 1 frame-RLE runs; the shorter is used)
                then the payload:
     1 ca_circuit   costs.rule_circuit bits (rule header: family, k, S, boundary, hidden init; gates)
     2 ca_lut_exc   costs.rule_lut_exceptions bits (rule header; keep default + exceptions)
     3 changelist / 4 xor_rle / 5 frame_rle   costs recording bits of frames 1..T-1
  methods 6..8 (zlib raw deflate 9 / bz2 9 / lzma raw preset 9e): 1 variant bit
     (0 = packed bits of all T frames, 1 = frame 0 + packed XOR deltas), pad to a byte, then the
     compressed bytes up to the end of the file (the first frame is inside the compressed stream).
Generic decoders (this module, the Node player, zlib/bz2/lzma) are not counted, for every method.
"""
import bz2, hashlib, lzma, zlib
import numpy as np
from ca import costs

METHODS = {"ca_circuit": 1, "ca_lut_exc": 2, "changelist": 3, "xor_rle": 4, "frame_rle": 5, "zlib": 6, "bz2": 7, "lzma": 8}
_NAME = {v: k for k, v in METHODS.items()}
_LZMA = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}]
_GP = {"zlib": (lambda b: (lambda c: c.compress(b) + c.flush())(zlib.compressobj(9, zlib.DEFLATED, -15)),
                lambda b: zlib.decompress(b, -15)),
       "bz2": (lambda b: bz2.compress(b, 9), bz2.decompress),
       "lzma": (lambda b: lzma.compress(b, lzma.FORMAT_RAW, filters=_LZMA),
                lambda b: lzma.decompress(b, lzma.FORMAT_RAW, filters=_LZMA))}


def sha(frames):
    return hashlib.sha256(np.ascontiguousarray(frames, np.uint8).tobytes()).hexdigest()


def _to_bytes(bits):
    return np.packbits(np.array(bits + [0] * (-len(bits) % 8), np.uint8)).tobytes()


def _bits(data):
    return np.unpackbits(np.frombuffer(data, np.uint8)).tolist()


def _first_frame(w, f0):
    raw = f0.ravel().tolist(); r = costs.BitW(); costs._runs_w(r, f0.ravel())
    if len(r) < len(raw):
        w.bits(1, 1); w.b += r.b
    else:
        w.bits(0, 1); w.b += raw


def encode(method, frames, rule=None):
    """rule: for ca_circuit dict(gates, outputs, family, k, S, boundary=0, hidden_init=0, n_in);
    for ca_lut_exc dict(luts, taps, family, k, S, boundary=0, hidden_init=0). Returns bytes."""
    T, H, W = frames.shape
    w = costs.BitW(); w.gamma(METHODS[method]); w.gamma(W); w.gamma(H); w.gamma(T)
    if method in _GP:
        enc = _GP[method][0]
        a = np.packbits(frames.ravel()).tobytes()
        d = frames.copy(); d[1:] ^= frames[:-1]; b = np.packbits(d.ravel()).tobytes()
        ca, cb = enc(a), enc(b)
        w.bits(int(len(cb) < len(ca)), 1)
        return _to_bytes(w.b) + (cb if len(cb) < len(ca) else ca)
    _first_frame(w, frames[0])
    if method == "ca_circuit":
        r = rule; p = costs.rule_circuit(r["gates"], r["outputs"], r["n_in"], r["family"], r["k"], r["S"], r.get("boundary", 0), r.get("hidden_init", 0))
    elif method == "ca_lut_exc":
        r = rule; p = costs.rule_lut_exceptions(r["luts"], r["taps"], r["family"], r["k"], r["S"], r.get("boundary", 0), r.get("hidden_init", 0))
    else:
        p = {"changelist": costs.changelist, "xor_rle": costs.xor_rle, "frame_rle": costs.frame_rle}[method](frames)
    w.b += p.b
    return _to_bytes(w.b)


def decode(data):
    """bytes -> (method name, frames (T,H,W) uint8, info). Uses nothing but the file."""
    from ca.rule import family_taps, rollout
    b = _bits(data); r = costs.BitR(b)
    m = _NAME[r.gamma()]; W, H, T = r.gamma(), r.gamma(), r.gamma(); info = {}
    if m in _GP:
        var = r.bits(1); off = (r.i + 7) // 8
        x = np.unpackbits(np.frombuffer(_GP[m][1](data[off:]), np.uint8))[:T * H * W].reshape(T, H, W)
        if var:
            x = np.bitwise_xor.accumulate(x, axis=0)
        return m, x.astype(np.uint8), dict(variant="xor" if var else "raw")
    if r.bits(1):
        f0 = costs._runs_r(r, H * W).reshape(H, W)
    else:
        f0 = np.array([r.bits(1) for _ in range(H * W)], np.uint8).reshape(H, W)
    rest = b[r.i:]
    if m in ("ca_circuit", "ca_lut_exc"):
        if m == "ca_circuit":
            d = costs.rule_circuit_decode(rest); luts = costs.eval_circuit(d["gates"], d["outputs"], len(d["taps"]))
        else:
            d = costs.rule_lut_exceptions_decode(rest); luts = d["luts"]
        vis, _, _ = rollout(f0, d["taps"], luts, T, d["S"], d["boundary"], d["hidden_init"])
        info = {k: d[k] for k in ("family", "k", "S", "boundary", "hidden_init")}
        info["updates"] = int((T - 1) * d["S"] * H * W)
        if m == "ca_circuit":
            info["gates"] = len(d["gates"])
        return m, vis.astype(np.uint8), info
    dec = {"changelist": costs.changelist_decode, "xor_rle": costs.xor_rle_decode, "frame_rle": costs.frame_rle_decode}[m]
    return m, dec(rest, f0, T).astype(np.uint8), info


def baselines(frames):
    """Every recording / general-purpose file for a clip, each verified by decoding: {method: bytes}."""
    out = {}
    for m in ("changelist", "xor_rle", "frame_rle", "zlib", "bz2", "lzma"):
        data = encode(m, frames)
        m2, f, _ = decode(data)
        assert m2 == m and (f == frames).all(), m
        out[m] = data
    return out


def decode_rule(data):
    """bytes of a CA container file -> (method, W, H, T, first frame, rule dict from ca.costs decoders)."""
    b = _bits(data); r = costs.BitR(b)
    m = _NAME[r.gamma()]; W, H, T = r.gamma(), r.gamma(), r.gamma()
    assert m in ("ca_circuit", "ca_lut_exc"), m
    f0 = costs._runs_r(r, H * W).reshape(H, W) if r.bits(1) else np.array([r.bits(1) for _ in range(H * W)], np.uint8).reshape(H, W)
    rest = b[r.i:]
    return m, W, H, T, f0, (costs.rule_circuit_decode(rest) if m == "ca_circuit" else costs.rule_lut_exceptions_decode(rest))
