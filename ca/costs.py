"""Reproducible description-length encoders, each with a decoder (round-trip checked).

Shared information excluded from BOTH sides (reported separately): W, H, T, fps, threshold,
first frame. Everything else a method needs to regenerate frames 1..T-1 is inside its bitstream.
All integers use Elias-gamma (gamma(x), x >= 1); `+1` shifts allow zero.
"""
import math
import numpy as np


class BitW:
    def __init__(self): self.b = []
    def bits(self, v, n): self.b += [(v >> i) & 1 for i in range(n - 1, -1, -1)]
    def gamma(self, x):
        assert x >= 1; n = x.bit_length(); self.b += [0] * (n - 1); self.bits(x, n)
    def __len__(self): return len(self.b)


class BitR:
    def __init__(self, b): self.b, self.i = b, 0
    def bits(self, n):
        v = 0
        for _ in range(n): v = (v << 1) | self.b[self.i]; self.i += 1
        return v
    def gamma(self):
        z = 0
        while self.b[self.i] == 0: z += 1; self.i += 1
        return self.bits(z + 1)


# ---------------- recordings of frames 1..T-1 given frame 0 ----------------

def _gaps_w(w, idx):
    w.gamma(len(idx) + 1); prev = -1
    for i in idx: w.gamma(int(i) - prev); prev = int(i)

def _gaps_r(r):
    m = r.gamma() - 1; out = []; prev = -1
    for _ in range(m): prev += r.gamma(); out.append(prev)
    return out


def changelist(frames):
    """Per frame: count, then gap-coded flat positions of pixels that differ from the previous frame."""
    w = BitW()
    for t in range(1, len(frames)):
        _gaps_w(w, np.flatnonzero(frames[t] != frames[t - 1]))
    return w

def changelist_decode(bits, f0, T):
    r = BitR(bits); out = [f0.copy()]
    for _ in range(1, T):
        f = out[-1].copy().ravel(); f[_gaps_r(r)] ^= 1; out.append(f.reshape(f0.shape))
    return np.stack(out)


def _runs_w(w, flat):
    """Alternating runs starting with a run of 0s (may be 0 long); decoder knows the length."""
    v, n = 0, 0
    for x in flat:
        if x == v: n += 1
        else: w.gamma(n + 1); v, n = x, 1
    w.gamma(n + 1)

def _runs_r(r, N):
    out, v = [], 0
    while len(out) < N:
        out += [v] * (r.gamma() - 1); v ^= 1
    assert len(out) == N
    return np.array(out, np.uint8)


def xor_rle(frames):
    """Per frame: XOR with the previous frame, row-major, alternating run lengths."""
    w = BitW()
    for t in range(1, len(frames)):
        _runs_w(w, (frames[t] ^ frames[t - 1]).ravel())
    return w

def xor_rle_decode(bits, f0, T):
    r = BitR(bits); out = [f0.copy()]
    for _ in range(1, T):
        out.append(out[-1] ^ _runs_r(r, f0.size).reshape(f0.shape))
    return np.stack(out)


def frame_rle(frames):
    """Weak reference without temporal prediction: each frame's own alternating runs."""
    w = BitW()
    for t in range(1, len(frames)):
        _runs_w(w, frames[t].ravel())
    return w

def frame_rle_decode(bits, f0, T):
    r = BitR(bits)
    return np.stack([f0] + [_runs_r(r, f0.size).reshape(f0.shape) for _ in range(1, T)])


# ---------------- rules ----------------
FAMILY = {"memoryless": 1, "F_priv": 2, "F_shared": 3}
BOUNDARY = {0: 0, 1: 1, "wrap": 2}


def _rule_header(w, family, k, S, boundary, hidden_init):
    w.gamma(FAMILY[family]); w.gamma(k + 1); w.gamma(S); w.bits(BOUNDARY[boundary], 2); w.bits(hidden_init, 1)

def _rule_header_r(r):
    fam = {v: k for k, v in FAMILY.items()}[r.gamma()]; k = r.gamma() - 1; S = r.gamma()
    bd = {v: k for k, v in BOUNDARY.items()}[r.bits(2)]; hi = r.bits(1)
    return fam, k, S, bd, hi


def default_luts(taps, C):
    """'keep' default: each output channel copies the cell's own current value of that channel."""
    n = len(taps); idx = np.arange(2 ** n)
    return np.stack([(idx >> taps.index((c, 0, 0))) & 1 for c in range(C)]).astype(np.uint8)


def rule_lut_exceptions(luts, taps, family, k, S, boundary=0, hidden_init=0):
    """Header + default-behaviour id (1 = keep) + gap-coded sorted global exception indices
    (c * 2^n + e). Taps/wiring are implied by family id and k (see ca.rule.family_taps)."""
    w = BitW(); _rule_header(w, family, k, S, boundary, hidden_init); w.gamma(1)
    d = default_luts(taps, luts.shape[0])
    _gaps_w(w, np.flatnonzero((luts != d).ravel()))
    return w

def rule_lut_exceptions_decode(bits):
    from ca.rule import family_taps
    r = BitR(bits); fam, k, S, bd, hi = _rule_header_r(r); assert r.gamma() == 1
    taps = family_taps(k, "self" if fam != "F_shared" else "moore")
    luts = default_luts(taps, k + 1).ravel()
    luts[_gaps_r(r)] ^= 1
    return dict(family=fam, k=k, S=S, boundary=bd, hidden_init=hi, taps=taps, luts=luts.reshape(k + 1, -1))


def rule_circuit(gates, outputs, n_in, family, k, S, boundary=0, hidden_init=0):
    """Live sub-circuit. gates: list of (op, a, b), where a/b index the pool
    [inputs 0..n_in-1, gate 0, gate 1, ...] (topological order); outputs: pool index per channel.
    Each reference costs ceil(log2(pool size so far)) bits; op costs 4 bits."""
    w = BitW(); _rule_header(w, family, k, S, boundary, hidden_init); w.gamma(len(gates) + 1)
    for j, (op, a, b) in enumerate(gates):
        nb = max(1, math.ceil(math.log2(n_in + j))); w.bits(op, 4); w.bits(a, nb); w.bits(b, nb)
    nb = max(1, math.ceil(math.log2(n_in + len(gates))))
    for o in outputs: w.bits(o, nb)
    return w

def rule_circuit_decode(bits):
    from ca.rule import family_taps
    r = BitR(bits); fam, k, S, bd, hi = _rule_header_r(r)
    taps = family_taps(k, "self" if fam != "F_shared" else "moore"); n_in = len(taps)
    G = r.gamma() - 1; gates = []
    for j in range(G):
        nb = max(1, math.ceil(math.log2(n_in + j))); gates.append((r.bits(4), r.bits(nb), r.bits(nb)))
    nb = max(1, math.ceil(math.log2(n_in + G)))
    outputs = [r.bits(nb) for _ in range(k + 1)]
    return dict(family=fam, k=k, S=S, boundary=bd, hidden_init=hi, taps=taps, gates=gates, outputs=outputs)


def eval_circuit(gates, outputs, n_in):
    """Hard evaluation on all 2^n_in inputs -> LUTs (C, 2^n_in)."""
    idx = np.arange(2 ** n_in)
    pool = [((idx >> i) & 1).astype(np.uint8) for i in range(n_in)]
    for op, a, b in gates:
        x, y = pool[a], pool[b]
        tt = [(op >> q) & 1 for q in range(4)]  # truth table bit for (a,b) = 00,01,10,11 (see OP_TT)
        pool.append(np.choose(x * 2 + y, [np.full_like(x, tt[0]), np.full_like(x, tt[1]), np.full_like(x, tt[2]), np.full_like(x, tt[3])]).astype(np.uint8))
    return np.stack([pool[o] for o in outputs])


# truth-table code (bit q = output for (a,b) = (q>>1, q&1)) of DiffLogic op i, from ca.difflogic.ops
OP_TT = []
for i in range(16):
    tt = 0
    for q in range(4):
        a, b = q >> 1, q & 1
        v = [0, a & b, a & (1 - b), a, (1 - a) & b, b, a ^ b, a | b, 1 - (a | b), 1 - (a ^ b), 1 - b,
             a | (1 - b), 1 - a, (1 - a) | b, 1 - (a & b), 1][i]
        tt |= v << q
    OP_TT.append(tt)


def first_frame_bits(f0):
    return int(f0.size)  # raw; shared by every method and excluded from comparisons


def recording_costs(frames):
    """All recording baselines for frames 1..T-1 given frame 0, each decode-verified."""
    out = {}
    for name, enc, dec in [("changelist", changelist, changelist_decode), ("xor_rle", xor_rle, xor_rle_decode),
                           ("frame_rle", frame_rle, frame_rle_decode)]:
        w = enc(frames)
        assert (dec(w.b, frames[0], len(frames)) == frames).all(), name
        out[name] = len(w)
    return out


def shared_info(frames):
    T, H, W = frames.shape
    return dict(note="excluded from every method's cost: W, H, T, fps, threshold, first frame",
                first_frame_raw_bits=int(H * W), W=int(W), H=int(H), T=int(T))
