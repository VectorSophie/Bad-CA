"""Declarative rule export, format ca-universal/rule-v1 (replayed by player/replay.mjs)."""
import base64, hashlib, json
import numpy as np

from ca.rule import family_taps


def _b64bits(a):
    return base64.b64encode(np.packbits(np.asarray(a, np.uint8).ravel(), bitorder="little").tobytes()).decode()


def export_rule(path, taps, luts, visible0, n_frames, steps_per_frame=1, boundary=0, hidden_init=0,
                meta=None, expected=None):
    """expected: visible frames (T,H,W) the replay must reproduce; stored as SHA-256."""
    H, W = visible0.shape
    C = int(luts.shape[0])
    ex = dict(format="ca-universal/rule-v1", width=W, height=H, channels=C, observed_channel=0,
              taps=[list(map(int, t)) for t in taps], boundary=boundary, hidden_init=hidden_init,
              steps_per_frame=steps_per_frame, n_frames=n_frames,
              luts_b64=[_b64bits(l) for l in luts], seed_visible_b64=_b64bits(visible0),
              sizes=dict(lut_bits=int(luts.size), seed_bits=int(visible0.size), hidden_bits_per_cell=C - 1,
                         hidden_init_bits=0),
              meta=meta or {})
    if expected is not None:
        ex["expected_sha256"] = hashlib.sha256(np.ascontiguousarray(expected, np.uint8).tobytes()).hexdigest()
    with open(path, "w") as f:
        json.dump(ex, f, indent=1)
    return ex


def export_lut(path, lut, seed, n_frames, r=1, boundary=0, steps_per_frame=1, meta=None, expected=None):
    assert r == 1
    return export_rule(path, family_taps(0), np.asarray(lut)[None], seed, n_frames, steps_per_frame, boundary,
                       meta=meta, expected=expected)
