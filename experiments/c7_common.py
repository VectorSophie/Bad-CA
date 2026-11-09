"""Phase 07 shared helpers: targets, export of a rule as a Campaign-05 container file, and independent
verification. The file must replay to EXACTLY the method's own Python rollout P (byte-identical, all
frames), checked by decoding the file in Python and in the Node player (decode_file.mjs). Whether P
matches the target is a separate (exact/approximate) evaluation."""
import json, os, subprocess, sys, time
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca import container, costs
from ca.approx import evaluate, triptych_gif, contact_sheet
from c5_targets import target as t5

C7 = os.path.join(ROOT, "runs", "c7")


def lifelike_lut(rule_id):
    """Memoryless 512-entry LUT (family_taps(0) bit order, centre = bit 4) of a B/S rule id (bits B0..8, S0..8)."""
    idx = np.arange(512); cnt = sum(((idx >> k) & 1) for k in range(9) if k != 4); alive = (idx >> 4) & 1
    return ((rule_id >> (alive * 9 + cnt)) & 1).astype(np.uint8)[None]


def target(name):
    """Phase 07 adds positive-control clips: 'life_b3s23' = Conway from a random 24x24 start (12 frames)."""
    if name == "life_b3s23":
        from ca.rule import family_taps, rollout
        f0 = (np.random.default_rng(7).random((24, 24)) < 0.35).astype(np.uint8)
        v, _, _ = rollout(f0, family_taps(0), lifelike_lut((1 << 3) | (1 << 11) | (1 << 12)), 12, 1)
        return v.astype(np.uint8)
    if name == "so_ctrl":  # known depth-3 second-order tree (exp22 encoding: taps 0-8 now, 9-17 prev; 3 = nowW, 12 = prevW, 4 = nowC, 13 = prevC)
        from exp22_secondorder import rollout as so_rollout
        f0 = (np.random.default_rng(11).random((16, 16)) < 0.3).astype(np.uint8)
        tree = (3, (4, 0, (13, 1, 0)), (12, 1, (4, 0, 1)))
        return so_rollout(tree, f0[None], 10)[0].astype(np.uint8)
    if "@" in name:  # Campaign 09 window: 'clip@a-b' = frames a..b-1 of clip (first frame = true frame a)
        base, ab = name.split("@"); a, b = map(int, ab.split("-")); return target(base)[a:b]
    return t5(name).astype(np.uint8)


def export_verify(P, F, rule, method, path):
    """rule: container rule dict (ca_circuit or ca_lut_exc). Returns file stats + verification flags."""
    data = container.encode(method, P, rule)  # P[0] == F[0]: the file stores the shared first frame
    os.makedirs(os.path.dirname(path), exist_ok=True); open(path, "wb").write(data)
    t0 = time.perf_counter(); _, D, info = container.decode(open(path, "rb").read()); py_s = time.perf_counter() - t0
    nd = json.loads(subprocess.run(["node", os.path.join(ROOT, "player", "decode_file.mjs"), path], capture_output=True, text=True, check=True).stdout)
    return dict(file=os.path.relpath(path, ROOT), file_bytes=len(data), python_replay_equals_P=bool((D == P).all()),
                node_replay_equals_P=nd["sha256"] == container.sha(P), replay_equals_target=bool((P == F).all()),
                python_decode_s=round(py_s, 4), node_decode_ms=nd["decode_ms"], updates=info["updates"])


def media(P, F, stem):
    triptych_gif(P, F, stem + ".gif"); contact_sheet(P, F, stem + ".png")
    return dict(gif=os.path.relpath(stem + ".gif", ROOT), sheet=os.path.relpath(stem + ".png", ROOT))


def baselines_bytes(F):
    b = {m: len(d) for m, d in container.baselines(F).items()}
    return dict(b, best=min(b.values()), best_method=min(b, key=b.get))
