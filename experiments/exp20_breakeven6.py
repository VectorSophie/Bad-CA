"""Exp20 (Campaign 06): break-even re-check with REAL files for ba32_3000_5 (F_priv k1, S2).
Random valid G-gate circuits (G = 10..14) are written as container files and parsed back by Python and by
the Node player (they do not reproduce the clip; only their size and decodability matter). The baseline
files from Campaign 05 are decoded again from their bytes. Writes runs/c6/breakeven.json."""
import glob, json, os, subprocess, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca import container
from c5_targets import target

F = target("ba32_3000_5"); rng = np.random.default_rng(0); out = dict(target_sha256=container.sha(F), circuit_files={}, baselines={})
os.makedirs(os.path.join(ROOT, "runs", "c6", "breakeven"), exist_ok=True)
for G in range(10, 15):
    gates = [(int(rng.integers(16)), int(a), int(rng.integers(a + 1, 10 + j))) for j in range(G) for a in [rng.integers(10 + j - 1)]]
    data = container.encode("ca_circuit", F, dict(gates=gates, outputs=[10 + G - 1, 9], n_in=10, family="F_priv", k=1, S=2))
    p = os.path.join(ROOT, "runs", "c6", "breakeven", f"random_G{G}.cab"); open(p, "wb").write(data)
    m, _, info = container.decode(open(p, "rb").read())
    nd = json.loads(subprocess.run(["node", os.path.join(ROOT, "player", "decode_file.mjs"), p], capture_output=True, text=True, check=True).stdout)
    out["circuit_files"][G] = dict(bytes=len(data), python_parsed_gates=info["gates"], node_parsed_gates=nd["gates"], node_file_bytes=nd["file_bytes"])
for p in sorted(glob.glob(os.path.join(ROOT, "runs", "c5", "files", "ba32_3000_5", "*.bin"))):
    m, Fd, _ = container.decode(open(p, "rb").read())
    out["baselines"][m] = dict(bytes=os.path.getsize(p), decodes_exact=bool((Fd == F).all()))
best = min(v["bytes"] for v in out["baselines"].values())
out["best_baseline_bytes"] = best; out["max_G_below_best"] = max(G for G, v in out["circuit_files"].items() if v["bytes"] < best)
json.dump(out, open(os.path.join(ROOT, "runs", "c6", "breakeven.json"), "w"), indent=1); print(json.dumps(out))
