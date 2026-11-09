"""Exp17 (Campaign 05): break-even gate count. For each clip x structure, the exact file size of a G-gate
circuit file (container header + first frame + rule header + gates with their real reference widths
+ outputs; the gate contents do not change the size) for G = 0..200, and the largest G whose file is
still smaller than the clip's best baseline file. Compared with the proved lower bounds (search.json)."""
import json, os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca import container
from ca.rule import family_taps
from c5_targets import STRUCTS, target

V = json.load(open(os.path.join(ROOT, "runs", "c5", "verdicts.json")))
B = {(b["clip"], b["struct"]): b for b in json.load(open(os.path.join(ROOT, "runs", "c5", "search.json")))["bounds"]}
out = []
for (clip, st), b in sorted(B.items()):
    if clip not in V: continue
    k, seen, S = STRUCTS[st]; n = len(family_taps(k, seen)); F = target(clip)
    fam = "memoryless" if k == 0 else ("F_priv" if seen == "self" else "F_shared")
    size = lambda G: len(container.encode("ca_circuit", F, dict(gates=[(0, 0, 1)] * G, outputs=[0] * (k + 1), n_in=n, family=fam, k=k, S=S)))
    best = V[clip]["best_baseline_bytes"]
    be = max((G for G in range(0, 201) if size(G) < best), default=None)
    rec = dict(clip=clip, struct=st, n_inputs=n, best_baseline=V[clip]["best_baseline"], best_baseline_bytes=best,
               file_bytes_G0=size(0), file_bytes_G12=size(12), break_even_max_G=be,
               proved_no_circuit_up_to_G=b["no_circuit_up_to_G"],
               compression_ruled_out=b["no_circuit_up_to_G"] is not None and be is not None and b["no_circuit_up_to_G"] >= be,
               open_window=None if be is None else [None if b["no_circuit_up_to_G"] is None else b["no_circuit_up_to_G"] + 1, be])
    out.append(rec); print(json.dumps(rec))
json.dump(out, open(os.path.join(ROOT, "runs", "c5", "breakeven.json"), "w"), indent=1)
