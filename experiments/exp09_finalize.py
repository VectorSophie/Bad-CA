"""Exp09 finalize (analysis, no search): export every saved exact rule from runs/<DIR>/*.L*.luts.npy and
replay it in the independent Node player; the SHA-256 must equal the target clip's.
usage: exp09_finalize.py DIR"""
import glob, json, os, re, subprocess, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca.rule import family_taps
from ca.export import export_rule
from exp03_sat_ablation import CONDS, target

out = {}
for p in sorted(glob.glob(os.path.join(ROOT, "runs", sys.argv[1], "*.L*.luts.npy"))):
    m = re.search(r"ba32_(\d+)_(\w+?)_(cold|incr)_.*\.L(\d+)\.luts\.npy$", p)
    start, cond, L = m.group(1), m.group(2), int(m.group(4))
    k, S, seen_by = CONDS[cond]; taps = family_taps(k, seen_by); tname = f"ba32_{start}_{L}"
    F = target(tname); luts = np.load(p)
    path = p.replace(".luts.npy", ".rule.json")
    ex = export_rule(path, taps, luts, F[0], L, S, meta=dict(source=tname, method=os.path.basename(p), cond=cond), expected=F)
    sha = json.loads(subprocess.run(["node", os.path.join(ROOT, "player", "replay.mjs"), path], capture_output=True, text=True, check=True).stdout)["sha256"]
    out[os.path.basename(path)] = sha == ex["expected_sha256"]
json.dump(out, open(os.path.join(ROOT, "runs", sys.argv[1], "node_verify.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
