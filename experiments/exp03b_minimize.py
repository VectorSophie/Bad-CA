"""Minimise every SAT-found rule in runs/exp03 (visited entries = care set), re-roll, re-export, Node-verify."""
import glob, json, os, subprocess, sys, time
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca.minimize import sop
from ca.rule import family_taps, rollout
from ca.export import export_rule
from exp03_sat_ablation import CONDS, target

for rf in sorted(glob.glob(os.path.join(ROOT, "runs", "exp03", "*.result.json"))):
    r = json.load(open(rf))
    if r.get("status") != "SAT" or "minimized" in r:
        continue
    k, S, seen_by = CONDS[r["cond"]]; taps = family_taps(k, seen_by); n = len(taps)
    if n > 12:
        r["minimized"] = f"skipped: {n} inputs exceeds sympy QM shortcut"; json.dump(r, open(rf, "w"), indent=1); continue
    stem = rf.replace(".result.json", "")
    luts = np.load(stem + ".luts.npy"); care = json.load(open(stem + ".visited.json"))
    F = target(r["target"])
    t0 = time.perf_counter(); out, new = [], []
    for c in range(k + 1):
        d, l = sop(luts[c], care, n); out.append(d); new.append(l)
    new = np.stack(new)
    vis, _, _ = rollout(F[0], taps, new, len(F), S)
    path = stem + ".min.rule.json"
    ex = export_rule(path, taps, new, F[0], len(F), S, meta=dict(source=r["target"], method="SAT+SOP-dontcare", cond=r["cond"],
                                                                  sop=[d["expr"] for d in out]), expected=F)
    node = json.loads(subprocess.run(["node", os.path.join(ROOT, "player", "replay.mjs"), path], capture_output=True, text=True, check=True).stdout)
    r["minimized"] = dict(seconds=round(time.perf_counter() - t0, 1), per_channel=[{kk: v for kk, v in d.items() if kk != "expr"} for d in out],
                          total_terms=sum(d["terms"] for d in out), total_literals=sum(d["literals"] for d in out),
                          rollout_exact=bool((vis == F).all()), node_hash_matches_target=node["sha256"] == ex["expected_sha256"])
    json.dump(r, open(rf, "w"), indent=1)
    print(r["target"], r["cond"], json.dumps(r["minimized"]))
