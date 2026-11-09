"""Exp16 (Campaign 05): prefix generalization diagnostic. Every circuit found by direct synthesis on a
prefix 'X:pL' is replayed autonomously on the full clip X (and X_long for the known-circuit clips).
This is a test of whether the dynamics' structure was captured, NOT compression evidence.
Writes runs/c5/prefix.json."""
import glob, json, os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca import costs
from ca.rule import family_taps, rollout
from c5_targets import KNOWN, target

out = []
for p in sorted(glob.glob(os.path.join(ROOT, "runs", "c5", "main", "*.result.json"))):
    r = json.load(open(p))
    if r["status"] != "SAT" or ":p" not in r["target"]: continue
    base = r["target"].split(":p")[0]; taps = family_taps(r["k"], "moore" if r["family"] == "F_shared" else "self")
    L = costs.eval_circuit(r["gates"], r["outputs"], len(taps))
    for full in [base] + ([base + "_long"] if base in KNOWN else []):
        F = target(full); vis, _, _ = rollout(F[0], taps, L, len(F), r["S"])
        wrong = (vis != F).reshape(len(F), -1).sum(1)
        out.append(dict(prefix_target=r["target"], G=r["G"], struct=r["struct"], test_clip=full, frames=len(F),
                        exact=bool(wrong.sum() == 0), first_wrong_frame=int(np.argmax(wrong > 0)) if wrong.sum() else None,
                        wrong_pixels_per_frame=wrong.tolist(), pixel_err=float((vis != F).mean())))
        print(json.dumps(out[-1]))
json.dump(out, open(os.path.join(ROOT, "runs", "c5", "prefix.json"), "w"), indent=1)
