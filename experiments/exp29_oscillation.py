"""Exp29 (Campaign 08, POST-HOC diagnostic, added after viewing the videos; does not change any criterion):
how much of each replay's change activity is local oscillation? For changes at frame t (t-1 -> t), the revert
rate = fraction of changed pixels that change back at t+1. Computed for the replay (decoded from its export)
and for the original clip, per segment. A replay whose late changes mostly revert (while the original's do
not) is flickering in place rather than moving. Writes runs/c8/oscillation.json."""
import json, os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca import container
from ca.motion import segments
from c7_common import target


def revert(X, sl):
    X = X.astype(bool); ch = X[1:] != X[:-1]; num = den = 0
    for t in sl:                                   # change into frame t, reverted by frame t+1
        if t + 1 >= len(X): continue
        c = ch[t - 1]; num += int((c & ch[t]).sum()); den += int(c.sum())
    return num / den if den else None


rows = json.load(open(os.path.join(ROOT, "runs", "c8", "results.json"))); out = []
clips = {}
for r in rows:
    if r["phase"] not in ("main", "second", "curriculum"): continue
    F = clips.setdefault(r["target"], target(r["target"])); P = container.decode(open(os.path.join(ROOT, r["export"]), "rb").read())[1]
    segs = segments(len(F))
    out.append(dict(result=r["result"], phase=r["phase"], target=r["target"], arm=r["arm"], seed=r["seed"], level=r["level"],
                    late_f1_tol=r["late_f1_tol"], revert_replay={n: revert(P, s) for n, s in zip(("early", "mid", "late"), segs)},
                    revert_original={n: revert(F, s) for n, s in zip(("early", "mid", "late"), segs)}))
json.dump(out, open(os.path.join(ROOT, "runs", "c8", "oscillation.json"), "w"), indent=1)
for o in sorted(out, key=lambda o: (o["phase"], o["target"], o["arm"], o["seed"])):
    f = lambda d: "/".join("-" if d[k] is None else f"{d[k]:.2f}" for k in ("early", "mid", "late"))
    print(o["phase"], o["target"], o["arm"], o["seed"], o["level"], f"lateF1 {o['late_f1_tol']:.2f}", "revert replay e/m/l", f(o["revert_replay"]), "| original", f(o["revert_original"]))
