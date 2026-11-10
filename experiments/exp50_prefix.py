"""Exp50 (Campaign 12): longest prefix of an export's FULL autonomous replay (numpy, from the file) that passes the C12
gates (exp46.score on frames 0..L-1, scored against the original), checked at every multiple of STEP frames.
usage: exp50_prefix.py OUT_JSON EXPORT START [STEP]"""
import json, os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca import phase_nca
from exp45_c12 import target
from exp46_eval12 import score
out, path, a0 = sys.argv[1], sys.argv[2], int(sys.argv[3]); step = int(sys.argv[4]) if len(sys.argv) > 4 else 120
hd, E, sched, h0, f0 = phase_nca.load(path); B, V = phase_nca.rollout_np(E, f0, sched, hd["S"], h0); F = target(f"{a0}:{hd['T']}"); res = []
for L in range(step, hd["T"] + 1, step):
    sch = [p for p in hd["schedule"] if p["start"] < L]; r = score(B[:L], V[:L], F[:L], [dict(p, end=min(p["end"], L)) for p in sch])
    res.append(dict(frames=L, err=r["all"]["err"], worst16=r["worst16"], gates=r["gates"], thin_fg=r["thin"]["thin_fg_recall"], thin_bg=r["thin"]["thin_bg_recall"]))
    print(L, round(r["all"]["err"], 4), round(r["worst16"], 4), r["gates"]["success"], flush=True)
ok = [x["frames"] for x in res if x["gates"]["success"]]; longest = max(ok) if ok else 0   # longest passing prefix (short prefixes can fail the motion gate on a static opening)
json.dump(dict(export=path, step=step, prefixes=res, longest_passing_prefix=longest), open(out, "w"), indent=1, default=float); print("longest passing prefix", longest)
