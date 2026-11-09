"""Exp25 (Campaign 08, step 1): comparison video of the Campaign 07 best CGP result on ba48_4500_60.
Panels: original processed frames | C7 best CGP (frames decoded from its EXPORT FILE, not from memory) |
hold baseline. Original frames are used only for display and error computation. Writes
runs/c8/media/c7_best_vs_hold_{1x,0.25x}.{mp4,gif} + .video_check.json."""
import json, os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca import container
from ca.video_out import make_video
from c7_common import target

START, FPS = 4500, 30
F = target("ba48_4500_60"); T = len(F)
res = json.load(open(os.path.join(ROOT, "runs", "c7", "c4_cgp", "ba48_4500_60__k1_S2__G32__s3.result.json")))
_, P, info = container.decode(open(os.path.join(ROOT, res["export"]["file"]), "rb").read())
assert P.shape == F.shape and (P[0] == F[0]).all()
H = np.repeat(F[:1], T, 0)
err = lambda X: [(X[t] != F[t]).mean() for t in range(T)]
eP, eH = err(P), err(H)
OUT = os.path.join(ROOT, "runs", "c8", "media"); os.makedirs(OUT, exist_ok=True)
panels = [("original (processed)", F), (f"C7 CGP s3: {info['gates']} gates k=1 S=2", P), ("hold (frame 0 frozen)", H)]
lab = lambda t: f"src frame {START + t}  t={t / FPS:5.2f}s  [{t + 1}/{T}]"
footer = ["clip ba48_4500_60 48x36 thr128, boundary 0, hidden init 0",
          "C7 panel = replay decoded from export " + os.path.basename(res["export"]["file"]),
          f"mean pixel err: C7 {np.mean(eP[1:]):.4f}  hold {np.mean(eH[1:]):.4f}  (frames 1..59)"]
checks = make_video(panels, footer, lab, os.path.join(OUT, "c7_best_vs_hold"))
print(json.dumps(checks, indent=1))
