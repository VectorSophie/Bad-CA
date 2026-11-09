"""Exp30 (Campaign 09): motion-task candidates. Every 16-frame window (stride 4) of ba48_4500_60 and ba48_1800_60
is scored BEFORE any model is trained: uniform frames, max per-frame change (scene cut if > 0.25), mean change
fraction, fraction of changes that revert next frame, and 4-frame net change (motion that accumulates). Also a
threshold check (96/128/160) of whether the window's motion survives binarisation. Writes runs/c9/task_candidates.json
and per-candidate contact sheets runs/c9/media/cand_<clip>_<t0>.png (every frame of the window)."""
import json, os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PIL import Image
from ca.video import binarize, gray_frames

OUT = os.path.join(ROOT, "runs", "c9"); os.makedirs(os.path.join(OUT, "media"), exist_ok=True)
G = gray_frames(os.path.join(ROOT, "refs", "Bad-Apple-Starter", "BadApple!!.mp4"), 48)[0]
rows = []
for clip, s0 in (("ba48_4500_60", 4500), ("ba48_1800_60", 1800)):
    for thr in (96, 128, 160):
        B = binarize(G, thr)[s0:s0 + 60]
        for t0 in range(0, 60 - 16 + 1, 4):
            W = B[t0:t0 + 16].astype(bool); ch = (W[1:] != W[:-1]).mean((1, 2))
            rev = [((W[t] != W[t - 1]) & (W[t + 1] != W[t])).sum() / max((W[t] != W[t - 1]).sum(), 1) for t in range(1, 15)]
            net4 = (W[4:] != W[:-4]).mean()
            rows.append(dict(clip=clip, thr=thr, t0=t0, src=(s0 + t0, s0 + t0 + 15), uniform=int(sum(f.min() == f.max() for f in W)),
                             max_change=round(float(ch.max()), 4), mean_change=round(float(ch.mean()), 4), revert=round(float(np.mean(rev)), 3),
                             net4=round(float(net4), 4), white=round(float(W.mean()), 3)))
            if thr == 128:
                img = np.concatenate([np.pad(f.astype(np.uint8) * 255, 1, constant_values=128) for f in W], 1)
                Image.fromarray(img).resize((img.shape[1] * 3, img.shape[0] * 3), Image.NEAREST).save(os.path.join(OUT, "media", f"cand_{clip}_{t0:02d}.png"))
json.dump(rows, open(os.path.join(OUT, "task_candidates.json"), "w"), indent=1)
for r in rows:
    if r["thr"] == 128: print(r)
