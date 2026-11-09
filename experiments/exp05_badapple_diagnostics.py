"""Exp05: necessary-condition diagnostics on Bad Apple clips (skips the black opening: every
clip starts at an explicitly chosen NON-uniform frame; full-song start is a separate decision).
For each clip: history conflicts (F_priv, S=1, any k), light-cone conflicts (any hidden family,
constant hidden init, S=1..6), state repeats (memoryless, any S)."""
import json, os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from ca.sat import history_conflicts, light_cone_conflicts, state_repeat_conflict
from ca.video import binarize, gray_frames
mp4 = os.path.join(ROOT, "refs", "Bad-Apple-Starter", "BadApple!!.mp4")
STARTS = [300, 900, 1860, 3000, 4500]   # frame indices (t = idx/30 s); all verified non-uniform below
R = {}
for W in (32, 64):
    B = binarize(gray_frames(mp4, W)[0])
    for s in STARTS:
        assert B[s].min() != B[s].max()
        for L in (3, 5, 8):
            clip = B[s:s + L]
            key = f"ba{W}_{s}_{L}"
            lc = {S: light_cone_conflicts(clip, S) for S in range(1, 7)}
            R[key] = dict(start_frame=s, t_start_s=s / 30, frames=L, changes=int((clip[1:] != clip[:-1]).sum()),
                          history_conflicts_any_k=history_conflicts(clip), state_repeat=state_repeat_conflict(clip),
                          light_cone={S: sum(b for _, b in v) for S, v in lc.items()},
                          min_S_light_cone_ok=next((S for S in range(1, 7) if not lc[S]), ">6"))
            r = R[key]
            print(f"{key:16s} changes={r['changes']:5d} hist_conflicts={r['history_conflicts_any_k'][0]:4d} "
                  f"lightcone(S=1..6)={list(r['light_cone'].values())} minS={r['min_S_light_cone_ok']}")
os.makedirs(os.path.join(ROOT, "runs", "exp05"), exist_ok=True)
json.dump(R, open(os.path.join(ROOT, "runs", "exp05", "diagnostics.json"), "w"), indent=1)
