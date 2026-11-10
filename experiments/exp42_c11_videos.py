"""Exp42 (Campaign 11): comparison video from EXPORT replays (first frame + hidden 0, nothing injected); the original is
display/diff only. Full interval always (no excerpt). Writes runs/c11/media/<name>_{1x,0.25x}.{mp4,gif} + decode-back check.
usage: exp42_c11_videos.py NAME TARGET "Title=export" [...] [--diff K,K] [--zoom y0,y1,x0,x1] [--foot "text"]"""
import json, os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca.pipeline import replay
from ca.video_out import diff_rgb, make_video
from exp40_c11 import TARGETS, target

if __name__ == "__main__":
    name, tname = sys.argv[1], sys.argv[2]; args = sys.argv[3:]; opt = lambda k: args[args.index(k) + 1] if k in args else None
    specs = [a for i, a in enumerate(args) if "=" in a and (i == 0 or args[i - 1] not in ("--diff", "--zoom", "--foot"))]
    F = target(tname); s0 = TARGETS[tname][0]; panels = [("original", F)]; errs = []
    for sp in specs:
        title, path = sp.split("=", 1); B, _, _ = replay(os.path.join(ROOT, path), node=False); assert len(B) == len(F)
        panels.append((title, B)); errs.append(f"{title}: err {(B[1:] != F[1:]).mean():.4f}")
    for k in ([int(x) for x in opt("--diff").split(",")] if opt("--diff") else []):
        panels.append((f"diff {panels[k][0].split(' (')[0]}", diff_rgb(panels[k][1], F)))
    scale = 6
    if opt("--zoom"):
        y0, y1, x0, x1 = map(int, opt("--zoom").split(",")); panels = [(t, X[:, y0:y1, x0:x1]) for t, X in panels]; scale = 12
    foot = [f"FULL autonomous replay, src {s0}-{s0 + len(F) - 1} ({len(F)} frames): each CA starts from the export's first frame, hidden 0, no reset or injection",
            "diff: red = FN (original on, CA off), blue = FP (original off, CA on)" + (f"; zoom rows {opt('--zoom')} (y0,y1,x0,x1), x12 nearest" if opt("--zoom") else "")]
    foot += [" | ".join(errs[i:i + 3]) for i in range(0, len(errs), 3)] + ([opt("--foot")] if opt("--foot") else [])
    os.makedirs(os.path.join(ROOT, "runs", "c11", "media"), exist_ok=True)
    ch = make_video(panels, foot, lambda t: f"src {s0 + t}  t={t / 30:5.2f}s", os.path.join(ROOT, "runs", "c11", "media", name), scale=scale)
    print(name, [c["ok"] for c in ch], ch[0]["decoded_frames"], ch[0]["fps"])
