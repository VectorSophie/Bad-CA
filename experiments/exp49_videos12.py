"""Exp49 (Campaign 12): comparison videos re-read from EXPORT files (ncab v1/v2 NCA, dlrule-v1 rules); the original is
display/diff only. Each frame label shows the source frame, time and the phase (rule/program id) every phased panel uses.
usage: exp49_videos12.py NAME START:LEN "Title=export" [...] [--diff K,K] [--zoom y0,y1,x0,x1] [--range a:b] [--foot TEXT]
       [--no-gif] [--slow-only-range]   (--range cuts frames a..b-1 out of the FULL autonomous rollout, never restarts)"""
import os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca import nca, ncab, phase_nca, rules
from ca.video_out import diff_rgb, make_video
from exp45_c12 import target


def replay(path, T):
    """-> binary frames (T,H,W), per-frame rule id array or None."""
    if path.endswith(".rule.json"):
        m = rules.load(path); B, _ = rules.run(m, T); s = np.zeros(T, int)
        for p in m["schedule"]: s[p["start"]:p["end"]] = p["program"]
        return B, s if len(m["programs"]) > 1 else None
    if open(path, "rb").read(5) == b"NCAB\x02":
        hd, E, sched, h0, f0 = phase_nca.load(path); sched = sched[:T]   # first T frames of a longer export = cut from its full rollout (causal)
        B, _ = phase_nca.rollout_np(E, f0, sched, hd["S"], h0); return B, sched if hd["K"] > 1 else None
    hd, P, f0 = ncab.read(path) if path.endswith(".ncab") else (lambda e: ({"S": e[0]["S"]}, e[1], e[2]))(nca.load(path))
    B, _ = nca.rollout_np(P, f0[None], T, hd["S"]); return B, None


if __name__ == "__main__":
    name, spec = sys.argv[1], sys.argv[2]; args = sys.argv[3:]; opt = lambda k: args[args.index(k) + 1] if k in args else None
    skip = {i + 1 for i, a in enumerate(args) if a in ("--diff", "--zoom", "--range", "--foot")}
    specs = [a for i, a in enumerate(args) if "=" in a and i not in skip]
    F = target(spec); s0 = int(spec.split(":")[0]); T = len(F); panels = [("original (binary target)", F)]; scheds = []; errs = []
    for sp in specs:
        title, path = sp.split("=", 1); B, sc = replay(os.path.join(ROOT, path), T); assert (B[0] == F[0]).all()
        panels.append((title, B)); scheds.append((title.split(" ")[0], sc)); errs.append(f"{title}: err {(B[1:] != F[1:]).mean():.4f}")
    for k in ([int(x) for x in opt("--diff").split(",")] if opt("--diff") else []): panels.append((f"diff {panels[k][0].split(' (')[0]}", diff_rgb(panels[k][1], F)))
    a, b = (0, T) if not opt("--range") else map(int, opt("--range").split(":")); panels = [(t, X[a:b]) for t, X in panels]
    scale = 6
    if opt("--zoom"): y0, y1, x0, x1 = map(int, opt("--zoom").split(",")); panels = [(t, X[:, y0:y1, x0:x1]) for t, X in panels]; scale = 12
    if len(panels) >= 6: scale = min(scale, 5)
    lab = lambda t: f"src {s0 + a + t}  t={(s0 + a + t) / 30:6.2f}s" + "".join(f" | {n} phase {int(sc[a + t])}" for n, sc in scheds if sc is not None)
    foot = [f"AUTONOMOUS replay from each export: first frame (+ stored initial hidden) only, no reset or injection; src {s0}-{s0 + T - 1}" + (f"; shown {s0 + a}-{s0 + b - 1}, CUT from the full rollout" if opt("--range") else ""),
            "phased panels: external schedule, state carried across boundaries; diff: red = FN (original on, CA off), blue = FP" + (f"; zoom {opt('--zoom')} x12 nearest" if opt("--zoom") else "")]
    foot += [" | ".join(errs[i:i + 3]) + (" (error over the whole interval)" if i == 0 else "") for i in range(0, len(errs), 3)] + ([opt("--foot")] if opt("--foot") else [])
    os.makedirs(os.path.join(ROOT, "runs", "c12", "media"), exist_ok=True)
    ch = make_video(panels, foot, lab, os.path.join(ROOT, "runs", "c12", "media", name), scale=scale, gif="--no-gif" not in args)
    print(name, [c["ok"] for c in ch], ch[0]["decoded_frames"], ch[0]["fps"])
