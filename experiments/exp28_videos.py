"""Exp28 (Campaign 08): final comparison videos. Every CA panel is DECODED FROM ITS EXPORTED FILE
(ca.container.decode); original frames are used only for display and errors. No interpolation, editing or
re-timing. Selection per docs/CAMPAIGN_08.md: arm with most success+partial seeds, then highest median late
F1tol; representative = median-late-F1 seed of that arm, plus its best and worst seed.
Writes runs/c8/media/*.{mp4,gif} + video_index.json."""
import json, os, statistics as stt, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca import container
from ca.video_out import diff_rgb, make_video
from c7_common import target
from c8_targets import control

OUT = os.path.join(ROOT, "runs", "c8", "media"); os.makedirs(OUT, exist_ok=True)
rows = json.load(open(os.path.join(ROOT, "runs", "c8", "results.json")))
C7 = {"ba48_4500_60": "runs/c7/c4_cgp/ba48_4500_60__k1_S2__G32__s3.best.cab", "ba48_1800_60": "runs/c7/c4_cgp/ba48_1800_60__k1_S2__G16__s3.best.cab"}
START = {"ba48_4500_60": 4500, "ba48_1800_60": 1800}
index = []


def dec(path):
    return container.decode(open(os.path.join(ROOT, path), "rb").read())


def tag(r):
    return f"arm {r['arm']} seed {r['seed']}: {r['active_gates']} gates k={r['k']} S={r['S']}"


def seg_line(r):
    return ("F1tol early/mid/late " + "/".join(f"{r[s + '_f1_tol']:.2f}" for s in ("early", "mid", "late")) +
            "   err/hold " + "/".join(f"{r[s + '_err'] / max(r[s + '_hold'], 1e-9):.2f}" for s in ("early", "mid", "late")) + f"   level: {r['level']}")


def video(name, panels, footer, start, fps=30):
    lab = (lambda t: f"src frame {start + t}  t={t / fps:5.2f}s  [{t + 1}/{len(panels[0][1])}]") if start is not None else (lambda t: f"frame {t}  t={t / fps:5.2f}s")
    checks = make_video(panels, footer, lab, os.path.join(OUT, name)); index.append(dict(name=name, checks=checks, footer=footer))


def pick(phase, clip):
    rs = [r for r in rows if r["phase"] == phase and r["target"] == clip]
    arms = {}
    for r in rs: arms.setdefault(r["arm"], []).append(r)
    key = lambda a: (sum(r["level"] in ("success", "partial") for r in arms[a]), stt.median(r["late_f1_tol"] for r in arms[a]))
    best_arm = max(arms, key=key); rs = sorted(arms[best_arm], key=lambda r: r["late_f1_tol"])
    return best_arm, rs[(len(rs) - 1) // 2], rs[-1], rs[0], arms


for phase, clip in [("main", "ba48_4500_60"), ("second", "ba48_1800_60")]:
    if not any(r["phase"] == phase for r in rows): continue
    F = target(clip); arm, rep, best, worst, arms = pick(phase, clip); _, P7, i7 = dec(C7[clip])
    for role, r in [("representative", rep), ("best", best), ("worst", worst)]:
        _, P, info = dec(r["export"]); assert (P[0] == F[0]).all()
        foot = [f"clip {clip} 48x36 thr128 boundary 0 hidden init 0 | {r['objective']} objective (lam {r['lam']:g}) | {role} seed of arm {arm}",
                seg_line(r), "replays decoded from exported files; red = missed white (FN), blue = extra white (FP)"]
        if role == "representative":
            video(f"{phase}_{clip}_1_orig_c7_c8", [("original (processed)", F), (f"C7 best CGP: {i7['gates']} gates k=1 S=2", P7), ("C8 " + tag(r), P)], foot, START[clip])
        video(f"{phase}_{clip}_2_{role}_orig_c8_diff", [("original (processed)", F), ("C8 " + tag(r), P), ("diff vs original", diff_rgb(P, F))], foot, START[clip])
    index.append(dict(name=f"{phase}_{clip}_selection", arm=arm, representative=rep["result"], best=best["result"], worst=worst["result"],
                      arm_seed_levels={a: [x["level"] for x in v] for a, v in arms.items()}))

for clip in ("pcmove_k1_S2", "pcmove_k1s_S4", "pccycle_k1_S2", "pccycle_k1s_S4"):
    rs = [r for r in rows if r["phase"] == "controls" and r["target"] == clip and r["objective"] == "mot"]
    if not rs: continue
    r = min(rs, key=lambda r: (not r["exact"], r["J1"], r["seed"])); F = control(clip)[0]; _, P, _ = dec(r["export"])
    video(f"control_{clip}_3_orig_recovered", [("original (known circuit)", F), ("recovered: " + tag(r), P), ("diff", diff_rgb(P, F))],
          [f"positive control {clip}: CGP saw only these frames | exact: {r['exact']} | {sum(x['exact'] for x in rs)}/{len(rs)} mot seeds exact", seg_line(r)], None)
json.dump(index, open(os.path.join(OUT, "video_index.json"), "w"), indent=1, default=float)
print(json.dumps([dict(name=x["name"], ok=all(c["ok"] for c in x.get("checks", [])) if "checks" in x else None) for x in index], indent=1))
