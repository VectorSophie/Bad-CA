"""Exp39 (Campaign 10): comparison videos from EXPORT replays (runs/c10/replays/*.npz written by exp38 from the
nca-v1 exports). Original frames are only displayed / diffed. All panels share frame index and time; nearest-neighbour
scaling; no interpolation, re-timing or morphological post-processing.
Candidate per task (docs/CAMPAIGN_10.md): the qualifying arm with the lowest median full-span contour-band error;
if none qualifies, the arm with the lowest median band error is shown and labelled 'NOT an improvement'.
Seeds: representative = median full-span pixel error of the candidate arm, plus best and worst.
usage: exp39_videos10.py  (reads runs/c10/eval/c10_main10_<task>_best.json)"""
import json, os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca.video_out import diff_rgb, make_video
from c7_common import target
from exp36_full10 import CROP, gray

C10 = os.path.join(ROOT, "runs", "c10"); OUT = os.path.join(C10, "media"); index = []
TASKS = {"main": ("ba48_4500_60", 4500, 40), "confirm": ("ba48_1800_60", 1800, 44)}


def vid(name, panels, footer, s0, scale=6):
    ch = make_video(panels, footer, lambda t: f"src {s0 + t}  t={t / 30:5.2f}s", os.path.join(OUT, name), scale=scale)
    index.append(dict(name=name, ok=all(c["ok"] for c in ch), frames=ch[0]["decoded_frames"], fps_1x=ch[0]["fps"], duration_1x=ch[0]["duration_s"], footer=footer))


for task, (clip, s0, t0) in TASKS.items():
    E = json.load(open(os.path.join(C10, "eval", f"c10_main10_{task}_best.json"))); rows, crit, tab = E["rows"], E["criteria"], E["median_table"]
    F = target(clip)[:t0 + 16]; T = len(F); end = s0 + T - 1; y0, y1, x0, x1 = CROP[task]
    arms = [a for a in ("A", "B", "C", "BC") if a in tab]
    q = [a for a in arms if crit[a]["improved"]]; cand = min(q or arms, key=lambda a: tab[a]["band"]); ok = bool(q)
    cs = sorted((r for r in rows.values() if r["arm"] == cand), key=lambda r: r["report"]["all"]["err"]); roles = dict(best=cs[0], rep=cs[1], worst=cs[2])
    rp = lambda arm, s: np.load(os.path.join(ROOT, rows[f"{arm}_s{s}"]["replay"]))
    lab = lambda arm, s: f"{arm} s{s}" + (" (C9 start)" if arm == "C9" else "")
    st = lambda arm, s: (lambda r: f"{arm} s{s}: err {r['all']['err']:.4f} band {r['all']['err_band']:.4f} lag1 {r['lag1']['f1_tol']:.2f} lag4 {r['lag4']['f1_tol']:.2f}")(
        rows[f"C9_s{s}"]["report"] if arm == "C9" else rows[f"{arm}_s{s}"]["report"])
    verdict = f"candidate {cand}: " + ("meets the pre-registered improvement criteria" if ok else "NOT an improvement by the pre-registered criteria")
    head = f"FULL autonomous replay from exports, src {s0}-{end} ({T} frames); 10,320 params each; same seed s{{}} start"
    for role, r in roles.items():
        s = r["seed"]
        # (1) full span: original / C9 / A / B / C (same seed)
        vid(f"c10_{task}_1_FULL_{s0}-{end}_orig_c9_A_B_C_{role}_s{s}", [("original", F)] + [(lab(a, s), rp(a, s)["B"]) for a in ["C9"] + arms],
            [head.format(s), " | ".join(st(a, s) for a in ["C9"] + arms[:2]), " | ".join(st(a, s) for a in arms[2:]) + f" | {role} seed of {cand}"], s0)
        # (2) candidate vs C9 with diffs
        Bo, Bc = rp("C9", s)["B"], rp(cand, s)["B"]
        vid(f"c10_{task}_2_FULL_{s0}-{end}_orig_c9_{cand}_diffs_{role}_s{s}", [("original", F), (lab("C9", s), Bo), (lab(cand, s), Bc), ("C9 diff", diff_rgb(Bo, F)), (f"{cand} diff", diff_rgb(Bc, F))],
            [head.format(s), "diff: red = FN (original on, replay off), blue = FP (original off, replay on)", st("C9", s) + " | " + st(cand, s), verdict + f"; {role} seed"], s0)
    s = roles["rep"]["seed"]; Rc = rp(cand, s); Ro = rp("C9", s)
    # (3) continuous vs binary, candidate and C9
    vid(f"c10_{task}_3_FULL_{s0}-{end}_continuous_vs_binary_rep_s{s}",
        [("original", F), (f"C9 s{s} continuous", gray(Ro["V"])), (f"C9 s{s} binary", Ro["B"]), (f"{cand} s{s} continuous", gray(Rc["V"])), (f"{cand} s{s} binary", Rc["B"])],
        [head.format(s), f"continuous: gray = clip(v,0,1) (v is the raw visible state, not a probability); binary = v > 0.5",
         f"v range C9 [{Ro['V'].min():.2f},{Ro['V'].max():.2f}]  {cand} [{Rc['V'].min():.2f},{Rc['V'].max():.2f}]", verdict], s0)
    # (4) zoom on the action region, all arms, representative seed
    z = lambda X: X[:, y0:y1, x0:x1]
    vid(f"c10_{task}_4_ZOOM_FULL_{s0}-{end}_rows{y0}-{y1 - 1}_cols{x0}-{x1 - 1}_orig_c9_{cand}_diffs_rep_s{s}",
        [("original", z(F)), (f"C9 s{s}", z(Ro["B"])), (f"{cand} s{s}", z(Rc["B"])), ("C9 diff", diff_rgb(z(Ro["B"]), z(F))), (f"{cand} diff", diff_rgb(z(Rc["B"]), z(F)))],
        [f"ZOOM x12 (nearest) rows {y0}-{y1 - 1} cols {x0}-{x1 - 1} of the FULL replay", "red = FN, blue = FP", verdict], s0, scale=12)
    vid(f"c10_{task}_4b_ZOOM_FULL_{s0}-{end}_orig_c9_A_B_C_rep_s{s}", [("original", z(F))] + [(f"{a} s{s}", z(rp(a, s)["B"])) for a in ["C9"] + arms],
        [f"ZOOM x12 rows {y0}-{y1 - 1} cols {x0}-{x1 - 1}, all arms, same seed s{s}"], s0, scale=12)
    index.append(dict(name=f"{task}_selection", candidate=cand, qualifies=ok, roles={k: v["seed"] for k, v in roles.items()}))
json.dump(index, open(os.path.join(OUT, "video_index10.json"), "w"), indent=1, default=str)
print(json.dumps([(x["name"], x.get("ok"), x.get("frames"), x.get("fps_1x")) for x in index], indent=0))
