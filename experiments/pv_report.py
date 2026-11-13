"""Path V0 report over runs/path_v/v0_test.json (test clips, scored once). Every number is an unweighted mean over
clips (all clips have 31 predicted frames, so the clip mean is also the frame-weighted mean), per atlas category and
over all categories (macro = mean of the category means). F1 = exact-pixel lag-1 change F1 (pd_vis).
NCA reference: the featured whole-song Neural CA replayed continuously from frame 0 (node player), cut to each clip;
it is NOT restarted at the clip (a different, much richer setting), shown only to calibrate how high lag-1 F1 gets.
usage: pv_report.py [NCA_F32]  -> runs/path_v/v0_summary.json"""
import json, os, sys
from collections import defaultdict
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path[:0] = [ROOT, os.path.dirname(os.path.abspath(__file__))]
import pv_v0
R = json.load(open(os.path.join(pv_v0.OUT, "v0_test.json"))); CATS = pv_v0.CATS
K = lambda r: (r["method"], tuple(r["grid"]) if r["grid"] else None, r["every"])
F = ("err", "hold", "f1", "prec", "rec", "iou", "bf1", "cc_abs", "hole_excess", "bits")


def agg(rows):
    d = {f: float(np.mean([r[f] for r in rows])) for f in F}; d["collapse_frac"] = float(np.mean([r["collapse"] is not None for r in rows])); d["n"] = len(rows)
    if "extrap" in rows[0]:
        for f in ("err", "hold", "f1", "iou"): d["extrap_" + f] = float(np.mean([r["extrap"][f] for r in rows]))
        d["extrap_beats_hold"] = float(np.mean([r["extrap"]["err"] < r["extrap"]["hold"] for r in rows]))
    return d


by = defaultdict(lambda: defaultdict(list))
for r in R: by[K(r)][r["clip"]["cat"]].append(r)
S = {"per_cat": {}, "macro": {}}
for k, cats in by.items():
    name = "|".join(map(str, k)); S["per_cat"][name] = {c: agg(v) for c, v in cats.items()}
    S["macro"][name] = {f: float(np.mean([S["per_cat"][name][c][f] for c in cats])) for f in S["per_cat"][name][CATS[0]] if f != "n"}

# NCA reference on the same clips
if len(sys.argv) > 1:
    X = pv_v0.video(); Vn = np.fromfile(sys.argv[1], np.float32).reshape(len(X), 36, 48); Bn = (Vn > 0.5).astype(np.uint8); Bn[0] = X[0]
    clips = json.load(open(pv_v0.MAN))["clips"]; nca = defaultdict(list)
    for c in clips:
        if c["split"] != "test": continue
        s = c["start"]; m = pv_v0.metrics(Bn[s + 1:s + pv_v0.N], X[s:s + pv_v0.N]); m["bits"] = 0.0; nca[c["cat"]].append(m)
    S["per_cat"]["nca_reference"] = {c: {f: float(np.mean([m[f] for m in v])) for f in ("err", "hold", "f1", "prec", "rec", "iou", "bf1")} for c, v in nca.items()}
    S["macro"]["nca_reference"] = {f: float(np.mean([S["per_cat"]["nca_reference"][c][f] for c in CATS])) for f in ("err", "hold", "f1", "prec", "rec", "iou", "bf1")}


def show(names, cats=CATS, fields=("err", "f1", "iou", "bf1", "bits")):
    print(f'{"config":28s}' + "".join(f"{c[:10]:>32s}" for c in cats) + f'{"macro":>32s}')
    for n in names:
        if n not in S["per_cat"]: continue
        cells = [" ".join(f'{S["per_cat"][n][c][f]:.3f}' if f != "bits" else f'{S["per_cat"][n][c].get(f, 0):5.0f}' for f in fields if f in S["per_cat"][n][c]) for c in cats]
        mac = " ".join(f'{S["macro"][n][f]:.3f}' if f != "bits" else f'{S["macro"][n].get(f, 0):5.0f}' for f in fields if f in S["macro"][n])
        print(f"{n:28s}" + "".join(f"{x:>32s}" for x in cells) + f"{mac:>32s}")


print("fields: err f1 iou bf1 bits")
order = ["hold|None|None", "nca_reference", "keyframe8|None|None", "keyframe16|None|None", "keyframe31|None|None"] + \
        [f"{m}|{g}|{e}" for m in ("codebook",) for g in (None,) for e in (4, 8, 16, None)] + \
        [f"{m}|{g}|{e}" for m in ("block", "vcaE", "vcaL", "vcaL_v") for g in ((1, 1), (2, 2), (3, 3)) for e in (4, 8, 16, None)]
show(order)

# matched budget: for each VCA config, the best block / keyframe / codebook config with bits <= its bits (by macro err)
print("\nmatched budget (macro): VCA config vs best cheaper-or-equal baseline")
mb = {}
for n in order:
    if not n.startswith("vca"): continue
    b = S["macro"][n]["bits"]
    cands = [m for m in order if m.split("|")[0] in ("block", "keyframe8", "keyframe16", "keyframe31", "codebook", "hold") and m in S["macro"] and S["macro"][m]["bits"] <= b + 1e-9]
    best = min(cands, key=lambda m: S["macro"][m]["err"]); mb[n] = best
    print(f'{n:22s} bits {b:6.0f} err {S["macro"][n]["err"]:.4f} f1 {S["macro"][n]["f1"]:.3f} iou {S["macro"][n]["iou"]:.3f} | {best:22s} bits {S["macro"][best]["bits"]:6.0f} err {S["macro"][best]["err"]:.4f} f1 {S["macro"][best]["f1"]:.3f} iou {S["macro"][best]["iou"]:.3f}')
S["matched_budget"] = mb

print("\npersistence (one event fitted on frames 1-8, scored on unseen 9-31): err / hold / F1 / beats-hold fraction")
for n in order:
    if n.endswith("|None") and n.split("|")[0] in ("vcaL", "vcaL_v", "vcaE", "block", "codebook"):
        print(f"{n:22s}", " ".join(f'{c[:6]}: {S["per_cat"][n][c]["extrap_err"]:.3f}/{S["per_cat"][n][c]["extrap_hold"]:.3f}/{S["per_cat"][n][c]["extrap_f1"]:.2f}/{S["per_cat"][n][c]["extrap_beats_hold"]:.2f}' for c in CATS))

# Lagrangian vs Eulerian, paired per clip
print("\nlagrangian - eulerian (paired, all test clips): mean d err, d f1; clips where L better")
pair = {}
for g in ((1, 1), (2, 2), (3, 3)):
    for e in (4, 8, 16, None):
        L = {r["clip"]["start"]: r for r in by[("vcaL", g, e)][c] for c in CATS} if False else {r["clip"]["start"]: r for c in CATS for r in by[("vcaL", g, e)][c]}
        E = {r["clip"]["start"]: r for c in CATS for r in by[("vcaE", g, e)][c]}
        de = [L[s]["err"] - E[s]["err"] for s in L]; df = [L[s]["f1"] - E[s]["f1"] for s in L]
        pair[f"{g}|{e}"] = dict(d_err=float(np.mean(de)), d_f1=float(np.mean(df)), L_better=float(np.mean([x < 0 for x in de])))
        print(f"{str(g):8s} {str(e):5s} d_err {np.mean(de):+.4f} d_f1 {np.mean(df):+.3f} L better in {np.mean([x < 0 for x in de]):.2f}")
S["lag_vs_euler"] = pair

# start offset 0 vs 8 (same scene), vcaL 3x3 every 8
print("\nsame scene, start +0 vs +8 (vcaL 3x3 every 8): err pairs")
rows = {(r["clip"]["start"], r["clip"]["offset"]): r for c in CATS for r in by[("vcaL", (3, 3), 8)][c]}
pairs = [(rows[(s, 0)], rows[(s + 8, 8)]) for (s, o) in rows if o == 0 and (s + 8, 8) in rows]
print("n", len(pairs), "mean |d err|", round(float(np.mean([abs(a["err"] - b["err"]) for a, b in pairs])), 4), "corr", round(float(np.corrcoef([a["err"] for a, b in pairs], [b["err"] for a, b in pairs])[0, 1]), 3))

# hardest vs easier clips (by hold error), vcaL 3x3 every 8 and block 3x3 every 4
allr = [r for c in CATS for r in by[("vcaL", (3, 3), 8)][c]]; q = np.quantile([r["hold"] for r in allr], [0.25, 0.9])
for lab, sel in (("easiest 25% (hold err)", lambda r: r["hold"] <= q[0]), ("hardest 10%", lambda r: r["hold"] >= q[1])):
    for n in ("hold|None|None", "vcaL|(3, 3)|8", "block|(3, 3)|4", "keyframe16|None|None"):
        k = tuple(n.split("|")); rr = [r for c in CATS for r in by[(k[0], eval(k[1]), eval(k[2]))][c] if sel(r)]
        print(f"{lab:24s} {n:22s} n={len(rr):2d} err {np.mean([r['err'] for r in rr]):.4f} f1 {np.mean([r['f1'] for r in rr]):.3f} iou {np.mean([r['iou'] for r in rr]):.3f}")
json.dump(S, open(os.path.join(pv_v0.OUT, "v0_summary.json"), "w"), indent=1, default=float)
