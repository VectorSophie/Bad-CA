"""Path V: re-stratify earlier results by the difficulty atlas (runs/path_v/atlas.json). Historical results are read,
never rewritten; the only new computation is cheap inference:
  - existing hard circuits (D4 G_s0, G_s1, B_s1 and the round-2 64-gate D3 circuit) run on EVERY atlas window
    (fresh start at the window start with their 2-frame observation prefix, 8 free steps, as in D4);
  - the featured Neural CA's continuous whole-song replay (node player output) scored per atlas window.
A window that does not align with the atlas grid gets the tag with the largest frame overlap and a cut flag if any
overlapping atlas window is cut_wipe.
usage: pv_reanalysis.py NCA_F32  -> runs/path_v/reanalysis.json"""
import glob, json, os, sys
from collections import defaultdict
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path[:0] = [ROOT, os.path.dirname(os.path.abspath(__file__))]
OUT = os.path.join(ROOT, "runs", "path_v")
A = json.load(open(os.path.join(OUT, "atlas.json")))["windows"]; TAGS = json.load(open(os.path.join(OUT, "atlas.json")))["tags"]


def tag(a, b):
    ov = defaultdict(int)
    for w in A:
        o = min(b, w["end"]) - max(a, w["start"])
        if o > 0: ov[w["tag"]] += o
    return max(ov, key=ov.get), "cut_wipe" in ov


def m(x): return float(np.mean(x)) if len(x) else None


def stage0():
    R = json.load(open(os.path.join(ROOT, "runs", "context_s0", "results.json")))["eval"]; out = defaultdict(lambda: defaultdict(list))
    for r in R:
        if r["mode"] != "m0" or r["true_change"] < 0.005: continue
        t, _ = tag(r["e"]["start"], r["e"]["end"]); o = r["K8"]["oracle1"]
        out[t]["K1"].append(r["K1"]["oracle1"]["one"]["err"]); out[t]["K8"].append(o["one"]["err"]); out[t]["pw"].append(r["perwin"]["one"]["err"])
        out[t]["obs"].append(r["K8"]["obs"]["one"]["err"]); out[t]["hold"].append(r["perwin"]["one"]["hold"]); out[t]["beat"].append(o["one"]["err"] < o["one"]["hold"])
        out[t]["roll_f1"].append(o["roll"]["f1"] or 0); out[t]["roll_beat"].append(o["roll"]["err"] < o["roll"]["hold"])
    res = {}
    for t, d in out.items():
        E1, E8, Ep, Eo = m(d["K1"]), m(d["K8"]), m(d["pw"]), m(d["obs"])
        res[t] = dict(n=len(d["K1"]), hold=m(d["hold"]), K1=E1, K8_oracle=E8, perwin=Ep, gap_oracle=(E1 - E8) / (E1 - Ep) if E1 > Ep else None,
                      gap_obs=(E1 - Eo) / (E1 - Ep) if E1 > Ep else None, beats_hold_onestep=m(d["beat"]), roll_f1=m(d["roll_f1"]), roll_beats_hold=m(d["roll_beat"]))
    return res


def g1():
    PRE = json.load(open(os.path.join(ROOT, "runs", "readable_rules", "gen", "prereg_windows.json")))["windows"]
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "runs", "readable_rules", "gen", "g1.jsonl"))]; out = defaultdict(lambda: defaultdict(list)); good = []
    for r in rows:
        w = PRE[r["window"]]
        for wn in ("adjacent", "distant", "cross"):
            x = r["transfer_" + wn]; t, cut = tag(*w[wn]); key = (r["method"], t)
            out[key]["d"].append(x["err"] - x["hold"]); out[key]["beat"].append(x["err"] < x["hold"]); out[key]["f1"].append(x["lag4_f1"] or 0)
            if x["err"] < x["hold"] and (x["lag4_f1"] or 0) >= 0.2: good.append(dict(method=r["method"], seed=r.get("seed"), window=w[wn], kind=wn, tag=t, err=x["err"], hold=x["hold"], f1=x["lag4_f1"]))
    return {f"{k[0]}|{k[1]}": dict(n=len(v["d"]), err_minus_hold=m(v["d"]), beats_hold=m(v["beat"]), lag4_f1=m(v["f1"])) for k, v in out.items()}, good


def d4_recorded():
    out = []
    for f in sorted(glob.glob(os.path.join(ROOT, "runs", "path_d", "d4", "*", "result.json"))):
        r = json.load(open(f)); name = os.path.basename(os.path.dirname(f))
        for split in ("train", "val", "test", "test16"):
            for w, x in r.get(split, {}).items():
                w = int(w); n = 16 if split == "test16" else 8; t, cut = tag(w, w + n + 1)
                out.append(dict(run=name, split=split, window=w, tag=t, cut=cut, err=x["err"], hold=x["hold"], f1=x["lag1"]["f1"], iou=x["iou"]))
    agg = defaultdict(list)
    for x in out:
        if x["split"] != "train": agg[(x["split"] if x["split"] != "test16" else "test", x["tag"])].append(x)
    return {f"{k[0]}|{k[1]}": dict(n=len(v), err=m([x["err"] for x in v]), hold=m([x["hold"] for x in v]), f1=m([x["f1"] for x in v]), beats_hold=m([x["err"] < x["hold"] for x in v])) for k, v in agg.items()}


def circuits_everywhere():
    """New inference: existing D3/D4 hard circuits on every atlas window (fresh start + 2-frame prefix, 8 free steps)."""
    from ca import dlca
    import pd_d3, pd_vis
    X = pd_d3.video(); nets = {"D4_G_s0 (503 g, growth 3998-4014)": "runs/path_d/d4/G_s0_b144k/circuit.logic.json",
                               "D4_G_s1 (476 g)": "runs/path_d/d4/G_s1_b144k/circuit.logic.json",
                               "D4_B_s1 (432 g, 3998+4066)": "runs/path_d/d4/B_s1_b96k/circuit.logic.json",
                               "D3_64g (3998-4006)": "ui/static/logic_examples/badapple_3998.logic.json"}
    res = {}
    for name, p in nets.items():
        net = dlca.load(os.path.join(ROOT, p))["net"]; C = len(net["outputs"]); rows = []
        for w in A:
            s = w["start"]
            if s < 2 or s + 9 > len(X): continue
            B = pd_d3.roll_net(net, X[s - 2:s + 1], 8, C); F = X[s:s + 9]; mm = pd_vis.sequence_metrics(np.concatenate([F[:1], B]), F, first_free=1)
            rows.append(dict(start=s, tag=w["tag"], err=mm["err"], hold=mm["hold"], f1=mm["lag1"]["f1"], iou=mm["iou"]))
        per = {t: dict(n=len(v), err=m([x["err"] for x in v]), hold=m([x["hold"] for x in v]), beats_hold=m([x["err"] < x["hold"] for x in v]),
                       f1=m([x["f1"] for x in v]), f1_ge_0_2_and_beats_hold=int(sum(x["f1"] >= 0.2 and x["err"] < x["hold"] for x in v)))
               for t in TAGS for v in [[x for x in rows if x["tag"] == t]] if v}
        top = sorted([x for x in rows if x["err"] < x["hold"]], key=lambda x: -x["f1"])[:8]
        res[name] = dict(per_tag=per, best_windows=top)
    return res


def nca(path):
    import pd_vis
    X = json.load(open(os.path.join(OUT, "atlas.json")))
    from exp45_c12 import target
    V = target("0:6572").astype(np.uint8); B = (np.fromfile(path, np.float32).reshape(len(V), 36, 48) > 0.5).astype(np.uint8); B[0] = V[0]
    rows = []
    for w in A:
        s, e = w["start"], w["end"]; F = V[s:e]; P = B[s + 1:e]
        mm = pd_vis.change_metrics(np.concatenate([B[s:s + 1], P]), F, 1)
        rows.append(dict(start=s, tag=w["tag"], err=float((P != F[1:]).mean()), f1=mm["f1"]))
    per = {t: dict(n=len(v), err=m([x["err"] for x in v]), f1=m([x["f1"] for x in v] if v else []),
                   err_p90=float(np.quantile([x["err"] for x in v], 0.9))) for t in TAGS for v in [[x for x in rows if x["tag"] == t and x["f1"] is not None]] if v}
    worst = sorted(rows, key=lambda x: -x["err"])[:10]
    return dict(per_tag=per, worst_windows=worst, frame_weighted_err=float(np.mean([x["err"] for x in rows])), macro_err=float(np.mean([v["err"] for v in per.values()])))


if __name__ == "__main__":
    out = dict(stage0=stage0()); out["g1"], out["g1_success_cases"] = g1(); out["d4_recorded"] = d4_recorded(); out["circuits_everywhere"] = circuits_everywhere()
    if len(sys.argv) > 1: out["nca"] = nca(sys.argv[1])
    json.dump(out, open(os.path.join(OUT, "reanalysis.json"), "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k not in ("g1_success_cases",)}, indent=0, default=lambda o: round(o, 3) if isinstance(o, float) else o)[:12000])
    print("G1 transfer successes (beats hold, lag-4 F1 >= 0.2):", len(out["g1_success_cases"]), [(x["method"], x["window"], x["tag"], round(x["f1"], 2)) for x in out["g1_success_cases"]])
