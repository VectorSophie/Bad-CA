"""Path D4 selection table: reads ONLY the 'val' (and 'train') entries of runs/path_d/d4/*/result.json.
usage: pd_d4_select.py [glob]"""
import glob, json, os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
pat = sys.argv[1] if len(sys.argv) > 1 else "*"
for f in sorted(glob.glob(os.path.join(ROOT, "runs/path_d/d4", pat, "result.json"))):
    r = json.load(open(f)); v = list(r["val"].values()); t = list(r["train"].values())
    m = lambda xs, k: np.mean([x[k] for x in xs]); f1 = lambda xs: np.mean([x["lag1"]["f1"] for x in xs]); rec = lambda xs: np.mean([x["lag1"]["rec"] for x in xs])
    print(f"{os.path.basename(os.path.dirname(f)):34s} gates {r['simplified']['gates']:4d} | train err {m(t,'err'):.3f} f1 {f1(t):.2f} iou {m(t,'iou'):.2f}"
          f" | VAL err {m(v,'err'):.3f} hold {m(v,'hold'):.3f} readable {m(v,'readable_err'):.3f} table {m(v,'table_err'):.3f}"
          f" | f1 {f1(v):.2f} rec {rec(v):.2f} (readable f1 {m(v,'readable_f1'):.2f}) iou {m(v,'iou'):.2f} bF1 {m(v,'boundary_f1'):.2f}"
          f" | per-window f1 {[round(x['lag1']['f1'], 2) for x in v]}")
