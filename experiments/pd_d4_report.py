"""Path D4 final report: renders and TEST tables for runs chosen on val (run only after selection is final).
usage: pd_d4_report.py RUN [RUN ...]  -> runs/path_d/d4/renders/<run>__*.png, runs/path_d/d4/report_<runs>.json"""
import json, os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
from ca import dlca
import pd_d4, pd_vis

R = os.path.join(ROOT, "runs", "path_d", "d4")


def fresh(net, w, n):
    V = pd_d4.X(); return np.concatenate([V[w:w + 1], pd_d4.roll(net, w, n)])


def render_run(run):
    d = os.path.join(R, run); r = json.load(open(os.path.join(d, "result.json"))); net = dlca.load(os.path.join(d, "circuit.logic.json"))["net"]
    V = pd_d4.X(); bench = pd_d4.BENCH[r["scene"]]
    windows = [(w, n, "train") for w, n in r["train_windows"]] + [(w, 8, "val") for w in bench["val"]] + [(w, 8, "test") for w in bench["test"]] + [(bench["test16"], 16, "test")]
    for w, n, lab in windows:
        B = fresh(net, w, n); F = V[w:w + n + 1]
        pd_vis.contact_sheet(os.path.join(R, "renders", f"{run}__{lab}_{w}.png"), F, B, list(range(w, w + n + 1)), [lab] * (n + 1), every=1,
                             title=f"{run}: fresh start at {w} (2 observed prefix frames before it), {lab} window, {n} free steps")
    a = r["train_windows"][0][0]; n = 32 if r["scene"] == "3998g" else 64; B = fresh(net, a, n); F = V[a:a + n + 1]
    lab = []
    for i in range(n + 1):
        f = a + i; lab.append("train" if any(w <= f <= w + m for w, m in r["train_windows"]) else ("val" if any(w <= f <= w + 8 for w in bench["val"]) else ("test" if any(w <= f <= w + 8 for w in bench["test"]) else "unseen")))
    pd_vis.contact_sheet(os.path.join(R, "renders", f"{run}__continuous_{a}.png"), F, B, list(range(a, a + n + 1)), lab, every=2,
                         title=f"{run}: continuous from {a}, no re-initialisation ({n} steps)")
    return dict(run=run, gates=r["simplified"]["gates"], depth=r["simplified"]["depth"], test=r["test"], test16=r["test16"],
                continuous={k: v for k, v in r.items() if k.startswith("continuous")})


if __name__ == "__main__":
    out = [render_run(x) for x in sys.argv[1:]]
    json.dump(out, open(os.path.join(R, "report_test.json"), "w"), indent=1, default=float)
    for o in out:
        print("==", o["run"], "gates", o["gates"])
        for w, m in list(o["test"].items()) + list(o["test16"].items()):
            print(f"   test {w}: err {m['err']:.3f} hold {m['hold']:.3f} readable {m['readable_err']:.3f} table {m['table_err']:.3f} | change F1 {m['lag1']['f1']:.2f} (prec {m['lag1']['prec']:.2f} rec {m['lag1']['rec']:.2f}, mass {m['lag1']['mass'] if m['lag1']['mass'] is None else round(m['lag1']['mass'], 2)}) | IoU {m['iou']:.2f} boundary F1 {m['boundary_f1']:.2f} | terminal period {m['terminal_period_pred']} (true {m['terminal_period_true']})")
        for k, m in o["continuous"].items():
            print(f"   {k}: err {m['err']:.3f} hold {m['hold']:.3f} | change F1 {m['lag1']['f1']:.2f} | IoU {m['iou']:.2f} bF1 {m['boundary_f1']:.2f} | first IoU<0.5 {m['first_iou_below_0_5']} | terminal period {m['terminal_period_pred']} | entropy {m['entropy_pred']:.2f} vs {m['entropy_true']:.2f}")
