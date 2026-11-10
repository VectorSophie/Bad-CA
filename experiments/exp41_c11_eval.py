"""Exp41 (Campaign 11): evaluate exports (nca-v1 JSON or ncab) on a Campaign 11 target with ca/diag10 full_report
(+ thin-part/gap recall), the C9 window rule where a window exists, and optionally the independent Node replay.
usage: exp41_c11_eval.py OUT_JSON TARGET EXPORT [EXPORT ...] [--node]   (replay from the export alone)"""
import hashlib, json, os, subprocess, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca.diag10 import full_report
from ca.pipeline import load_export, replay
from exp40_c11 import target

WIN = {"main": (40, 0.50), "confirm": (44, 0.65)}


def evaluate(path, tname, node):
    F = target(tname); B, V, rep = replay(path, node=node); assert len(B) == len(F) and (B[0] == F[0]).all()
    r = full_report(B, V, F, (WIN[tname][0], WIN[tname][0] + 15) if tname in WIN else None); r.pop("err_curve")
    r["err_curve"] = [round(float((B[t] != F[t]).mean()), 4) for t in range(len(F))]
    if tname in WIN:
        from ca.motion9 import window_report
        t0, thr = WIN[tname]; w = window_report(B[t0:], F[t0:])
        r["c9_rule"] = dict(err_over_hold=w["err_over_hold"], lag4_f1=w["lag4"]["f1_tol"], pass_C=bool(w["err_over_hold"] <= 0.70 and w["lag4"]["f1_tol"] >= thr and w["lag4"]["prec_tol"] >= 0.40))
    if tname.startswith("L"):   # long intervals: sustained-motion gate per third (docs/CAMPAIGN_11.md section 3)
        from ca.motion9 import lag_terms
        idx = np.array_split(np.arange(len(F)), 3); r["thirds_lag4"] = []
        for ix in idx:
            a, b = ix[0], ix[-1] + 1; lt = lag_terms(B[a:b], F[a:b], 4); r["thirds_lag4"].append(dict(f1=lt["f1_tol"], mass=lt["mass_ratio"]))
        e = [x["err"] for x in r["thirds"]]
        r["sustained"] = bool(all(x["f1"] >= 0.5 and x["mass"] is not None and 0.5 <= x["mass"] <= 1.5 for x in r["thirds_lag4"]) and (e[2] <= 3 * e[0] or e[2] <= 0.03))
    r["verify"] = {k: v for k, v in rep.items() if k not in ("T", "H", "W")}
    r["file_bytes"] = os.path.getsize(path); r["export"] = os.path.relpath(path, ROOT)
    return r, B, V


if __name__ == "__main__":
    out, tname = sys.argv[1], sys.argv[2]; node = "--node" in sys.argv; paths = [p for p in sys.argv[3:] if not p.startswith("--")]
    res = {}
    for p in paths:
        r, B, V = evaluate(p, tname, node); res[os.path.relpath(p, ROOT)] = r
        a = r["all"]; print(f"{os.path.basename(p)}: err {a['err']:.4f} band {a['err_band']:.4f} thirds {[round(x['err'], 4) for x in r['thirds']]} lag1 {r['lag1']['f1_tol']:.3f} "
                            f"mass1 {r['lag1']['mass_ratio']:.2f} blips {r['blips_spurious']} specks {r['small_cc']['pred_specks']}/{r['small_cc']['true_specks']} "
                            f"thinFG {r['thin']['thin_fg_recall']:.3f} thinBG {r['thin']['thin_bg_recall']:.3f}" + (f" passC {r['c9_rule']['pass_C']}" if 'c9_rule' in r else "")
                            + (f" node mism {r['verify'].get('binary_mismatch_px')}" if node else "") + (f" thirds_lag4 {[(round(x['f1'], 2), round(x['mass'] or 0, 2)) for x in r['thirds_lag4']]} sustained {r['sustained']}" if "sustained" in r else ""), flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(res, open(out, "w"), indent=1, default=float)
