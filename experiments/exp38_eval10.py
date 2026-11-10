"""Exp38 (Campaign 10): evaluate fine-tuned NCAs and their C9 starting checkpoints FROM EXPORTS.
For each model: write an nca-v1 export, replay it from its first frame only (numpy), compute ca/diag10 full-span
metrics + the C9 window rule, and (--node) replay the export in the independent Node player.
Then apply the pre-registered rules of docs/CAMPAIGN_10.md: --pilot -> weight choice; otherwise -> improvement criteria.
usage: exp38_eval10.py RUN_DIR(under runs/) TASK [--node] [--pilot] [--which best|last]
writes runs/c10/eval/<run>_<which>.json, runs/c10/replays/<run>_<which>_<name>.npz, prints the decision."""
import hashlib, json, os, statistics as stt, subprocess, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca import nca
from ca.diag10 import full_report, shift_curve
from ca.motion9 import window_report
from c7_common import target

TASKS = {"main": ("ba48_4500_60", 40, 0.50), "confirm": ("ba48_1800_60", 44, 0.65)}
C10 = os.path.join(ROOT, "runs", "c10")
for d in ("eval", "replays", "exports"): os.makedirs(os.path.join(C10, d), exist_ok=True)


def evaluate(P, task, ex_path, meta, node):
    clip, t0, thr = TASKS[task]; F = target(clip)[:t0 + 16]; T = len(F)
    nca.export(ex_path, P, F[0], T, 4, meta); ex, Pl, f0 = nca.load(ex_path)
    B, V = nca.rollout_np(Pl, f0[None], T, 4)                                 # first frame only
    r = full_report(B, V, F, (t0, t0 + 15)); r.pop("err_curve")
    w = window_report(B[t0:], F[t0:]); w.pop("err_curve")
    r["c9_rule"] = dict(err_over_hold=w["err_over_hold"], lag4_f1=w["lag4"]["f1_tol"], lag4_prec=w["lag4"]["prec_tol"],
                        pass_C=bool(w["err_over_hold"] <= 0.70 and w["lag4"]["f1_tol"] >= thr and w["lag4"]["prec_tol"] >= 0.40))
    r["shift_posthoc_full"] = shift_curve(B, F, range(1, T))
    r["file_bytes"] = os.path.getsize(ex_path)
    if node:
        nb = ex_path + ".node.f32"
        o = json.loads(subprocess.run(["node", os.path.join(ROOT, "player", "nca_replay.mjs"), ex_path, nb], capture_output=True, text=True, check=True).stdout)
        Vn = np.fromfile(nb, np.float32).reshape(V.shape); os.remove(nb); Bn = np.concatenate([f0[None], Vn[1:] > 0.5]).astype(np.uint8)
        r["verify"] = dict(numpy_vs_node_binary_mismatch_px=int((Bn != B).sum()), numpy_vs_node_max_abs_visible=float(np.abs(Vn[1:] - V[1:]).max()),
                           near_threshold_values_1e3=int((np.abs(V[1:] - 0.5) < 1e-3).sum()), node_sha256=o["sha256"],
                           numpy_sha256=hashlib.sha256(np.ascontiguousarray(B, np.uint8).tobytes()).hexdigest())
    return r, B, V


def med(rows, f): return stt.median(f(r) for r in rows)


KEYS = dict(err=lambda r: r["all"]["err"], band=lambda r: r["all"]["err_band"], lag1=lambda r: r["lag1"]["f1_tol"], lag4=lambda r: r["lag4"]["f1_tol"],
            blips=lambda r: r["blips_spurious"], contourF1=lambda r: r["contour"]["f1"], fp=lambda r: r["all"]["fp"], fn=lambda r: r["all"]["fn"],
            e1=lambda r: r["thirds"][0]["err"], e2=lambda r: r["thirds"][1]["err"], e3=lambda r: r["thirds"][2]["err"], win_err=lambda r: r["window"]["err"],
            mass1=lambda r: r["lag1"]["mass_ratio"], mass4=lambda r: r["lag4"]["mass_ratio"], passC=lambda r: int(r["c9_rule"]["pass_C"]),
            win_lag4=lambda r: r["c9_rule"]["lag4_f1"])


def criteria(arm, base):
    """arm/base: lists of 3 reports, same seed order. Pre-registered rules (docs/CAMPAIGN_10.md)."""
    mb, ma = (lambda k: med(base, KEYS[k])), (lambda k: med(arm, KEYS[k]))
    c = dict(shape=ma("band") <= 0.9 * mb("band") and sum(a["all"]["err_band"] < b["all"]["err_band"] for a, b in zip(arm, base)) >= 2,
             motion=ma("lag1") >= mb("lag1") - 0.03 and ma("lag4") >= mb("lag4") - 0.03 and sum(KEYS["passC"](r) for r in arm) >= sum(KEYS["passC"](r) for r in base),
             no_local_trade=all(ma(k) <= 1.1 * mb(k) for k in ("e1", "e2", "e3")),
             no_new_flicker=ma("blips") <= max(1.5 * mb("blips"), mb("blips") + 10))
    c["improved"] = all(c.values()); return c


if __name__ == "__main__":
    run, task = sys.argv[1], sys.argv[2]; node = "--node" in sys.argv; which = sys.argv[sys.argv.index("--which") + 1] if "--which" in sys.argv else "best"
    runs = run.split(","); tag = runs[0].replace("/", "_")   # several run dirs (e.g. main + B+C follow-up) -> one table named after the first
    rows = {}
    for s in range(3):   # C9 baselines
        P = dict(np.load(os.path.join(ROOT, "runs", "c9", "nca", f"nca_{task}_C", f"best_seed{s}.npz")))
        r, B, V = evaluate(P, task, os.path.join(C10, "exports", f"c9_{task}_C_s{s}.nca.json"), dict(task=task, source="C9 unchanged", seed=s), node)
        rows[f"C9_s{s}"] = dict(arm="C9", seed=s, report=r, export=f"runs/c10/exports/c9_{task}_C_s{s}.nca.json", replay=f"runs/c10/replays/c9_{task}_s{s}.npz"); np.savez_compressed(os.path.join(C10, "replays", f"c9_{task}_s{s}.npz"), B=B, V=V)
    for rn in runs:
        D = os.path.join(ROOT, "runs", rn)
        for c in json.load(open(os.path.join(D, "config.json"))):
            P = dict(np.load(os.path.join(D, f"{which}_{c['name']}.npz"))); meta = json.load(open(os.path.join(D, f"{which}_{c['name']}.json")))
            name = f"{rn.replace('/', '_')}_{which}_{c['name']}"
            r, B, V = evaluate(P, task, os.path.join(C10, "exports", name + ".nca.json"), dict(meta, source=f"runs/{rn}/{which}_{c['name']}.npz"), node)
            rows[c["name"]] = dict(arm=c["name"].rsplit("_s", 1)[0], seed=c["seed"], lamB=c["lamB"], lamC=c["lamC"], train=meta, report=r,
                                   export=f"runs/c10/exports/{name}.nca.json", replay=f"runs/c10/replays/{name}.npz")
            np.savez_compressed(os.path.join(C10, "replays", name + ".npz"), B=B, V=V)
    arms = sorted({v["arm"] for v in rows.values()}, key=lambda a: (a != "C9", a)); by = {a: [rows[k]["report"] for k in sorted(rows) if rows[k]["arm"] == a] for a in arms}
    table = {a: {k: round(med(by[a], f), 5) for k, f in KEYS.items()} for a in arms}
    out = dict(run=run, task=task, which=which, rows=rows, median_table=table)
    if "--pilot" in sys.argv:
        A = table["A"]
        okB = [a for a in arms if a.startswith("B") and table[a]["lag1"] >= A["lag1"] - 0.03 and table[a]["lag4"] >= A["lag4"] - 0.03]
        okC = [a for a in arms if a.startswith("C") and a != "C9" and table[a]["err"] <= 1.05 * A["err"]]
        lam = lambda a: float(a[1:])
        out["choice"] = dict(B=min(okB, key=lambda a: table[a]["band"]) if okB else min((a for a in arms if a.startswith("B")), key=lam),
                             C=max(okC, key=lambda a: table[a]["lag1"]) if okC else min((a for a in arms if a.startswith("C") and a != "C9"), key=lam), okB=okB, okC=okC)
    else:
        out["criteria"] = {a: criteria(by[a], by["C9"]) for a in arms if a != "C9"}
    json.dump(out, open(os.path.join(C10, "eval", f"{tag}_{which}.json"), "w"), indent=1, default=float)
    hdr = list(KEYS); print("arm".ljust(10), *[k[:8].rjust(8) for k in hdr])
    for a in arms: print(a.ljust(10), *[f"{table[a][k]:8.4f}" for k in hdr])
    print(json.dumps(out.get("choice") or out.get("criteria"), indent=1))
