"""Track G CPU evaluation + summary. Runs eval_win on every finished window run (and the zero-shot cross-video
evaluation of the two 100 % rules), eval_fit on every finished fit run, then prints and writes runs/gen/summary.json.
usage: gen_eval.py"""
import glob, json, os, sys
from multiprocessing import Pool
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path[:0] = [ROOT, os.path.dirname(os.path.abspath(__file__))]
import gen_train as G
OUT = os.path.join(ROOT, "runs", "gen")


def ew(a):
    run, on = a
    try: return run, on, G.eval_win(run, on)
    except Exception as e: return run, on, dict(error=repr(e))


def ef(run):
    try: return run, G.eval_fit(run)
    except Exception as e: return run, dict(error=repr(e))


if __name__ == "__main__":
    wins = sorted(d for d in glob.glob(os.path.join(OUT, "win", "*")) if os.path.exists(os.path.join(d, "rule.npz")))
    jobs = [(w, None) for w in wins if not os.path.exists(os.path.join(w, "eval.json"))]
    for src, dst in (("bad_apple_spread_f1.0", "fantasmagorie"), ("fantasmagorie_spread_f1.0", "bad_apple")):
        w = os.path.join(OUT, "win", src)
        if os.path.exists(os.path.join(w, "rule.npz")) and not os.path.exists(os.path.join(w, f"eval_on_{dst}.json")): jobs.append((w, dst))
    fits = sorted(d for d in glob.glob(os.path.join(OUT, "fit", "*")) if os.path.exists(os.path.join(d, "model.ncab")) and not os.path.exists(os.path.join(d, "eval.json")))
    with Pool(32) as p:
        for r in p.imap_unordered(ew, jobs): print("win", os.path.basename(r[0]), r[1], "error" in r[2] and r[2]["error"] or "ok", flush=True)
        for r in p.imap_unordered(ef, fits): print("fit", os.path.basename(r[0]), "error" in r[1] and r[1]["error"] or "ok", flush=True)
    S = dict(win={}, cross={}, fit={})
    for w in wins:
        c = json.load(open(os.path.join(w, "config.json"))); e = json.load(open(os.path.join(w, "eval.json")))
        S["win"][os.path.basename(w)] = dict(ds=c["ds"], rep=c["rep"], frac=c["frac"], mode=c["mode"], stride=c["stride"], seed=c["seed"], n_train_windows=c["n_train_windows"],
                                              n_train_frames=c["n_train_frames"], val=e["val"]["agg"], test=e["test"]["agg"])
        for f in glob.glob(os.path.join(w, "eval_on_*.json")):
            S["cross"][os.path.basename(w) + " -> " + f.split("eval_on_")[1][:-5]] = json.load(open(f))["test"]["agg"]
    for d in sorted(glob.glob(os.path.join(OUT, "fit", "*", "eval.json"))):
        e = json.load(open(d)); cfg = json.load(open(os.path.join(os.path.dirname(d), "config.json")))
        S["fit"][os.path.basename(os.path.dirname(d))] = dict(ds=cfg["ds"], phases=cfg["phases"], Fh=cfg["Fh"], steps=cfg["steps"], frames=e["T"],
                                                              recon={k: e["recon"][k] for k in ("err", "hold", "f1", "iou", "bf1", "collapse", "blip_pred", "blip_true")},
                                                              cont={k: e["cont"][k] for k in ("err", "hold", "f1", "iou")} if "cont" in e else None, fresh=e["fresh"], sizes=e["sizes"])
    # failure-aware view of the full-film fit: per atlas window (Path V rules applied to the film, runs/shape/atlas_fantasmagorie.json)
    fz = os.path.join(OUT, "fit", "fa_full", "replay_bits.npz")
    if os.path.exists(fz):
        from ca import datasets
        z = np.load(fz); B = np.unpackbits(z["bits"])[:int(np.prod(z["shape"]))].reshape(tuple(z["shape"])); Fv = datasets.frames("fantasmagorie")[:len(B)]
        A = json.load(open(os.path.join(ROOT, "runs", "shape", "atlas_fantasmagorie.json")))["windows"]; per = {}
        for w in A:
            if w["end"] > len(B): continue
            e = float((B[w["start"] + 1:w["end"]] != Fv[w["start"] + 1:w["end"]]).mean()); h = float((Fv[w["start"]:w["end"] - 1] != Fv[w["start"] + 1:w["end"]]).mean())
            per.setdefault(w["tag"], []).append((e, h, w["start"]))
        S["fa_full_by_tag"] = {t: dict(n=len(v), err=float(np.mean([x[0] for x in v])), err_p90=float(np.quantile([x[0] for x in v], 0.9)), change=float(np.mean([x[1] for x in v])),
                                       worst=sorted(v)[-3:]) for t, v in per.items()}
        S["fa_full_frame_err"] = float((B[1:] != Fv[1:len(B)]).mean())
        print("\n== fa_full by atlas tag:", {t: (v["n"], round(v["err"], 4), round(v["change"], 4)) for t, v in S["fa_full_by_tag"].items()})
    json.dump(S, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=float)
    print("\n== window runs (test; novel-scene subset in brackets)")
    for k, v in sorted(S["win"].items()):
        t = v["test"]; nv = t.get("novel_scene") or {}
        print(f'{k:34s} trainW {v["n_train_windows"]:4d} | val err {v["val"]["err"]:.4f}/{v["val"]["hold"]:.4f} | test err {t["err"]:.4f} hold {t["hold"]:.4f} '
              f'beat {t["beats_hold"]:.2f} f1 {t["f1"]:.3f} iou {t["iou"]:.3f} 1step {t["onestep_err"]:.4f}/{t["onestep_hold"]:.4f} '
              f'[novel n={t.get("novel_n")} err {nv.get("err", float("nan")):.4f} hold {nv.get("hold", float("nan")):.4f}]')
    print("\n== cross-video zero-shot (test windows of the other video)")
    for k, t in S["cross"].items(): print(f'{k:50s} err {t["err"]:.4f} hold {t["hold"]:.4f} beat {t["beats_hold"]:.2f} f1 {t["f1"]:.3f}')
    print("\n== fits")
    for k, v in S["fit"].items():
        r, c, f = v["recon"], v["cont"], v["fresh"]
        print(f'{k:22s} {v["phases"]:12s} Fh {v["Fh"]:3d} params {v["sizes"]["params"]:7d} | recon err {r["err"]:.4f} (hold {r["hold"]:.4f}) f1 {r["f1"]:.3f} iou {r["iou"]:.3f} blip {r["blip_pred"]:.4f}/{r["blip_true"]:.4f} '
              f'| cont err {c["err"] if c else float("nan"):.4f} hold {c["hold"] if c else float("nan"):.4f} f1 {c["f1"] if c else float("nan"):.3f} | fresh err {f["err"]:.4f} vs carried {f["carried_err"]:.4f} hold {f["hold"]:.4f}')
