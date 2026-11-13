"""Stage 0 report: aggregates runs/context_s0/results.json into runs/context_s0/summary.json and prints tables.
All aggregates are unweighted means over windows of the stated subset ("moving" = true change >= 0.005 per transition,
fixed in the manifest). Gap recovery g = (E[K=1] - E[K]) / (E[K=1] - E[per-window]) on mean one-step error.
usage: s0_report.py [--test]   (without --test only VAL is printed; K is selected on VAL first)"""
import json, math, os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path[:0] = [ROOT, os.path.dirname(os.path.abspath(__file__))]
import s0_codebook as s0
R = json.load(open(os.path.join(s0.OUT, "results.json"))); MAN = json.load(open(s0.MAN))
TEST = "--test" in sys.argv


def subset(mode, split=None, group=None, moving=True, lo=None, hi=None):
    out = []
    for r in R["eval"]:
        e = r["e"]
        if r["mode"] != mode or (split and e["split"] != split) or (group and e["group"] != group): continue
        if moving and r["true_change"] < s0.MOVING: continue
        if lo is not None and not (lo <= r["true_change"] < hi): continue
        out.append(r)
    return out


def agg(rows, K):
    if not rows: return None
    m = lambda f: float(np.mean([f(r) for r in rows]))
    E1 = m(lambda r: r["K1"]["oracle1"]["one"]["err"]); Ep = m(lambda r: r["perwin"]["one"]["err"]); g = lambda E: (E1 - E) / (E1 - Ep) if E1 > Ep else float("nan")
    d = dict(n=len(rows), hold=m(lambda r: r["perwin"]["one"]["hold"]), K1=E1, perwin=Ep,
             perwin_roll=m(lambda r: r["perwin"]["roll"]["err"]), hold_roll=m(lambda r: r["perwin"]["roll"]["hold"]),
             shift_oracle=m(lambda r: r["shift_oracle"]["one"]["err"]), shift_obs=m(lambda r: r["shift_obs"]["one"]["err"]),
             shift_oracle_roll=m(lambda r: r["shift_oracle"]["roll"]["err"]), shift_obs_roll=m(lambda r: r["shift_obs"]["roll"]["err"]))
    for a in ("oracle1", "oracle_roll", "obs", "knn", "prior"):
        if a not in rows[0][f"K{K}"]: continue
        x = lambda r: r[f"K{K}"][a]
        d[a] = dict(one=m(lambda r: x(r)["one"]["err"]), gap=g(m(lambda r: x(r)["one"]["err"])),
                    beats_hold_one=m(lambda r: x(r)["one"]["err"] < x(r)["one"]["hold"]), one_f1=m(lambda r: x(r)["one"]["f1"]),
                    one_prec=m(lambda r: x(r)["one"]["prec"]), one_rec=m(lambda r: x(r)["one"]["rec"]),
                    roll=m(lambda r: x(r)["roll"]["err"]), beats_hold_roll=m(lambda r: x(r)["roll"]["err"] < x(r)["roll"]["hold"]),
                    roll_f1=m(lambda r: x(r)["roll"]["f1"] or 0), roll_iou=m(lambda r: x(r)["roll"]["iou"]), roll_bf1=m(lambda r: x(r)["roll"]["bf1"]),
                    roll_h=[m(lambda r: x(r)["roll"][f"err_h{h}"]) for h in (1, 4, 8, 15)],
                    frozen=m(lambda r: x(r)["roll"]["period"] == 1), match_oracle=m(lambda r: x(r)["code"] == r[f"K{K}"]["oracle1"]["code"]))
    d["perwin_beats_hold_one"] = m(lambda r: r["perwin"]["one"]["err"] < r["perwin"]["one"]["hold"])
    d["perwin_roll_f1"] = m(lambda r: r["perwin"]["roll"]["f1"] or 0)
    return d


def fmt(d, K):
    o = d["oracle1"]
    return (f'n={d["n"]:2d} hold {d["hold"]:.4f} K1 {d["K1"]:.4f} pw {d["perwin"]:.4f} | K{K} oracle {o["one"]:.4f} gap {o["gap"]:.2f} '
            f'beat {o["beats_hold_one"]:.2f} | obs {d["obs"]["one"]:.4f} gap {d["obs"]["gap"]:.2f} ' + (f'knn {d["knn"]["one"]:.4f} ' if "knn" in d else "")
            + f'| roll orc {o["roll"]:.4f} (hold {d["hold_roll"]:.4f}) beat {o["beats_hold_roll"]:.2f} F1 {o["roll_f1"]:.2f} IoU {o["roll_iou"]:.2f}')


S = dict(train=R["train"], val={}, select={}, test={}, regional={}, info={}, near_dup={}, seeds={})
for mode in s0.NBITS:
    S["val"][mode] = {K: agg(subset(mode, "val", "cross_scene"), K) for K in s0.KS}
    print(f"\n== VAL cross-scene moving windows, {mode} (train fit: {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in R['train'][mode].items()})})")
    for K in s0.KS: print(f"K={K:2d}", fmt(S["val"][mode][K], K))
    gaps = {K: S["val"][mode][K]["oracle1"]["gap"] for K in s0.KS}; mx = max(gaps.values())
    S["select"][mode] = min(K for K in s0.KS if gaps[K] >= 0.9 * mx)
    print("selected K (val):", S["select"][mode], {K: round(v, 3) for K, v in gaps.items()})
    sp = {}
    for K in s0.KS:
        v = [r["val_oracle1_err"] for r in json.load(open(os.path.join(s0.OUT, "em_restarts.json"))) if r["mode"] == mode and r["K"] == K and r["units"] == "global"]
        sp[K] = dict(min=min(v), max=max(v), sd=float(np.std(v)))
    S["seeds"][mode] = sp; print("restart spread of val oracle error:", {K: f'{x["min"]:.4f}-{x["max"]:.4f}' for K, x in sp.items()})

if TEST:
    for mode in s0.NBITS:
        S["test"][mode] = {}
        for grp in ("cross_scene", "same_scene", "cut"):
            S["test"][mode][grp] = {K: agg(subset(mode, "test", grp), K) for K in s0.KS}
            print(f"\n== TEST {grp} moving, {mode}")
            for K in s0.KS: print(f"K={K:2d}", fmt(S["test"][mode][grp][K], K))
        Ks = S["select"][mode]
        S["test"][mode]["strata"] = {f"{lo}-{hi}": agg(subset(mode, "test", None, True, lo, hi), Ks) for lo, hi in s0.STRATA if subset(mode, "test", None, True, lo, hi)}
        S["test"][mode]["all_test_nonmoving_count"] = sum(1 for r in R["eval"] if r["mode"] == mode and r["e"]["split"] == "test" and r["true_change"] < s0.MOVING)
        print("strata (selected K):", {k: (v["n"], round(v["oracle1"]["gap"], 2), round(v["obs"]["gap"], 2)) for k, v in S["test"][mode]["strata"].items()})
    # regional diagnostic (m0, oracle per region)
    for x in R["regional"]:
        if x["e"]["split"] not in ("val", "test") or x["e"]["group"] == "cut": continue
        key = f'{x["e"]["split"]}|{x["grid"][0]}x{x["grid"][1]}|K{x["K"]}|{x["codebook"]}'
        rr = next(r for r in R["eval"] if r["mode"] == "m0" and r["e"]["start"] == x["e"]["start"])
        if rr["true_change"] < s0.MOVING: continue
        S["regional"].setdefault(key, []).append((x["one"]["err"], x["roll"]["err"], x["roll"]["f1"] or 0, rr[f'K{x["K"]}']["oracle1"]["one"]["err"], rr["K1"]["oracle1"]["one"]["err"], rr["perwin"]["one"]["err"]))
    print("\n== regional (moving, val+test cross/same scene): one-step, rollout, roll F1, global-oracle one-step, K1, perwin; bits/window")
    for k, v in sorted(S["regional"].items()):
        a = np.mean(v, 0); g, K = k.split("|")[1], int(k.split("|")[2][1:]); nreg = int(g[0]) * int(g[2])
        S["regional"][k] = dict(n=len(v), one=a[0], roll=a[1], roll_f1=a[2], global_oracle=a[3], K1=a[4], perwin=a[5], gap=(a[4] - a[0]) / (a[4] - a[5]), bits=nreg * math.log2(K))
        print(k, len(v), " ".join(f"{t:.4f}" for t in a), f"gap {S['regional'][k]['gap']:.2f} bits {nreg * math.log2(K):.0f}")
    # information cost on the TEST cross-scene moving windows, m0, selected K
    import numpy as np_
    cb = np_.load(os.path.join(s0.OUT, "codebooks.npz"))
    for mode in s0.NBITS:
        rows = subset(mode, "test", "cross_scene"); Ks = S["select"][mode]
        book = sum(s0.table_bits(cb[f"{mode}_K{Ks}_global"][k])["flip_bits"] for k in range(Ks))
        book_direct = sum(s0.table_bits(cb[f"{mode}_K{Ks}_global"][k])["direct_bits"] for k in range(Ks))
        pw = [r["perwin"]["bits"]["flip_bits"] for r in rows]; pwd = [r["perwin"]["bits"]["direct_bits"] for r in rows]
        S["info"][mode] = dict(K=Ks, codebook_flip_bits=book, codebook_direct_bits=book_direct, assign_bits_per_window=math.log2(Ks),
                               perwin_flip_bits_mean=float(np_.mean(pw)), perwin_direct_bits_mean=float(np_.mean(pwd)), shift_bits_per_window=math.log2(49),
                               initial_state_bits=2 * 36 * 48, frames_bits_raw_per_window=16 * 36 * 48, n=len(rows))
        print(f"\ninfo {mode}: K={Ks} codebook {book:.0f} bits (direct form {book_direct:.0f}); per-window table {np_.mean(pw):.0f} bits (direct {np_.mean(pwd):.0f}); assign {math.log2(Ks):.1f} bits/window")
    S["near_dup"] = {k: v["near_dup"] for k, v in R["near_dup"].items()}; print("near-dup windows:", sum(S["near_dup"].values()), "of", len(S["near_dup"]))
json.dump(S, open(os.path.join(s0.OUT, "summary.json" if TEST else "summary_val.json"), "w"), indent=1, default=float)
