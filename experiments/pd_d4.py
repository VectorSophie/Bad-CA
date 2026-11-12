"""Path D4: motion fidelity of visible-only logic CA on Bad Apple. Benchmark windows are frozen in BENCH before any D4
fit (scene continuity checked on contact sheets runs/path_d/d4/renders/scene*_every4.png). Every window is 8 frames
[w, w+8] with the 2-frame observation prefix [w-2, w-1]; test16 is 16 frames.
Selection uses VAL only; TEST windows are scored and stored but never used to choose a configuration.
usage: pd_d4.py CONFIG SEED [BUDGET] [LOSS a,b,c] [K]   CONFIG in A | B | C | D | Aq (scene 3998), or clip:<name>"""
import json, os, sys, time
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import pd25, pd_vis

OUT = os.path.join(ROOT, "runs", "path_d", "d4"); P = 2; L = 8
BENCH = {   # frozen 2026-11-12 before any D4 training
    "3998": dict(desc="one continuous scene 3996-4182 (wipe at 4183); growth 3998-4030, near-static 4030-4060, second figure rises 4062-4090, "
                      "arm motion 4090-4110, slow 4110-4146, zoom 4150-4180",
                 train_pool=[3998, 4014, 4066, 4150], val=[4038, 4078, 4162], test=[4022, 4090, 4170], test16=4106, contiguous=(3998, 24)),
    "1221": dict(desc="teacup silhouette translating up-right 1228-1260 (cut at 1263)", train_pool=[1232], val=[1244], test=[], test16=1244),
    "4500": dict(desc="figure morphs then grows/deforms 4520-4560 (cut at 4565)", train_pool=[4520], val=[4532], test=[], test16=4544),
    "1499": dict(desc="wipe: line grows 1484-1504, bunting and figure wipe in 1506-1517", train_pool=[1500], val=[1508], test=[], test16=1516),
}
CONFIGS = {"A": [3998], "B": [3998, 4066], "C": [3998, 4014, 4066, 4150], "Aq": [3998]}
BENCH["3998g"] = dict(desc="amendment (registered before running, docs/path-d/D4_LOG.md): growth phase only", train_pool=[3998], val=[4014], test=[4022],
                      test16=4106, contiguous=(3998, 16))


def X():
    import pd_d3
    return pd_d3.video()


def data_for(config):
    V = X()
    if config in ("D", "G"):
        a, n = BENCH["3998" if config == "D" else "3998g"]["contiguous"]; return dict(train=V[a - P:a + n + 1][None], periodic=False, windows=dict(train=(P, n + 1))), [(a, n)]
    if config.startswith("clip:"):
        ws = BENCH[config[5:]]["train_pool"]
    else: ws = CONFIGS[config]
    return dict(train=np.stack([V[w - P:w + L + 1] for w in ws]), periodic=False, windows=dict(train=(P, L + 1))), [(w, L) for w in ws]


def roll(net, w, n):
    import pd_d3
    V = X(); return pd_d3.roll_net(net, V[w - P:w + 1], n, len(net["outputs"]))   # channel count from the netlist (k = 4 runs)


def score_window(net, w, n, base):
    V = X(); F = V[w:w + n + 1]; B = np.concatenate([F[:1], roll(net, w, n)])
    m = pd_vis.sequence_metrics(B, F, first_free=1)
    for k, f in base.items():
        Bb = np.concatenate([F[:1], f(V[w], n)]); mb = pd_vis.sequence_metrics(Bb, F, first_free=1)
        m[f"{k}_err"], m[f"{k}_f1"], m[f"{k}_iou"] = mb["err"], mb["lag1"]["f1"], mb["iou"]
    return m


def baselines(train_windows):
    """Fitted on the run's own training windows: memoryless 3x3 table (all windows), shift and the prior Readable Rules
    staged dlrule (first training window, as in round 2)."""
    from rr_gen import fit_shift, shift_roll
    from rr_scale import method, roll as rr_roll
    V = X(); w0, n0 = train_windows[0]; F0 = V[w0:w0 + n0 + 1]
    lut = pd25.lut_baseline([V[w:w + n + 1] for w, n in train_windows], False); d = fit_shift(F0)
    p = method("two_stage_staged", F0, 2, np.random.default_rng(0))
    return dict(shift=lambda v, n: shift_roll(v, d, n), table=lambda v, n: lut(v, n), readable=lambda v, n: rr_roll(p, v, np.zeros((0,) + v.shape, np.uint8), n, 2))


def run(config, seed, budget, loss=None, k=2, tag=""):
    from ca import dlca, logic_simplify as ls
    import jax
    pd25.K_HIDDEN = k; C = 1 + k
    name = f"{config.replace(':', '_')}_s{seed}_b{budget // 1000}k" + (f"_L{','.join(map(str, loss))}" if loss else "") + (f"_k{k}" if k != 2 else "") + tag
    d = os.path.join(OUT, name)
    if os.path.exists(os.path.join(d, "result.json")): print("exists", name); return
    os.makedirs(d, exist_ok=True); data, tw = data_for(config)
    lc = dict(zip("abc", loss)) if loss else None
    params, wires, log, secs = pd25.train(data, "burnin", seed, budget, k=k, loss_cfg=lc)
    raw = dlca.to_netlist(dlca.harden(params, wires), C); net = ls.simplify(ls.abc(ls.simplify(raw))); eq, _ = ls.equivalent(raw, net)
    scene = config[5:] if config.startswith("clip:") else ("3998g" if config == "G" else "3998"); bench = BENCH[scene]; base = baselines(tw)
    r = dict(config=config, scene=scene, seed=seed, budget=budget, loss=lc, k_hidden=k, prefix_frames=P, train_windows=tw, train_seconds=round(secs, 1),
             backend=jax.default_backend(), final_loss=log[-1], gates_raw=len(raw["gates"]), simplified=ls.stats(net), simplified_equivalent=eq)
    r["train"] = {str(w): score_window(net, w, n, base) for w, n in tw}
    r["val"] = {str(w): score_window(net, w, L, base) for w in bench["val"]}
    # TEST: scored and stored for the final report only; never read by selection code (select.py reads 'val' only)
    r["test"] = {str(w): score_window(net, w, L, base) for w in bench["test"]}
    r["test16"] = {str(bench["test16"]): score_window(net, bench["test16"], 16, base)}
    if scene == "3998g":  # continuous through train (3998-4014), val (4014-4022) and test (4022-4030)
        V = X(); F = V[3998:3998 + 33]; B = np.concatenate([F[:1], roll(net, 3998, 32)]); r["continuous_3998_32"] = pd_vis.sequence_metrics(B, F, first_free=1)
    if scene == "3998":   # continuous continuation from the first training window through val/test territory (64 steps)
        V = X(); F = V[3998:3998 + 65]; B = np.concatenate([F[:1], roll(net, 3998, 64)]); r["continuous_3998_64"] = pd_vis.sequence_metrics(B, F, first_free=1)
    init = np.zeros(X().shape[1:] + (C,), np.uint8); w0 = tw[0][0]; init[..., 0] = X()[w0 - P]
    dlca.export(os.path.join(d, "circuit.logic.json"), net, C, False, init_state=init, steps=P + 64, prefix=[X()[w0 - P + i + 1] for i in range(P)],
                meta=dict(config=config, seed=seed, budget=budget, loss=lc, gates_raw=len(raw["gates"]), source_frames=[w0 - P, w0 + 64]))
    np.savez_compressed(os.path.join(d, "params.npz"), **{f"{n_}_{i}": np.asarray(g) for n_ in ("perceive", "update") for i, g in enumerate(params[n_])})
    json.dump(r, open(os.path.join(d, "result.json"), "w"), indent=1, default=float); open(os.path.join(d, "log.jsonl"), "w").write("\n".join(json.dumps(x) for x in log) + "\n")
    v = r["val"]; print(name, "gates", r["simplified"]["gates"], "| val err", [round(x["err"], 3) for x in v.values()], "hold", [round(x["hold"], 3) for x in v.values()],
                        "f1", [round(x["lag1"]["f1"], 2) for x in v.values()], "iou", [round(x["iou"], 2) for x in v.values()], flush=True)


if __name__ == "__main__":
    cfg, seed = sys.argv[1], int(sys.argv[2]); budget = int(sys.argv[3]) if len(sys.argv) > 3 else 96_000
    loss = tuple(float(x) for x in sys.argv[4].split(",")) if len(sys.argv) > 4 and sys.argv[4] != "-" else None
    k = int(sys.argv[5]) if len(sys.argv) > 5 else 2
    run(cfg, seed, budget, loss, k)


def prune(run, tol=0.003, shape=False):
    """Unit tying scored on the run's TRAINING windows only (mean fresh-start error may rise <= tol); writes
    <run>__pruned/{result.json, circuit.logic.json} with val/test re-scored by the same code as the parent run."""
    from ca import dlca, logic_simplify as ls
    d = os.path.join(OUT, run); r = json.load(open(os.path.join(d, "result.json"))); net = dlca.load(os.path.join(d, "circuit.logic.json"))["net"]
    tw = [tuple(x) for x in r["train_windows"]]; V = X(); C = len(net["outputs"])
    def score(n_):
        ms = [pd_vis.sequence_metrics(np.concatenate([V[w:w + 1], roll(n_, w, m)]), V[w:w + m + 1], first_free=1) for w, m in tw]
        return float(np.mean([x["err"] for x in ms])), float(np.mean([x["iou"] for x in ms])), float(np.mean([x["lag1"]["f1"] for x in ms]))
    e0, i0, f0 = score(net)
    def ok(n_):   # shape=True: pixel error AND foreground IoU AND change F1 must be kept (D4-D, after pixel-only pruning degenerated)
        e, i, f = score(n_); return e <= e0 + tol and (not shape or (i >= i0 - 0.02 and f >= f0 - 0.03))
    Pv = []
    for w, m in tw:
        st = np.zeros(V.shape[1:] + (C,), np.uint8); st[..., 0] = V[w - P]
        for i in range(P + m):
            Pv.append(dlca.np_patches(st, False).reshape(-1, 9, C)); st = dlca.net_eval(net, Pv[-1].reshape(st.shape[:2] + (9, C)))
            if i < P: st[..., 0] = V[w - P + i + 1]
    Pv = np.concatenate(Pv); changed = True; trace = []
    while changed:
        changed = False; vals = ls.node_values(net, Pv); n_in = net["n_in"]
        for j in sorted(range(len(net["gates"])), key=lambda j: -abs(vals[n_in + j].mean() - 0.5)):
            if net["gates"][j][0] in (0, 15): continue
            cand = ls.tie(net, n_in + j, int(vals[n_in + j].mean() >= 0.5))
            if len(cand["gates"]) < len(net["gates"]) and ok(cand): net = cand; changed = True; trace.append(len(net["gates"])); break
    net = ls.simplify(ls.abc(net)); bench = BENCH[r["scene"]]; base = baselines(tw)
    o = dict(parent=run, tol=tol, criterion="error+IoU+changeF1" if shape else "pixel error only", gates_before=r["simplified"]["gates"], simplified=ls.stats(net), trace=trace, scene=r["scene"], train_windows=r["train_windows"],
             k_hidden=r["k_hidden"], seed=r["seed"], budget=r["budget"], loss=r["loss"], config=r["config"],
             train={str(w): score_window(net, w, m, base) for w, m in tw}, val={str(w): score_window(net, w, L, base) for w in bench["val"]},
             test={str(w): score_window(net, w, L, base) for w in bench["test"]}, test16={str(bench["test16"]): score_window(net, bench["test16"], 16, base)})
    if r["scene"] == "3998g": F = V[3998:3998 + 33]; o["continuous_3998_32"] = pd_vis.sequence_metrics(np.concatenate([F[:1], roll(net, 3998, 32)]), F, first_free=1)
    if r["scene"] == "3998": F = V[3998:3998 + 65]; o["continuous_3998_64"] = pd_vis.sequence_metrics(np.concatenate([F[:1], roll(net, 3998, 64)]), F, first_free=1)
    dd = os.path.join(OUT, run + ("__pruned_shape" if shape else "__pruned")); os.makedirs(dd, exist_ok=True); w0 = tw[0][0]; init = np.zeros(V.shape[1:] + (C,), np.uint8); init[..., 0] = V[w0 - P]
    dlca.export(os.path.join(dd, "circuit.logic.json"), net, C, False, init_state=init, steps=P + 64, prefix=[V[w0 - P + i + 1] for i in range(P)],
                meta=dict(parent=run, method=f"unit tying scored on training windows only (tol {tol}; {'error+IoU+change F1 kept' if shape else 'pixel error only'})", gates_raw=r["gates_raw"], source_frames=[w0 - P, w0 + 64]))
    json.dump(o, open(os.path.join(dd, "result.json"), "w"), indent=1, default=float); return o
