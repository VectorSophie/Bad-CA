"""Path D3: visible-only logic CA on real Bad Apple clips (48x36 binary, the project's target() preprocessing).
Clips are fixed in CLIPS before any training. Per clip (L = 16 frames, P = burn-in prefix):
  train  = frames [a, a+L]           (the circuit is fitted here; the P frames before a are the observation prefix)
  val    = frames [a+L, a+2L]        (model selection only)
  final  = frames [a+2L, a+3L]       (untouched until the end; reported once)
Tests: fresh-start transfer (visible frame + P-frame prefix, hidden 0, then free run) on val and final, and continuous
continuation (start at a, run through train, val and final with no re-initialisation).
Baselines on the same windows: hold, shift (best constant (dy, dx) fitted on train), memoryless 3x3 table fitted on train
(best memoryless radius-1 rule), and a Readable Rules staged dlrule (rr_scale.method 'two_stage_staged', k = 0, S = 2)
fitted on the same train window.
usage: pd_d3.py CLIP SETTING SEED [L] [BUDGET]  (GPU via gpuq)"""
import json, os, sys, time
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import pd25

OUT = os.path.join(ROOT, "runs", "path_d", "d3")
CLIPS = {   # fixed 2026-11-12 before any D3 fit (FINDINGS / docs/path-d/README.md section 9)
    "1221": "mostly translating silhouette (the 'translate' clip of Readable Rules rounds 2-3, A2 start)",
    "4500": "smooth contour deformation (the 'contour' clip, A2 start)",
    "3998": "growing silhouette on white (Readable Rules G1 scene 16 train window)",
    "1499": "wipe (stress test; the 'wipe' clip of round 2)",
}
_V = None


def video():
    global _V
    if _V is None:
        from exp45_c12 import target
        _V = target("0:6572").astype(np.uint8)
    return _V


def clip_data(clip, L=16):
    a = int(clip); X = video(); P = pd25.P_BURNIN
    return dict(train=X[a - P:a + L + 1][None], periodic=False, windows=dict(train=(P, L + 1)), a=a, L=L, P=P)


def roll_net(net, seq_prefix, n, C):
    """Hard rollout of a netlist from a visible frame with a P-frame prefix (len(seq_prefix) = P + 1)."""
    from ca import dlca
    P = len(seq_prefix) - 1; s = np.zeros(seq_prefix.shape[1:] + (C,), np.uint8); s[..., 0] = seq_prefix[0]
    for i in range(P): s = dlca.net_eval(net, dlca.np_patches(s, False)); s[..., 0] = seq_prefix[i + 1]
    out = [];
    for _ in range(n): s = dlca.net_eval(net, dlca.np_patches(s, False)); out.append(s[..., 0].copy())
    return np.stack(out)


def baselines(a, L):
    """Fitted on the train window only; return functions start_frame -> rollout of n frames."""
    from rr_gen import fit_shift, shift_roll
    X = video(); F = X[a:a + L + 1]; d = fit_shift(F)
    lut = pd25.lut_baseline(F[None], False)
    from rr_scale import method, roll
    p = method("two_stage_staged", F, 2, np.random.default_rng(0))
    return dict(shift=lambda v, n: shift_roll(v, d, n), lut=lambda v, n: lut(v, n),
                readable=lambda v, n: roll(p, v, np.zeros((0,) + v.shape, np.uint8), n, 2)), dict(shift=list(d), readable_program=p)


def window_metrics(B, F):
    e = (B != F[1:]).mean((1, 2)); f1, ms = pd25.motion(B, F)
    return dict(err=float(e.mean()), hold=float((F[1:] != F[0]).mean()), f1=f1, mass=ms, diverge=next((t + 1 for t in range(len(e)) if e[t] > 0.1), None),
                black=float((B == 0).mean()), white=float((B == 1).mean()))


def evaluate_clip(net, C, a, L, P, base):
    X = video(); r = {}
    for name, s in (("train", a), ("val", a + L), ("final", a + 2 * L)):
        F = X[s:s + L + 1]; B = roll_net(net, X[s - P:s + 1], L, C); r[name] = dict(window=[s, s + L], **window_metrics(B, F))
        for b, f in base.items(): r[name][f"{b}_err"] = float((f(X[s], L) != F[1:]).mean())
    B = roll_net(net, X[a - P:a + 1], 3 * L, C); F = X[a:a + 3 * L + 1]                 # continuous: train -> val -> final
    r["continuous"] = {w: window_metrics(np.concatenate([[F[i * L]], B[i * L:(i + 1) * L]])[1:], F[i * L:(i + 1) * L + 1]) for i, w in enumerate(("train", "val", "final"))}
    return r


def main(clip, setting, seed, L=16, budget=48_000):
    from ca import dlca, logic_simplify as ls
    import jax
    data = clip_data(clip, L); a, P = data["a"], data["P"]; C = 1 + pd25.K_HIDDEN
    if not setting.startswith("burnin"): raise SystemExit("D3 uses the burn-in prefix (D2.5-A); zero init is not run here")
    d = os.path.join(OUT, f"{clip}_L{L}_{setting}_s{seed}" + ("" if budget == 48_000 else f"_b{budget // 1000}k")); os.makedirs(d, exist_ok=True)
    params, wires, log, secs = pd25.train(data, setting, seed, budget)
    net = dlca.to_netlist(dlca.harden(params, wires), C); simp = ls.simplify(ls.abc(ls.simplify(net)))
    eq, _ = ls.equivalent(net, simp); base, binfo = baselines(a, L)
    r = dict(clip=clip, description=CLIPS[clip], setting=setting, seed=seed, L=L, budget=budget, k_hidden=pd25.K_HIDDEN, burnin_frames=P,
             train_seconds=round(secs, 1), backend=jax.default_backend(), final_loss=log[-1], gates_raw=len(net["gates"]), simplified=ls.stats(simp),
             simplified_equivalent=eq, baselines_info=binfo, **evaluate_clip(simp, C, a, L, P, base))
    X = video(); init = np.zeros(X.shape[1:] + (C,), np.uint8); init[..., 0] = X[a - P]
    dlca.export(os.path.join(d, "circuit.logic.json"), simp, C, False, init_state=init, steps=P + 3 * L, prefix=[X[a - P + i + 1] for i in range(P)],
                meta=dict(clip=clip, setting=setting, seed=seed, source_frames=[a - P, a + 3 * L],
                          note=f"frame {a - P} + {P} observed prefix frames, then {3 * L} free-running steps = frames {a}..{a + 3 * L}"))
    np.savez_compressed(os.path.join(d, "params.npz"), **{f"{n_}_{i}": np.asarray(g) for n_ in ("perceive", "update") for i, g in enumerate(params[n_])})
    json.dump(r, open(os.path.join(d, "result.json"), "w"), indent=1, default=float); print(json.dumps({k: r[k] for k in ("clip", "setting", "seed", "simplified", "train", "val")}, default=float))
    return r


if __name__ == "__main__":
    clip, setting, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]); L = int(sys.argv[4]) if len(sys.argv) > 4 else 16; budget = int(sys.argv[5]) if len(sys.argv) > 5 else 48_000
    if os.path.exists(os.path.join(OUT, f"{clip}_L{L}_{setting}_s{seed}" + ("" if budget == 48_000 else f"_b{budget // 1000}k"), "result.json")): print("exists"); sys.exit(0)
    main(clip, setting, seed, L, budget)


# ---------------- optional extension: reduce a D3 circuit while keeping its fitted motion (CPU) ----------------
def prune_clip(run_dir, tol=0.003):
    """Greedy unit tying scored ONLY on the training window (visible error vs the original frames may rise by <= tol);
    val/final are re-scored afterwards, never used. Label: approximate, fitted-trajectory pruning."""
    from ca import dlca, logic_simplify as ls
    r0 = json.load(open(os.path.join(run_dir, "result.json"))); L0 = dlca.load(os.path.join(run_dir, "circuit.logic.json"))
    a, L, P, C = int(r0["clip"]), r0["L"], r0["burnin_frames"], L0["channels"]; X = video(); F = X[a:a + L + 1]
    net = ls.simplify(ls.abc(ls.simplify(L0["net"]))); err = lambda n: float((roll_net(n, X[a - P:a + 1], L, C) != F[1:]).mean()); e0 = err(net)
    # visited patches along the training rollout, for the candidate order (most-constant gates first)
    st = np.zeros(X.shape[1:] + (C,), np.uint8); st[..., 0] = X[a - P]; Ps = []
    for i in range(P + L):
        Ps.append(dlca.np_patches(st, False)); st = dlca.net_eval(net, Ps[-1])
        if i < P: st[..., 0] = X[a - P + i + 1]
    Pv = np.concatenate([p.reshape(-1, 9, C) for p in Ps]); log = []; changed = True
    while changed:
        changed = False; vals = ls.node_values(net, Pv); n_in = net["n_in"]
        for j in sorted(range(len(net["gates"])), key=lambda j: -abs(vals[n_in + j].mean() - 0.5)):
            if net["gates"][j][0] in (0, 15): continue
            cand = ls.tie(net, n_in + j, int(vals[n_in + j].mean() >= 0.5))
            if len(cand["gates"]) < len(net["gates"]) and err(cand) <= e0 + tol: net = cand; changed = True; log.append(len(net["gates"])); break
    net = ls.simplify(ls.abc(net)); base, _ = baselines(a, L)
    out = dict(source=os.path.relpath(run_dir, ROOT), tol=tol, gates_before=len(ls.simplify(ls.abc(ls.simplify(L0["net"])))["gates"]), stats=ls.stats(net), trace=log, **evaluate_clip(net, C, a, L, P, base))
    name = os.path.basename(run_dir); json.dump(out, open(os.path.join(OUT, f"{name}__pruned.json"), "w"), indent=1, default=float)
    init = np.zeros(X.shape[1:] + (C,), np.uint8); init[..., 0] = X[a - P]
    dlca.export(os.path.join(OUT, f"{name}__pruned.logic.json"), net, C, False, init_state=init, steps=P + 3 * L, prefix=[X[a - P + i + 1] for i in range(P)],
                meta=dict(clip=str(a), source=out["source"], method=f"unit tying scored on the training window only (tol {tol})", gates_raw=r0["gates_raw"],
                          source_frames=[a - P, a + 3 * L], note=f"frame {a - P} + {P} observed prefix frames, then {3 * L} free-running steps = frames {a}..{a + 3 * L}"))
    return out
