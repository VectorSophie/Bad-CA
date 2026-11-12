"""Path D, stage D2: behaviour-preserving circuit simplification. CPU only (run under experiments/c9_meter.py).
Methods, all starting from the same learned circuit:
  A  exact:   ca.logic_simplify.simplify + ABC (+ simplify); equivalence proved by SAT.
  B  one-step pruning: greedy unit tying (gate -> its majority value on the visited states, most-constant gates first),
     a tie is kept if one-step outputs on the visited patches change on at most eps of them.
  C  multi-step pruning: same candidates and order, a tie is kept if the visible rollouts from the selection starts stay
     within eps of the original circuit's at every step up to 128.
  C-task (checkerboard only): a tie is kept if the target still forms by step 20 and holds to step 128.
After pruning, A is applied again. Scored on unseen starts never used for selection.
usage: pd_d2.py SUBJECT_DIR [SUBJECT_DIR ...]   -> runs/path_d/d2/<name>.json"""
import json, os, sys, time
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
from ca import dlca, logic_simplify as ls

OUT = os.path.join(ROOT, "runs", "path_d", "d2"); HMAX = 128


def subject(d):
    """(net, C, periodic, starts(rng, n), task_ok(traj) or None) for a D0/D1 run directory."""
    L = dlca.load(os.path.join(d, "circuit.logic.json")); m = L["meta"]; C = L["channels"]; periodic = L["boundary"] == "periodic"
    task = m.get("task"); ok = None
    if task == "checker":
        tgt = (((np.arange(16)[None] // 2) + (np.arange(16)[:, None] // 2)) % 2).astype(np.uint8)
        starts = lambda rng, n: rng.integers(0, 2, (n, 16, 16, C)).astype(np.uint8)
        ok = lambda tr: bool((tr[:, 20:, ..., 0] == tgt).all())
    elif task == "life":
        starts = lambda rng, n: (rng.random((n, 32, 32, 1)) < 0.5).astype(np.uint8)
    elif m.get("setting", "").startswith("burnin"):   # D2.5 visible-only circuit: full start states come from its own 2-frame prefix
        import pd_d1
        def starts(rng, n, dens=(0.004, 0.004)):
            out = []
            for i in range(n):
                seq = pd_d1.trajectories("gh", pd_d1.init_states("gh", 1, rng, size=48, density=dens), 64)[0, :, :, :, 0]; t0 = int(rng.integers(4, 60))
                st = np.zeros(seq.shape[1:] + (C,), np.uint8); st[..., 0] = seq[t0 - 2]
                for j in range(2): st = dlca.net_eval(L["net"], dlca.np_patches(st, periodic)); st[..., 0] = seq[t0 - 1 + j]
                out.append(st)
            return np.stack(out)
    else:
        import pd_d1
        starts = lambda rng, n: pd_d1.init_states(task, n, rng)
    return L["net"], C, periodic, starts, ok


def roll(net, X, n, periodic):
    out = [X]
    for _ in range(n): out.append(dlca.net_eval(net, np.stack([dlca.np_patches(x, periodic) for x in out[-1]])))
    return np.stack(out, 1)                                            # (B, n+1, H, W, C)


def patches_of(tr, periodic):
    B, T = tr.shape[:2]; return np.stack([dlca.np_patches(tr[b, t], periodic) for b in range(B) for t in range(T)]).reshape(-1, 9, tr.shape[-1])


def prune(net, mode, eps, ref, P, periodic, ok):
    """Greedy unit tying. ref: original rollouts from the selection starts; P: visited patches (one-step data)."""
    X = ref[:, 0]; y_ref = dlca.net_eval(net, P); log = []; evals = 0
    def accept(cand):
        if mode == "B": return float((dlca.net_eval(cand, P) != y_ref).any(-1).mean()) <= eps
        tr = roll(cand, X, HMAX, periodic)
        if mode == "C": return float((tr[..., 0] != ref[..., 0]).mean((0, 2, 3)).max()) <= eps
        return ok(tr)                                                    # C-task
    changed = True
    while changed:
        changed = False; vals = ls.node_values(net, P); n_in = net["n_in"]
        order = sorted(range(len(net["gates"])), key=lambda j: -abs(vals[n_in + j].mean() - 0.5))
        for j in order:
            if net["gates"][j][0] in (0, 15): continue
            bit = int(vals[n_in + j].mean() >= 0.5); cand = ls.tie(net, n_in + j, bit); evals += 1
            if len(cand["gates"]) >= len(net["gates"]): continue
            if accept(cand): log.append(dict(gate=j, bit=bit, gates_after=len(cand["gates"]))); net = cand; changed = True; break
    return net, log, evals


def score(orig, net, C, periodic, starts, ok, rng):
    X = starts(rng, 64); ro = roll(orig, X, HMAX, periodic); rn = roll(net, X, HMAX, periodic)
    st = ls.stats(net); eq, _ = ls.equivalent(orig, net)
    Pu = rng.integers(0, 2, (200_000, 9, C)).astype(np.uint8)        # uniform random neighbourhoods (whole input domain)
    r = dict(**st, exact_equivalent=eq, one_step_mismatch_uniform=float((dlca.net_eval(orig, Pu) != dlca.net_eval(net, Pu)).any(-1).mean()),
             one_step_mismatch_visited=float((dlca.net_eval(orig, patches_of(ro[:8, :17], periodic)) != dlca.net_eval(net, patches_of(ro[:8, :17], periodic))).any(-1).mean()))
    for h in (16, 64, 128): r[f"divergence_{h}"] = float((rn[:, 1:h + 1, ..., 0] != ro[:, 1:h + 1, ..., 0]).mean())
    r["identical_rollouts_128"] = int((rn[..., 0] == ro[..., 0]).all((1, 2, 3)).sum())
    from rr_run import evaluate
    fo = [evaluate(ro[i, :, :, :, 0], ro[i, :, :, :, 0]) for i in range(4)]; fn = [evaluate(rn[i, :, :, :, 0], ro[i, :, :, :, 0]) for i in range(4)]
    r["motion_f1_vs_original"] = float(np.mean([x["lag4_f1"] for x in fn])); r["change_mass_vs_original"] = float(np.mean([x["lag4_mass"] or 0 for x in fn]))
    if ok is not None: r["task_ok_unseen"] = f"{sum(ok(rn[i:i + 1]) for i in range(64))}/64"
    return r


def run(d):
    name = os.path.basename(os.path.normpath(d)); net0, C, periodic, starts, ok = subject(d); rng = np.random.default_rng(7)
    res = dict(subject=os.path.relpath(d, ROOT), C=C, raw=ls.stats(net0)); t0 = time.process_time()
    exact = ls.simplify(ls.abc(ls.simplify(net0))); res["A_exact"] = score(net0, exact, C, periodic, starts, ok, np.random.default_rng(99))
    sel = starts(rng, 16); ref = roll(exact, sel, HMAX, periodic); P = patches_of(ref, periodic)
    modes = [("B", 0.0), ("B", 0.001), ("B", 0.01), ("C", 0.0), ("C", 0.001), ("C", 0.01)] + ([("C-task", None)] if ok is not None else [])
    for mode, eps in modes:
        t = time.process_time(); pr, log, evals = prune(exact, mode, eps, ref, P, periodic, ok); fin = ls.simplify(ls.abc(pr))
        key = f"{mode}_eps{eps}" if eps is not None else mode
        res[key] = dict(score(net0, fin, C, periodic, starts, ok, np.random.default_rng(99)), ties=len(log), candidate_evals=evals, cpu_s=round(time.process_time() - t, 1))
        os.makedirs(OUT, exist_ok=True); dlca.export(os.path.join(OUT, f"{name}__{key}.logic.json"), fin, C, periodic, init_state=sel[0], steps=HMAX,
                                                     meta=dict(subject=res["subject"], method=key))
    res["cpu_s"] = round(time.process_time() - t0, 1)
    json.dump(res, open(os.path.join(OUT, f"{name}.json"), "w"), indent=1, default=float); return res


if __name__ == "__main__":
    for d in sys.argv[1:]:
        r = run(d); print(r["subject"], json.dumps({k: (v["gates"], v.get("divergence_128"), v.get("task_ok_unseen")) for k, v in r.items() if isinstance(v, dict) and "gates" in v and k != "raw"}), flush=True)
