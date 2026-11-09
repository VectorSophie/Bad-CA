"""Exp04: DiffLogic pilot on the exp03 SAT ablation. GPU: submit through gpuq only.

Two questions, separated:
  capacity     -- can this gate architecture express a known-feasible rule? Distill the SAT
                  rule (visited LUT entries only) supervised; harden; check exactness.
  optimisation -- trained ONLY from the frames (autonomous soft rollout loss, no SAT info),
                  does any seed's hardened rule reproduce the clip?
Infeasible configs (SAT UNSAT) are still trained to measure approximate quality.

usage: exp04_difflogic.py TARGET COND [--seeds N] [--steps N] [--kernels 16] [--depth 16] [--lr F] [--tag S]
       [--sat_ref keep] [--skip_auto 1] [--skip_distill 1]
"""
import json, os, subprocess, sys, time
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax, jax.numpy as jnp
from ca.difflogic import adamw, hard_lut, init_logits, live_gates, make_wiring, net, soft_rollout
from ca.rule import family_taps, rollout
from ca.core import metrics
from ca.export import export_rule
from exp03_sat_ablation import CONDS, target

a = sys.argv[1:]
tname, cond = a[0], a[1]
opt = dict(seeds=16, steps=3000, kernels=16, depth=16, lr=0.05, tag="pilot", eval_every=250, sat_ref="", skip_auto=0, skip_distill=0, budget_s=1500)
for i in range(2, len(a), 2):
    opt[a[i].lstrip("-")] = type(opt[a[i].lstrip("-")])(a[i + 1])
k, S, seen_by = CONDS[cond]; C = k + 1
taps = family_taps(k, seen_by); n = len(taps)
UPDATE = (128,) * opt["depth"] + (128, 64, 32, 16, 8, 4)
F = target(tname); T = len(F)
OUT = os.path.join(ROOT, "runs", "exp04", f"{tname}_{cond}_{opt['tag']}"); os.makedirs(OUT, exist_ok=True)
res = dict(target=tname, cond=cond, hidden_bits=k, steps_per_frame=S, family="F_priv" if seen_by == "self" else "F_shared",
           arch=f"ref: {opt['kernels']} perception kernels [8,4,2,1] + update {UPDATE}+[{C}]", opt=opt, device=str(jax.devices()[0]), boundary=0, hidden_init=0,
           seed_scheme="seed s: wiring permutations np.random.default_rng(1000+s); logits deterministic pass-through init")


def hard_eval(logits, wiring):
    luts = hard_lut(logits, wiring, n)
    vis, _, seen = rollout(F[0], taps, luts, T, S)
    m = metrics(vis, F)
    return luts, vis, seen, m


def train(loss_fn, seeds, steps, log_name, eval_fn):
    """vmapped training over seeds; every seed counted in compute. Returns per-seed best."""
    wirings = [make_wiring(np.random.default_rng(1000 + s), n, C, opt["kernels"], UPDATE) for s in range(seeds)]
    W = [jnp.asarray(np.stack([w[l] for w in wirings])) for l in range(len(wirings[0]))]
    P = [jnp.stack([l] * seeds) for l in init_logits(wirings[0])]
    init, upd = adamw(opt["lr"])
    st = jax.vmap(init)(P)
    vg = jax.jit(jax.vmap(jax.value_and_grad(loss_fn)))
    @jax.jit
    def step(P, st, W):
        l, g = vg(P, W)
        P, st = jax.vmap(upd)(g, st, P)
        return P, st, l
    t0 = time.perf_counter()
    P, st, l = step(P, st, W); jax.block_until_ready(l)
    compile_s = time.perf_counter() - t0
    best = [dict(score=-1.0) for _ in range(seeds)]; log = []
    t0 = time.perf_counter()
    for it in range(1, steps + 1):
        P, st, l = step(P, st, W)
        if it % opt["eval_every"] == 0 or it == steps:
            l = np.asarray(l)
            for s in range(seeds):
                Ps = [p[s] for p in P]; Ws = [np.asarray(w[s]) for w in W]
                e = eval_fn(Ps, Ws)
                log.append(dict(it=it, seed=s, loss=float(l[s]), **{kk: v for kk, v in e.items() if not kk.startswith("_")}))
                if e["score"] > best[s]["score"]:
                    best[s] = dict(e, it=it, Ps=Ps, Ws=Ws, soft_loss=float(l[s]))
            if any(b.get("exact") for b in best) or time.perf_counter() - t0 > opt["budget_s"]:
                break
    train_s = time.perf_counter() - t0
    json.dump(log, open(os.path.join(OUT, f"{log_name}_log.json"), "w"))
    return best, dict(compile_s=round(compile_s, 2), train_s=round(train_s, 2), steps_run=it, seeds=seeds,
                      gpu_seconds=round(train_s + compile_s, 2))


# ---------- capacity: distill the SAT rule if one exists ----------
sat_stem = os.path.join(ROOT, "runs", "exp03", f"{tname}_{cond}" + (f"_{opt['sat_ref']}" if opt["sat_ref"] else ""))
sat = json.load(open(sat_stem + ".result.json")) if os.path.exists(sat_stem + ".result.json") else {}
res["sat_status"] = sat.get("status", "not run"); res["sat_ref"] = os.path.basename(sat_stem)
if sat.get("status") == "SAT" and not opt["skip_distill"]:
    sl = np.load(sat_stem + ".luts.npy"); care = np.array(json.load(open(sat_stem + ".visited.json")))
    X = jnp.asarray(((care[:, None] >> np.arange(n)) & 1).astype(np.float32)); Y = jnp.asarray(sl[:, care].T.astype(np.float32))
    def dloss(P, W): return jnp.sum((net(P, W, X) - Y) ** 2)
    def deval(Ps, Ws):
        luts = hard_lut(Ps, Ws, n); acc = float((luts[:, care] == sl[:, care]).mean())
        out = dict(score=acc, care_acc=acc, exact=acc == 1.0)
        if acc == 1.0:
            vis, _, _ = rollout(F[0], taps, luts, T, S); out["rollout_exact"] = bool((vis == F).all())
        return out
    best, comp = train(dloss, opt["seeds"], opt["steps"], "distill", deval)
    res["capacity_distill"] = dict(compute=comp, best_care_acc=max(b["score"] for b in best),
                                   seeds_exact=sum(bool(b.get("exact")) for b in best), care_entries=int(len(care)))
    print("distill", json.dumps(res["capacity_distill"]), flush=True)

if opt["skip_auto"]:
    json.dump(res, open(os.path.join(OUT, "result.json"), "w"), indent=1); print(json.dumps(res, indent=1)); sys.exit(0)

# ---------- optimisation: frames only, autonomous rollout loss ----------
tgt = jnp.asarray(F[1:].astype(np.float32)); vis0 = jnp.asarray(F[0].astype(np.float32))
def aloss(P, W): return jnp.sum((soft_rollout(P, W, taps, vis0, C, T, S) - tgt) ** 2)
def aeval(Ps, Ws):
    luts, vis, seen, m = hard_eval(Ps, Ws)
    soft = np.asarray(soft_rollout(Ps, [jnp.asarray(w) for w in Ws], taps, vis0, C, T, S))
    return dict(score=1 - m["pixel_err"], exact=bool((vis == F).all()), hard_pixel_err=m["pixel_err"],
                hard_change_iou=m["change_iou"], soft_thresh_pixel_err=float(((soft > 0.5) != F[1:]).mean() * (T - 1) / T),
                visited=len(seen), _luts=luts, _vis=vis)
best, comp = train(aloss, opt["seeds"], opt["steps"], "auto", aeval)
b = max(best, key=lambda b: b["score"]); s_best = best.index(b)
path = os.path.join(OUT, "best.rule.json")
ex = export_rule(path, taps, b["_luts"], F[0], T, S, meta=dict(source=tname, method="DiffLogic", cond=cond, seed=s_best,
                                                                 it=b["it"], arch=res["arch"]), expected=b["_vis"])
node = json.loads(subprocess.run(["node", os.path.join(ROOT, "player", "replay.mjs"), path], capture_output=True, text=True, check=True).stdout)
res["autonomous"] = dict(compute=comp, seeds_exact=sum(bool(x.get("exact")) for x in best), best_seed=s_best, best_it=b["it"],
                         hard_pixel_err=b["hard_pixel_err"], hard_change_iou=b["hard_change_iou"],
                         soft_thresh_pixel_err=b["soft_thresh_pixel_err"], soft_loss=b["soft_loss"],
                         hold_pixel_err=metrics(np.repeat(F[:1], T, 0), F)["pixel_err"],
                         live_gates=live_gates(b["Ps"], b["Ws"], n), visited_entries=b["visited"],
                         node_replay_matches_python=node["sha256"] == ex["expected_sha256"],
                         per_seed_hard_err=[round(x["hard_pixel_err"], 4) for x in best])
json.dump(res, open(os.path.join(OUT, "result.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
