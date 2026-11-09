"""Exp07: hard-rule optimisation on ba32_3000_5 E (F_priv k=1, S=2, boundary 0, hidden init 0).
GPU via gpuq. Same 4 wiring seeds (1000..1003), same wall budget per method. Checkpoints are
selected by HARD autonomous rollout error; soft metrics recorded separately.

methods (video only):  R0 soft gates + soft states (relaxation baseline)
                       HG straight-through gates          HS straight-through states (all channels)
                       HH straight-through hidden only    HB HG + HS
campaign 04 (aux):     BP R0 + binarization penalty lam(it) * sum s(1-s) over every generation's state,
                          lam ramps linearly 0 -> lam_max over ramp_steps optimizer steps (step-based, so
                          deterministic). Selection is still by hard rollout; penalty/soft loss are logged only.
teacher (diagnostic):  T1 R0 + loss on hidden channel vs SAT solution's hidden trajectory
                       T2 supervised distillation of the SAT LUT (visited entries)
usage: exp07_hard_rule_opt.py METHOD [--budget_s 600] [--seeds 4] [--tag main] [--target ba32_3000_5] [--lam_max 1] [--ramp_steps 8000]
"""
import json, os, subprocess, sys, time
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax, jax.numpy as jnp
from ca.difflogic import adamw, extract_circuit, hard_lut, init_logits, make_wiring, net, soft_rollout, ca_inputs
from ca.rule import family_taps, rollout
from ca.core import metrics
from ca.export import export_rule
from ca import costs
from exp03_sat_ablation import target

METHODS = {"R0": (False, "soft", False), "HG": (True, "soft", False), "HS": (False, "ste", False),
           "HH": (False, "ste_hidden", False), "HB": (True, "ste", False), "T1": (False, "soft", True), "T2": None, "BP": (False, "soft", False)}
a = sys.argv[1:]; method = a[0]
opt = dict(budget_s=600.0, seeds=4, tag="main", lr=0.05, eval_s=20.0, teacher_w=1.0, target="ba32_3000_5", lam_max=1.0, ramp_steps=8000,
           outdir="exp07")
for i in range(1, len(a), 2):
    opt[a[i].lstrip("-")] = type(opt[a[i].lstrip("-")])(a[i + 1])
TNAME, K, S, C = opt["target"], 1, 2, 2
taps = family_taps(K); n = len(taps)
F = target(TNAME); T = len(F)
SAT_STEM = os.path.join(ROOT, "runs", "exp03", f"{TNAME}_E_keep")
OUT = os.path.join(ROOT, "runs", opt["outdir"], f"{method}_{opt['tag']}"); os.makedirs(OUT, exist_ok=True)
SNAP = os.path.basename(os.path.realpath(ROOT))
res = dict(method=method, track="teacher (diagnostic, not discovery)" if method.startswith("T") else "video only",
           target=TNAME, family="F_priv", k=K, S=S, boundary=0, hidden_init=0, opt=opt, code_snapshot=SNAP,
           device=str(jax.devices()[0]), seeds=[1000 + s for s in range(opt["seeds"])])

wir = [make_wiring(np.random.default_rng(1000 + s), n, C) for s in range(opt["seeds"])]
W = [jnp.asarray(np.stack([w[l] for w in wir])) for l in range(len(wir[0]))]
P = [jnp.stack([l] * opt["seeds"]) for l in init_logits(wir[0])]
tgt = jnp.asarray(F[1:].astype(np.float32)); vis0 = jnp.asarray(F[0].astype(np.float32))

if method.startswith("T"):  # teacher diagnostics only: everything else never reads a SAT solution
    sat_luts = np.load(SAT_STEM + ".luts.npy")
    _, sat_states, sat_seen = rollout(F[0], taps, sat_luts, T, S)
    sat_hidden = jnp.asarray(sat_states[1:, 1].astype(np.float32))  # hidden channel at frame times 1..T-1

if method == "BP":
    def loss(P, W, lam):
        g = soft_rollout(P, W, taps, vis0, C, T, S, gens=True)  # (T-1, S, C, H, W)
        return jnp.sum((g[:, -1, 0] - tgt) ** 2) + lam * jnp.sum(g * (1 - g))
    def penalty(P, W):
        g = soft_rollout(P, W, taps, vis0, C, T, S, gens=True); return jnp.sum(g * (1 - g)), jnp.sum((g[:, -1, 0] - tgt) ** 2)
elif method == "T2":
    care = np.array(sat_seen)
    X = jnp.asarray(((care[:, None] >> np.arange(n)) & 1).astype(np.float32)); Y = jnp.asarray(sat_luts[:, care].T.astype(np.float32))
    def loss(P, W, lam): return jnp.sum((net(P, W, X) - Y) ** 2)
else:
    gate_ste, state, teach = METHODS[method]
    def loss(P, W, lam):
        st = soft_rollout(P, W, taps, vis0, C, T, S, gate_ste=gate_ste, state=state, full=True)
        l = jnp.sum((st[:, 0] - tgt) ** 2)
        return l + (opt["teacher_w"] * jnp.sum((st[:, 1] - sat_hidden) ** 2) if teach else 0.0)

init, upd = adamw(opt["lr"]); ost = jax.vmap(init)(P)
vg = jax.vmap(jax.value_and_grad(loss), in_axes=(0, 0, None))
@jax.jit
def step(P, ost, lam):
    l, g = vg(P, W, lam); P, ost = jax.vmap(upd)(g, ost, P); return P, ost, l
lam_at = lambda it: opt["lam_max"] * min(1.0, it / opt["ramp_steps"]) if method == "BP" else 0.0


def hard_eval(Ps, Ws):
    L = hard_lut(Ps, Ws, n); vis, _, seen = rollout(F[0], taps, L, T, S)
    return L, vis, metrics(vis, F)


def gaps(Ps, Ws):
    """Where does soft != hard come from? Errors are pixel error over frames 1..T-1 (thresholded)."""
    Wj = [jnp.asarray(w) for w in Ws]
    e = lambda **kw: float(((np.asarray(soft_rollout(Ps, Wj, taps, vis0, C, T, S, **kw)) > 0.5) != F[1:]).mean())
    pf = lambda **kw: [float(x) for x in ((np.asarray(soft_rollout(Ps, Wj, taps, vis0, C, T, S, **kw)) > 0.5) != F[1:]).mean((1, 2))]
    out = dict(soft_gates_soft_state=e(), soft_gates_round_state=e(state="round"), soft_gates_round_hidden=e(state="round_hidden"),
               hard_gates_soft_state=e(hard=True), hard_gates_round_state=e(hard=True, state="round"),
               per_frame_soft=pf(), per_frame_hard=pf(hard=True, state="round"))
    # one-step gate-selection gap on the hard trajectory's own (binary) states, no accumulation
    s = np.zeros((C,) + F[0].shape, np.float32); s[0] = F[0]; dis, tot = np.zeros(C), 0
    for _ in range((T - 1) * S):
        x = ca_inputs(jnp.asarray(s), taps)
        hs = np.asarray(net(Ps, Wj, x, hard=True)); ss = np.asarray(net(Ps, Wj, x)) > 0.5
        dis += (hs.round().astype(bool) != ss).reshape(-1, C).sum(0); tot += hs.shape[0] * hs.shape[1]
        s = np.moveaxis(hs, -1, 0).round()
    out["one_step_disagreement_per_channel"] = (dis / tot).tolist()
    return out


t0 = time.perf_counter(); P, ost, l = step(P, ost, 0.0); jax.block_until_ready(l); compile_s = time.perf_counter() - t0
best = [dict(err=2.0) for _ in range(opt["seeds"])]; log = []; it = 1
t0 = time.perf_counter(); next_eval = 0.0
while True:
    el = time.perf_counter() - t0
    if el >= next_eval or el >= opt["budget_s"]:
        l_np = np.asarray(l)
        for s in range(opt["seeds"]):
            Ps = [p[s] for p in P]; Ws = [np.asarray(w[s]) for w in W]
            L, vis, m = hard_eval(Ps, Ws)
            log.append(dict(t=round(el, 1), it=it, seed=1000 + s, loss=float(l_np[s]), hard_err=m["pixel_err"], change_iou=m["change_iou"]))
            if method == "BP":
                pen, vl = penalty(Ps, [w[s] for w in W]); log[-1].update(lam=lam_at(it), penalty=float(pen), video_sq_loss=float(vl))
            if m["pixel_err"] < best[s]["err"]:
                best[s] = dict(err=m["pixel_err"], it=it, t=round(el, 1), Ps=Ps, Ws=Ws, L=L, vis=vis, m=m)
        next_eval = el + opt["eval_s"]
        if el >= opt["budget_s"] or any(b["err"] == 0 for b in best):
            break
    P, ost, l = step(P, ost, lam_at(it)); it += 1
train_s = time.perf_counter() - t0

per_seed = []
for s, b in enumerate(best):
    g, outs = extract_circuit(b["Ps"], b["Ws"], n)
    assert (costs.eval_circuit(g, outs, n) == b["L"]).all()
    cw = costs.rule_circuit(g, outs, n, "F_priv", K, S); lw = costs.rule_lut_exceptions(b["L"], taps, "F_priv", K, S)
    per_seed.append(dict(seed=1000 + s, exact=b["err"] == 0, hard_pixel_err=b["err"], change_iou=b["m"]["change_iou"], it=b["it"], t=b["t"],
                         live_gates=len(g), circuit_bits=len(cw), lut_exception_bits=len(lw),
                         own_output_recording_bits=costs.recording_costs(b["vis"]), gaps=gaps(b["Ps"], b["Ws"])))
sel = int(np.argmin([b["err"] for b in best]))
b = best[sel]
path = os.path.join(OUT, "best.rule.json")
ex = export_rule(path, taps, b["L"], F[0], T, S, meta=dict(method=method, seed=1000 + sel, it=b["it"], code_snapshot=SNAP), expected=b["vis"])
node = json.loads(subprocess.run(["node", os.path.join(ROOT, "player", "replay.mjs"), path], capture_output=True, text=True, check=True).stdout)
np.save(os.path.join(OUT, "best_luts.npy"), b["L"])
import pickle; pickle.dump(dict(Ps=[np.asarray(p) for p in b["Ps"]], Ws=b["Ws"]), open(os.path.join(OUT, "best_params.pkl"), "wb"))
res.update(compute=dict(compile_s=round(compile_s, 2), train_wall_s=round(train_s, 1), steps=it, seeds=opt["seeds"]),
           selected_seed=1000 + sel, selected=per_seed[sel], per_seed=per_seed, any_exact=any(p["exact"] for p in per_seed),
           hold_pixel_err=metrics(np.repeat(F[:1], T, 0), F)["pixel_err"], node_replay_matches_python=node["sha256"] == ex["expected_sha256"],
           target_recording_bits=costs.recording_costs(F))
json.dump(log, open(os.path.join(OUT, "log.json"), "w"))
json.dump(res, open(os.path.join(OUT, "result.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k not in ("per_seed",)}, indent=1))
