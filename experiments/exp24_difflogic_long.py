"""Exp24 (Phase 07, candidate C5): DiffLogic CA (reference architecture, ca/difflogic.py) on a LONG clip,
approximate objective. GPU via gpuq. Full soft rollout from frame 0 (lax.scan over all T-1 frames x S
generations), loss = squared error on the visible channel at every frame. Every eval_s seconds each
seed's network is HARDENED and rolled out autonomously as a binary CA; the checkpoint with the lowest
hard mean pixel error is kept (selection by hard rollout, never by soft loss). Saved immediately.
Same 4 wiring seeds as exp07 (1000..1003). usage: exp24_difflogic_long.py TARGET STRUCT BUDGET_S TAG
"""
import json, os, pickle, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c7_common import ROOT, baselines_bytes, evaluate, export_verify, media, target
from c5_targets import STRUCTS
import jax, jax.numpy as jnp
from ca.difflogic import adamw, extract_circuit, hard_lut, init_logits, make_wiring, soft_rollout
from ca.rule import family_taps, rollout
from ca import costs

tname, st, budget, tag = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
k, seen, S = STRUCTS[st]; taps = family_taps(k, seen); n = len(taps); C = k + 1; SEEDS = 4
fam = "memoryless" if k == 0 else ("F_priv" if seen == "self" else "F_shared")
F = target(tname); T = len(F)
OUT = os.path.join(ROOT, "runs", "c7", "c5_difflogic", f"{tname.replace(':', '_')}__{st}__{tag}"); os.makedirs(OUT, exist_ok=True)
wir = [make_wiring(np.random.default_rng(1000 + s), n, C) for s in range(SEEDS)]
W = [jnp.asarray(np.stack([w[l] for w in wir])) for l in range(len(wir[0]))]
P = [jnp.stack([l] * SEEDS) for l in init_logits(wir[0])]
tgt = jnp.asarray(F[1:].astype(np.float32)); vis0 = jnp.asarray(F[0].astype(np.float32))


def loss(P, W):
    return jnp.sum((soft_rollout(P, W, taps, vis0, C, T, S, remat=True) - tgt) ** 2)


init, upd = adamw(0.05); ost = jax.vmap(init)(P); vg = jax.vmap(jax.value_and_grad(loss))


@jax.jit
def step(P, ost):
    l, g = vg(P, W); P, ost = jax.vmap(upd)(g, ost, P); return P, ost, l


def hard(Ps, Ws):
    L = hard_lut(Ps, Ws, n); vis, _, _ = rollout(F[0], taps, L, T, S); return L, vis.astype(np.uint8)


t0 = time.perf_counter(); P, ost, l = step(P, ost); jax.block_until_ready(l); compile_s = time.perf_counter() - t0
best = [dict(err=2.0) for _ in range(SEEDS)]; log = []; it = 1; t0 = time.perf_counter(); nxt = 0.0
while True:
    el = time.perf_counter() - t0
    if el >= nxt or el >= budget:
        for s in range(SEEDS):
            Ps = [p[s] for p in P]; Ws = [np.asarray(w[s]) for w in W]; L, vis = hard(Ps, Ws)
            e = float((vis[1:] != F[1:]).mean()); log.append(dict(t=round(el, 1), it=it, seed=1000 + s, soft_loss=float(np.asarray(l)[s]), hard_err=e))
            if e < best[s]["err"]:
                best[s] = dict(err=e, it=it, t=round(el, 1), L=L, vis=vis, Ps=[np.asarray(p) for p in Ps], Ws=Ws)
                pickle.dump(best[s], open(os.path.join(OUT, f"best_seed{1000 + s}.pkl"), "wb"))   # saved immediately
        nxt = el + 30.0
        if el >= budget or any(b["err"] == 0 for b in best): break
    P, ost, l = step(P, ost); it += 1
train_s = time.perf_counter() - t0
per_seed = []
for s, b in enumerate(best):
    ev = evaluate(b["vis"], F); g, o = extract_circuit([jnp.asarray(p) for p in b["Ps"]], b["Ws"], n)
    assert (costs.eval_circuit(g, o, n) == b["L"]).all()
    per_seed.append(dict(seed=1000 + s, it=b["it"], t=b["t"], mean_err=ev["mean_err"], ratio=ev["mean_ratio_to_hold"], change_iou=ev["change_iou"],
                         exact=ev["exact"], horizon=ev["horizon"], live_gates=len(g)))
sel = int(np.argmin([b["err"] for b in best])); b = best[sel]
g, o = extract_circuit([jnp.asarray(p) for p in b["Ps"]], b["Ws"], n); ev = evaluate(b["vis"], F)
ex = export_verify(b["vis"], F, dict(gates=g, outputs=o, n_in=n, family=fam, k=k, S=S), "ca_circuit", os.path.join(OUT, "best.cab"))
res = dict(method="C5 DiffLogic CA, long-horizon approximate, selection by hard rollout", target=tname, struct=st, budget_s=budget, tag=tag,
           device=str(jax.devices()[0]), compile_s=round(compile_s, 1), train_s=round(train_s, 1), steps=it, seeds=[1000 + s for s in range(SEEDS)],
           selected_seed=1000 + sel, live_gates=len(g), rule_bits=len(costs.rule_circuit(g, o, n, fam, k, S)), per_seed=per_seed, eval=ev, export=ex,
           baselines_bytes=baselines_bytes(F), media=media(b["vis"], F, os.path.join(OUT, "best")))
json.dump(log, open(os.path.join(OUT, "log.json"), "w")); json.dump(res, open(os.path.join(OUT, "result.json"), "w"), indent=1)
print(json.dumps({x: y for x, y in res.items() if x not in ("eval",)} | {"mean_err": ev["mean_err"], "hold": ev["mean_hold"], "change_iou": ev["change_iou"]}))
