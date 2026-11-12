"""Path D4-C: where does the soft-to-hard gap come from? For a trained run (params.npz + its seed's wiring):
  - per layer: mean max softmax probability, share of ambiguous gates (max p < 0.9);
  - per layer: share of units whose soft value (rounded) differs from the hard value, on the same visited inputs
    (hard inputs fed to both, so the mismatch is local to that layer's decisions, then propagated);
  - soft vs hard rollout from the same prefix: visible disagreement per step, and soft-state 'fuzziness' (mean |s - round(s)|).
usage: pd_d4_harden.py RUN_DIR [RUN_DIR ...]"""
import json, os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax, jax.numpy as jnp
from ca import dlca
import pd25


def load(run_dir):
    r = json.load(open(os.path.join(run_dir, "result.json"))); C = 1 + r["k_hidden"]
    _, wires = dlca.init(pd25.hparams(C), r["seed"]); z = np.load(os.path.join(run_dir, "params.npz"))
    params = {n: [jnp.asarray(z[f"{n}_{i}"]) for i in range(len(wires[n]))] for n in ("perceive", "update")}
    return r, C, params, wires


def layers(params, wires, patch, hard):
    """Per-layer outputs of the update net (perception outputs first), same maths as dlca.circuit."""
    x = jnp.swapaxes(patch, -1, -2); outs = []
    for k in range(params["perceive"][0].shape[0]):
        y = x
        for g, w in zip(params["perceive"], wires["perceive"]): y = dlca._run_layer(g[k], w, y, hard)
        outs.append(y)
    per = jnp.stack(outs, -1); per = per.reshape(per.shape[:-3] + (-1,)); y = jnp.concatenate([patch[..., 4, :], per], -1); rec = [per]
    for g, w in zip(params["update"], wires["update"]): y = dlca._run_layer(g, w, y, hard); rec.append(y)
    return rec


def diagnose(run_dir):
    import pd_d4
    r, C, params, wires = load(run_dir); V = pd_d4.X(); w, n = r["train_windows"][0]
    gate = []
    for name in ("perceive", "update"):
        for i, g in enumerate(params[name]):
            p = jax.nn.softmax(g, -1).max(-1); gate.append(dict(layer=f"{name}{i}", mean_max_p=float(p.mean()), ambiguous=float((p < 0.9).mean())))
    # hard rollout from the training prefix; collect visited patches
    s = np.zeros(V.shape[1:] + (C,), np.uint8); s[..., 0] = V[w - 2]; net = dlca.to_netlist(dlca.harden(params, wires), C); Ps = []
    for i in range(2 + n):
        Ps.append(dlca.np_patches(s, False)); s = dlca.net_eval(net, Ps[-1])
        if i < 2: s[..., 0] = V[w - 1 + i]
    Pv = jnp.asarray(np.concatenate([p.reshape(-1, 9, C) for p in Ps]), jnp.float32)
    hard = layers(params, wires, Pv, True); soft = layers(params, wires, Pv, False)
    mism = [dict(layer=i, mismatch=float((jnp.round(sl) != hl).mean()), fuzz=float(jnp.abs(sl - jnp.round(sl)).mean())) for i, (sl, hl) in enumerate(zip(soft, hard))]
    # soft vs hard rollouts (soft states carried)
    def roll(hd):
        s0 = jnp.zeros(V.shape[1:] + (C,)).at[..., 0].set(V[w - 2])
        for i in range(2): s0 = dlca.step(params, wires, s0, hd, False).at[..., 0].set(V[w - 1 + i])
        return np.asarray(dlca.rollout(params, wires, s0, 16, hd, False, keep=True))
    Rs, Rh = roll(False), roll(True)
    grow = [dict(step=t + 1, visible_disagree=float((np.round(Rs[t, ..., 0]) != Rh[t, ..., 0]).mean()), soft_fuzz=float(np.abs(Rs[t] - np.round(Rs[t])).mean()),
                 soft_err=float(np.abs(Rs[t, ..., 0] - V[w + t + 1]).mean()) if t < n else None, hard_err=float((Rh[t, ..., 0] != V[w + t + 1]).mean()) if t < n else None) for t in range(16)]
    return dict(run=os.path.relpath(run_dir, ROOT), gates=gate, layer_mismatch=mism, rollout=grow)


if __name__ == "__main__":
    for d in sys.argv[1:]:
        out = diagnose(d); json.dump(out, open(os.path.join(d, "hardening.json"), "w"), indent=1)
        g = out["gates"]; print(out["run"], "ambiguous gates (perceive/update mean):", round(np.mean([x["ambiguous"] for x in g[:3]]), 3), round(np.mean([x["ambiguous"] for x in g[3:]]), 3),
              "| first layer with mismatch > 1%:", next((x["layer"] for x in out["layer_mismatch"] if x["mismatch"] > 0.01), None),
              "| final-layer mismatch", round(out["layer_mismatch"][-1]["mismatch"], 4),
              "| rollout visible soft/hard disagree at 1, 4, 8, 16:", [round(out["rollout"][t]["visible_disagree"], 3) for t in (0, 3, 7, 15)],
              "| soft fuzz at 1, 8, 16:", [round(out["rollout"][t]["soft_fuzz"], 3) for t in (0, 7, 15)])
