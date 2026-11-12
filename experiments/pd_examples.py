"""Path D: build the Logic CA Inspector examples (ui/static/logic_examples/) from committed Path D results.
A semantic statement is attached only when it was checked against the circuit on every possible neighbourhood.
usage: pd_examples.py"""
import json, os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
os.environ.setdefault("JAX_PLATFORMS", "cpu")
from ca import dlca, logic_simplify as ls
import pd_d1

OUT = os.path.join(ROOT, "ui", "static", "logic_examples"); R = os.path.join(ROOT, "runs", "path_d")


def gh_statement(net):
    """Exact GH statement iff the circuit equals the Greenberg-Hastings generator on all 2^18 neighbourhoods."""
    C, periodic, gen, _ = pd_d1.TASKS["gh"]; idx = np.arange(2 ** 18); bits = ((idx[:, None] >> np.arange(18)) & 1).astype(np.uint8).reshape(-1, 9, 2)
    T = np.stack([gen(np.tile(x, (3, 3, 1)))[4, 4] for x in bits.reshape(-1, 3, 3, 2)])
    if (dlca.net_eval(net, bits) != T).any(): return None
    return dict(text="channel 0 (excited) turns on if the cell is resting (neither excited nor refractory) and at least one of its 8 neighbours is excited; channel 1 (refractory) = the cell was excited",
                verified="equal to the Greenberg-Hastings generator on all 262,144 neighbourhoods")


def main():
    os.makedirs(OUT, exist_ok=True); idx = []; rng = np.random.default_rng(2026)
    def put(fname, net, C, periodic, init, steps, label, note, semantics=None):
        dlca.export(os.path.join(OUT, fname), net, C, periodic, init_state=init, steps=steps, meta=dict(note=note, **({"semantics": semantics} if semantics else {})))
        idx.append(dict(file=fname, label=label))
    life = dlca.load(f"{R}/d0/life_s1/circuit.logic.json")["net"]; L = ls.simplify(ls.abc(ls.simplify(life))); assert ls.equivalent(life, L)[0]
    d = ls.describe_totalistic(L)
    put("life_B3S23.logic.json", L, 1, True, (rng.random((64, 64, 1)) < 0.4).astype(np.uint8), 300, f"Game of Life, learned then simplified ({len(L['gates'])} gates)",
        "Learned by the faithful DiffLogic CA reproduction (D0, seed 1: 2,543 gates), then exact simplification + ABC; SAT-proved equivalent to the learned circuit.",
        dict(text=d["text"], verified=f"{d['rule']}, checked on all 512 neighbourhoods"))
    ch = dlca.load(f"{R}/d0/checker_s2_e2000/circuit.logic.json")["net"]; E = ls.simplify(ls.abc(ls.simplify(ch))); assert ls.equivalent(ch, E)[0]
    x0 = rng.integers(0, 2, (16, 16, 8)).astype(np.uint8)
    put("checkerboard_exact.logic.json", E, 8, False, x0, 40, f"Checkerboard from noise, exact simplification ({len(E['gates'])} gates)",
        "D0 reproduction (seed 2, 2,000 epochs: 4,608-gate netlist), exact simplification + ABC, SAT-proved equivalent. 8 channels; channel 0 is the pattern.")
    six = dlca.load(f"{R}/d2/checker_s2_e2000__C-task.logic.json")["net"]
    put("checkerboard_pruned.logic.json", six, 8, False, x0, 40, f"Checkerboard, behaviour-aware pruning ({len(six['gates'])} gates, approximate)",
        "Unit tying kept only if the pattern still forms by step 20 and holds to 128 from 16 selection starts; then 64/64 unseen 16x16 starts. NOT equivalent to the learned circuit: transients differ.")
    gh = dlca.load(f"{R}/d2/gh_curriculum_s0__C_eps0.0.logic.json")["net"]; Xg = pd_d1.init_states("gh", 1, rng, size=48)[0]
    put("greenberg_hastings.logic.json", gh, 2, True, Xg, 200, f"Excitable waves, recovered by pruning ({len(gh['gates'])} gates)",
        "Learned with the horizon curriculum (D1 gh, seed 0), then multi-step pruning with zero tolerance: the result equals the generator everywhere.", gh_statement(gh))
    bad = dlca.load(f"{R}/d2/gh_onestep_s1__C_eps0.0.logic.json")["net"]; Xb = pd_d1.init_states("gh", 1, rng, size=48, density=(0.08, 0.2))[0]
    assert gh_statement(bad) is None
    put("waves_family_limited.logic.json", bad, 2, True, Xb, 200, f"Excitable waves, pruned one-step circuit ({len(bad['gates'])} gates): fails outside its family",
        "Identical to the generator on all tested starts without refractory cells (the training family), but this start has 20% refractory cells: the waves break (error 0.41 vs the generator). Pruning only certifies the family it was scored on.")
    tr = dlca.load(f"{R}/d2/translate_onestep_s1__C_eps0.0.logic.json")["net"]
    put("translate_wire.logic.json", tr, 1, True, (rng.random((32, 32, 1)) < 0.3).astype(np.uint8), 64, "Translation: 0 gates (the output is a wire to the west neighbour)",
        "Learned (D1 translate, one-step, seed 1: 1,375 gates); exact simplification leaves a wire. Equal to the generator on all 512 neighbourhoods.")
    # ---- D2.5 / D3 (visible-only) ----
    def target(fname, start, frames):
        json.dump(dict(start=int(start), note="original frames for side-by-side comparison only; the circuit never reads this file",
                       frames=[np.packbits(np.asarray(f, np.uint8).ravel()).tobytes().hex() for f in frames]), open(os.path.join(OUT, fname), "w"))
    vo = dlca.load(f"{R}/d2/gh_burnin_s0__C_eps0.0.logic.json")["net"]
    seq = pd_d1.trajectories("gh", pd_d1.init_states("gh", 1, np.random.default_rng(31), size=48, density=(0.004, 0.004)), 140)[0, ..., 0]
    init = np.zeros((48, 48, 3), np.uint8); init[..., 0] = seq[6]
    dlca.export(os.path.join(OUT, "waves_visible_only.logic.json"), vo, 3, True, init_state=init, steps=130, prefix=[seq[7], seq[8]],
                meta=dict(gates_raw=1823, note="Learned from the EXCITED channel only (D2.5, 2-frame observation prefix, seed 0), exact simplification (14 gates) then multi-step pruning: 10 gates. Hidden channels h1, h2 were invented by training. Exact on all 30 untouched test windows for 128 steps; not equal to Greenberg-Hastings on every input (it misses 'only NW and S excited')."))
    target("waves_visible_only.target.json", 6, seq[6:137]); idx.append(dict(file="waves_visible_only.logic.json", target="waves_visible_only.target.json", label="Waves learned from the visible channel only (10 gates, 2 latent channels)"))
    from pd_d3 import video
    X = video()
    for clip, run, label in (("3998", "3998_L8_burnin_s1", "Bad Apple frames 3998-4022: 64-gate circuit (fitted on 3998-4006)"),
                             ("4500", "4500_L8_burnin_s0", "Bad Apple frames 4500-4524: 38 gates, fits but does not transfer")):
        f = f"{R}/d3/{run}__pruned.logic.json"; d = dlca.load(f); a = int(clip)
        dlca.export(os.path.join(OUT, f"badapple_{clip}.logic.json"), d["net"], 3, False, init_state=d["init_state"], steps=d["steps"], prefix=d["prefix_frames"], meta=d["meta"])
        target(f"badapple_{clip}.target.json", a - 2, X[a - 2:a - 2 + d["steps"] + 1]); idx.append(dict(file=f"badapple_{clip}.logic.json", target=f"badapple_{clip}.target.json", label=label))
    json.dump(idx, open(os.path.join(OUT, "index.json"), "w"), indent=1); print(json.dumps(idx, indent=1))


if __name__ == "__main__":
    main()
