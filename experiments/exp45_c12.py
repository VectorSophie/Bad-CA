"""Exp45 (Campaign 12): train a phased NCA arm on a Bad Apple interval and export ncab-v2 (one file per seed).
TARGET = START:LEN (decoded-frame indices of refs/Bad-Apple-Starter/BadApple!!.mp4, 48 px wide, gray >= 128).
ARM:
  fixed       K=1                         (Fh from --Fh, e.g. 64 or the capacity-matched 256)
  uniform     K equal phases, mode indep
  auto        K phases from ca/segment.best_k (stage-1 MDL), mode indep
  autodelta   same phases, shared base + rank-r per-phase deltas
  greedy      auto phases (or --phases JSON), mode indep, sequential training with carried state (ca/phase_train.greedy)
  chunked     phases (--phases JSON or auto) grouped into ~200-280-frame chunks, joint inside a chunk, state carried between chunks
usage: exp45_c12.py TARGET OUT ARM [--K 4] [--seeds 0,1] [--steps 15000] [--lr 5e-4] [--Fh 64] [--rank 4]
                    [--init-factor F] [--phases phases.json] [--eval-every 500] [--lmin 8] [--grid 1] [--cold]
  --noise SIGMA (chunked): training-only Gaussian state noise per frame (robustness); replay is unchanged
  --cold (greedy): each phase starts from the 'hold' initialisation (zero update) instead of the previous phase's rule
          (the warm start exploded when a rule ran past its trained horizon, pilot greedy4 phase 4)."""
import json, os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca import phase_nca, segment
from ca.phase_train import chunked, collect_greedy, greedy, joint
from ca.video import binarize, gray_frames

SRC = os.path.join(ROOT, "refs", "Bad-Apple-Starter", "BadApple!!.mp4")


def target(spec):
    a, n = map(int, spec.split(":")); return binarize(gray_frames(SRC, 48)[0])[a:a + n]


def opt(name, default, cast=str):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


if __name__ == "__main__":
    spec, out, arm = sys.argv[1], os.path.join(ROOT, "runs", sys.argv[2]), sys.argv[3]; os.makedirs(out, exist_ok=True)
    F = target(spec); T = len(F); K = opt("--K", 4, int); seeds = [int(s) for s in opt("--seeds", "0,1").split(",")]
    steps = opt("--steps", 15000, int); lr = opt("--lr", 5e-4, float); Fh = opt("--Fh", 64, int); rank = opt("--rank", 4, int)
    fac = opt("--init-factor", None, int); ev = opt("--eval-every", 500, int); sched = ("cos", lr, lr / 100)
    if "--phases" in sys.argv: phases = json.load(open(opt("--phases", None)))
    elif arm == "fixed": phases = [dict(start=1, end=T, reason="fixed")]
    elif arm == "uniform": phases = segment.uniform_k(T, K)
    else: phases = segment.best_k(F, K, lmin=opt("--lmin", 8, int), grid=opt("--grid", 1, int))["phases"]
    json.dump(dict(spec=spec, arm=arm, phases=phases, argv=sys.argv[1:]), open(os.path.join(out, "phases.json"), "w"), indent=1)
    log = lambda s: print(s, flush=True); a0 = int(spec.split(":")[0])
    meta = dict(source="refs/Bad-Apple-Starter/BadApple!!.mp4", src_start=a0, src_frames=[a0, a0 + T - 1], arm=arm, width=48, threshold=128)
    if arm == "chunked":
        P, g, phases = chunked(F, phases, out, seed=seeds[0], Fh=Fh, steps=steps, sched=sched, init_factor=fac, eval_every=ev, log=log, noise=opt("--noise", 0.0, float))
        mode = "indep"; Ps = {seeds[0]: (P, g)}
    elif arm == "greedy":
        rec = greedy(F, phases, out, seed=seeds[0], Fh=Fh, steps_per_phase=steps, sched=sched, init_factor=fac, eval_every=ev, log=log, warm="--cold" not in sys.argv)
        P, g = collect_greedy(out, phases, seeds[0]); mode = "indep"; Ps = {seeds[0]: (P, g)}
    else:
        mode = {"fixed": "fixed", "uniform": "indep", "auto": "indep", "autodelta": "delta"}[arm]
        res = joint(F, phases, mode, out, seeds=seeds, Fh=Fh, rank=rank, steps=steps, sched=sched, init_factor=fac, eval_every=ev, log=log)
        Ps = {}
        for s in seeds:
            z = dict(np.load(os.path.join(out, f"best_joint_s{s}.npz"))); g = z.pop("init_g", None); Ps[s] = (z, g)
    for s, (P, g) in Ps.items():
        sizes = phase_nca.export(os.path.join(out, f"export_s{s}.ncab"), P, mode, phases, F[0], T, 4, init=dict(kind="lowres", factor=fac, g=g) if g is not None else None,
                                 meta=dict(meta, seed=s))
        print(json.dumps(dict(seed=s, export=os.path.relpath(os.path.join(out, f"export_s{s}.ncab"), ROOT), sizes=sizes)), flush=True)
