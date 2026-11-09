"""Exp21 (Phase 07, candidate C2): exhaustive search over ALL 2^18 outer-totalistic (Life-like) rules.

Rule language: B (born: dead cell with n live Moore neighbours, n=0..8) and S (survive: live cell with
n live neighbours) -> 18 bits, memoryless, isotropic. Boundary 0, S_steps generations per frame.
Every rule is rolled out autonomously from frame 0; objective = mean pixel error over frames 1..T-1
(approximate); exact = 0. Global optimum within the language (no sampling). Ties: lowest rule id.
The best rule is exported as a memoryless LUT container file and replayed by Python and Node.
usage: exp21_lifelike.py TARGET STEPS [--batch 2048]
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c7_common import C7, baselines_bytes, evaluate, export_verify, lifelike_lut, media, target
from ca.rule import family_taps, rollout

tname, S = sys.argv[1], int(sys.argv[2]); B = 2048
F = target(tname); T, H, W = F.shape
OUT = os.path.join(C7, "c2_lifelike"); os.makedirs(OUT, exist_ok=True)
stem = os.path.join(OUT, f"{tname.replace(':', '_')}__S{S}")


def tables(ids):  # (R,18) bit table: [B0..B8, S0..S8]
    return ((ids[:, None] >> np.arange(18)) & 1).astype(np.uint8)


def run(ids, n_frames):
    """Batched autonomous rollout; returns per-rule mean error over frames 1..T-1 (and states if 1 rule)."""
    tab = tables(ids); R = len(ids)
    s = np.broadcast_to(F[0], (R, H, W)).astype(np.uint8).copy()
    err = np.zeros(R); outs = [s.copy()] if R == 1 else None
    for t in range(1, n_frames):
        for _ in range(S):
            p = np.pad(s, ((0, 0), (1, 1), (1, 1)))
            cnt = sum(p[:, 1 + dy:1 + dy + H, 1 + dx:1 + dx + W] for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0))
            idx = (s.astype(np.int64) * 9 + cnt).reshape(R, -1)
            s = np.take_along_axis(tab, idx, 1).reshape(R, H, W)
        err += (s != F[t]).mean((1, 2))
        if outs is not None: outs.append(s.copy())
    return err / (T - 1), (np.stack(outs)[:, 0] if outs is not None else None)


def name(r):
    b = [i for i in range(9) if (r >> i) & 1]; s = [i for i in range(9) if (r >> (9 + i)) & 1]
    return "B" + "".join(map(str, b)) + "/S" + "".join(map(str, s))


t0 = time.process_time(); errs = np.empty(2 ** 18)
for a in range(0, 2 ** 18, B):
    ids = np.arange(a, min(a + B, 2 ** 18)); errs[ids], _ = run(ids, T)
search_cpu = time.process_time() - t0
order = np.lexsort((np.arange(2 ** 18), errs)); best = int(order[0])
_, P = run(np.array([best]), T)
ev = evaluate(P, F)
hold = ev["mean_hold"]
# export: memoryless LUT over taps family_taps(0) (bit k = Moore cell k, centre = bit 4)
taps = family_taps(0); lut = lifelike_lut(best)
v, _, _ = rollout(F[0], taps, lut, T, S); assert (v == P).all(), "LUT export disagrees with batched rollout"
ex = export_verify(P, F, dict(luts=lut, taps=taps, family="memoryless", k=0, S=S), "ca_lut_exc", stem + ".best.cab")
res = dict(method="C2 Life-like exhaustive", target=tname, shape=[T, H, W], S=S, rules_searched=2 ** 18, search_cpu_s=round(search_cpu, 1),
           best_rule=name(best), best_rule_id=best, rule_bits=18, n_conditions=bin(best).count("1"),
           n_rules_better_than_hold=int((errs < hold).sum()), n_rules_exact=int((errs == 0).sum()),
           top10=[dict(rule=name(int(r)), mean_err=round(float(errs[r]), 5)) for r in order[:10]],
           error_quantiles={q: round(float(np.quantile(errs, q)), 5) for q in (0.0, 0.01, 0.1, 0.5)},
           eval=ev, export=ex, baselines_bytes=baselines_bytes(F), media=media(P, F, stem + ".best"))
np.save(stem + ".errs.npy", errs.astype(np.float32))
json.dump(res, open(stem + ".result.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k not in ("eval",)} | {"mean_err": ev["mean_err"], "hold": hold, "change_iou": ev["change_iou"], "horizon": ev["horizon"]}))
