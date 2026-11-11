"""Short-clip fixed-rule check: per config the best export (lowest err); loophole baselines run autonomously on the same clip
(hold, majority, flicker = invert every frame, best classic shift); and the SAME rule run, unchanged, from the original frame
of two later 16-frame clips it never saw (held-out continuation, not used in search). Writes runs/readable_rules/diag/short.json."""
import glob, json, os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca import rules, rr_search
from exp45_c12 import target
from rr_run import evaluate

def run(prog, f0, k, T, S): return rules.run(dict(k=k, S=S, T=T, programs=[prog], schedule=[dict(start=1, end=T, program=0)], first=f0, h0=np.zeros((k,) + f0.shape, np.uint8) if k else None))[0]
out = {}
for f in sorted(glob.glob(os.path.join(ROOT, "runs/readable_rules/pilot/short_*/result.json"))):
    o = json.load(open(f)); cfg = os.path.basename(os.path.dirname(f)); task = o["task"]; a, n = map(int, task.split(":")); F = target(task).astype(np.uint8)
    b = min(o["results"], key=lambda r: r["eval"]["err"]); L = rules.load(os.path.join(ROOT, b["file"])); p = rules.resolve(L["programs"])[0]; k, S = L["k"], L["S"]
    base = {}
    for name in ("hold", "majority") + tuple(f"shift_from_{d}" for d in "udlr"): base[name] = float((run(rr_search.LIBRARY[name], F[0], 0, n, S) != F)[1:].mean())
    fl = F.copy(); fl[1:] = np.array([F[0] if i % 2 == 0 else 1 - F[0] for i in range(1, n)]); base["flicker"] = float((fl != F)[1:].mean())
    held = []
    for a2 in (a + 16, a + 32):
        G = target(f"{a2}:{n}").astype(np.uint8); B = run(p, G[0], k, n, S); e = evaluate(B, G)
        held.append(dict(clip=f"{a2}:{n}", err=e["err"], hold=e["hold_err"], lag4_f1=e["lag4_f1"], lag4_mass=e["lag4_mass"]))
    out[cfg] = dict(task=task, seeds_err=[r["eval"]["err"] for r in o["results"]], best=dict(err=b["eval"]["err"], lag4_f1=b["eval"]["lag4_f1"], lag4_mass=b["eval"]["lag4_mass"], spurious=b["eval"]["spurious_blips"],
                    rules=b["complexity"]["rules"], conditions=b["complexity"]["conditions"], bytes=b["complexity"]["file_bytes"], file=b["file"]), baselines=base, heldout=held, text=rules.to_text(p, k))
    print(f"{cfg:26s} err {b['eval']['err']:.4f} F1 {b['eval']['lag4_f1']:.2f} mass {b['eval']['lag4_mass']:.2f} conds {b['complexity']['conditions']:2d} | best baseline {min(base, key=base.get)} {min(base.values()):.4f} | held-out "
          + " ".join(f"{h['err']:.4f}(hold {h['hold']:.4f}, F1 {h['lag4_f1']:.2f})" for h in held))
json.dump(out, open(os.path.join(ROOT, "runs/readable_rules/diag/short.json"), "w"), indent=1, default=float)
