"""Exp14 (Campaign 05): feasibility certificates per target x S before any search (CPU, seconds).
light cone (any radius-1 rule, constant init/boundary, any k/visibility), state repeat (memoryless, any S),
history conflict (F_priv, S=1, any k). Output runs/c5/certs.json."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca.sat import history_conflicts, light_cone_conflicts, state_repeat_conflict
from c5_targets import target

out = {}
for t in ["pcA", "pcB", "translate24", "disc12", "bar12", "ba32_3000_5", "ba32_4500_5", "ba32_4500_10"]:
    F = target(t)
    out[t] = dict(shape=list(F.shape), changed_px=int((F[1:] != F[:-1]).sum()),
                  state_repeat_memoryless=state_repeat_conflict(F),
                  history_conflict_Fpriv_S1=history_conflicts(F),
                  light_cone={S: light_cone_conflicts(F, S) for S in (1, 2, 4, 8)})
    print(t, json.dumps(out[t]))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", "c5", "certs.json"), "w"), indent=1)
