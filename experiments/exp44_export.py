"""Exp44 (Campaign 11): export every best_<name>.npz of trained run dirs to runs/c11/exports/<run>_<name>.(ncab|nca.json)
(f32 weights; first frame = the target's first frame). Prints the export paths (one per line).
usage: exp44_export.py RUN_DIR [RUN_DIR ...]   (dirs under runs/, written by exp40_c11.py)"""
import json, os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca import nca, ncab
from exp40_c11 import target

OUT = os.path.join(ROOT, "runs", "c11", "exports"); os.makedirs(OUT, exist_ok=True)
for d in sys.argv[1:]:
    d = os.path.join(ROOT, "runs", d) if not os.path.isabs(d) else d; tr = json.load(open(os.path.join(d, "train.json")))
    F = target(tr["meta"]["target"]); tag = os.path.relpath(d, os.path.join(ROOT, "runs", "c11")).replace("/", "_")
    for name, b in tr["best"].items():
        P = dict(np.load(os.path.join(d, f"best_{name}.npz")))
        meta = dict(target=tr["meta"]["target"], src_frames=tr["meta"]["src_frames"], run=os.path.relpath(d, ROOT), model=name, best_step=b["step"], train_err=b["err"],
                    steps=tr["steps"], sched=tr["sched"], teacher=tr["meta"].get("teacher"))
        base = os.path.join(OUT, f"{tag}_{name}"); ncab.write(base + ".ncab", P, F[0], len(F), 4, meta); nca.export(base + ".nca.json", P, F[0], len(F), 4, meta)
        print(os.path.relpath(base + ".ncab", ROOT))
