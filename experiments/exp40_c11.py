"""Exp40 (Campaign 11): train NCAs with ca/nca_train.py on the Campaign 11 targets (fixed in docs/CAMPAIGN_11.md).
Targets (binary 48-px Bad Apple, decoded-frame indices of refs/Bad-Apple-Starter/BadApple!!.mp4, threshold 128):
  main    = 4500-4555 (56)   confirm = 1800-1859 (60)   L120 = 1760-1879 (120)   L240 = 1760-1999 (240)
Teacher (main/confirm only): frozen C10 export, its continuous visible replayed on the same interval (numpy, from the
export's first frame alone); used only through the agreement-masked LT term.
usage: exp40_c11.py TARGET OUT C Fh STEPS SCHED(const:lr | cos:lr0:lr1) CFG_JSON [--teacher EXPORT] [--eval-every N]"""
import json, os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca import nca
from ca.nca_train import train
from ca.video import binarize, gray_frames

SRC = os.path.join(ROOT, "refs", "Bad-Apple-Starter", "BadApple!!.mp4")
TARGETS = {"main": (4500, 56), "confirm": (1800, 60), "L120": (1760, 120), "L240": (1760, 240)}


def target(name):
    a, n = TARGETS[name]; return binarize(gray_frames(SRC, 48)[0])[a:a + n]


if __name__ == "__main__":
    tname, out, C, Fh, steps, sched, cfg_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6], sys.argv[7]
    sp = sched.split(":"); sched = (sp[0],) + tuple(float(x) for x in sp[1:])
    F = target(tname); tv = None; teacher = None
    if "--teacher" in sys.argv:
        teacher = sys.argv[sys.argv.index("--teacher") + 1]; ex, Pt, f0 = nca.load(teacher)
        assert ex["n_frames"] == len(F) and (f0 == F[0]).all(), "teacher must cover exactly this interval"
        _, tv = nca.rollout_np(Pt, f0[None], len(F), ex["S"])
    ev = int(sys.argv[sys.argv.index("--eval-every") + 1]) if "--eval-every" in sys.argv else 500
    res = train(F, json.load(open(cfg_path)), os.path.join(ROOT, "runs", out), C=C, Fh=Fh, S=4, steps=steps, sched=sched, teacher_v=tv, eval_every=ev,
                meta=dict(target=tname, src_frames=[TARGETS[tname][0], TARGETS[tname][0] + len(F) - 1], teacher=teacher), log=lambda s: print(s, flush=True))
    print(json.dumps(dict(res, models=len(res["models"]))))
