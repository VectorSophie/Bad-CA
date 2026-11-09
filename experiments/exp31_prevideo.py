"""Exp31 (Campaign 09, before any training): the chosen task windows, original vs the Campaign 08 representative and
best (both autonomous from the clip's first frame, decoded from their export files), shown over the task frames only."""
import json, os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca import container
from ca.video_out import diff_rgb, make_video
from c7_common import target

idx = {x["name"]: x for x in json.load(open(os.path.join(ROOT, "runs", "c8", "media", "video_index.json"))) if "selection" in x["name"]}
OUT = os.path.join(ROOT, "runs", "c9", "media"); os.makedirs(OUT, exist_ok=True); checks = {}
for clip, s0, t0, sel in (("ba48_4500_60", 4500, 40, "main_ba48_4500_60_selection"), ("ba48_1800_60", 1800, 44, "second_ba48_1800_60_selection")):
    F = target(clip); rep = json.load(open(os.path.join(ROOT, idx[sel]["representative"])))
    _, P, info = container.decode(open(os.path.join(ROOT, rep["export"]["file"]), "rb").read())
    w = slice(t0, t0 + 16)
    checks[clip] = make_video([("original (task window)", F[w]), (f"C8 rep: arm B seed {rep['seed']}, {info['gates']} gates k=1 S=2", P[w]), ("diff", diff_rgb(P[w], F[w]))],
                              [f"task {clip} frames {s0 + t0}-{s0 + t0 + 15}; C8 replay is autonomous from frame {s0} (decoded export)", "red = missed white, blue = extra white"],
                              lambda t, a=s0 + t0: f"src frame {a + t}  t={(t0 + t) / 30:5.2f}s", os.path.join(OUT, f"pre_task_{clip}_c8"))
print(json.dumps({k: [c["ok"] for c in v] for k, v in checks.items()}))
