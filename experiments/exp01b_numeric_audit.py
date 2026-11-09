"""Audit of CHECKPOINT_01 numbers, with exact integer counts.

Definitions (per clip, Moore r=1 LUT family, boundary 0, 1 gen/frame, binary visible only):
  N        = cells * (T-1)                       all observed one-step transitions
  changes  = #{(t,cell): f[t+1] != f[t]}
  minerr   = sum over contexts c of min(n0(c), n1(c))   (one-step error of the best LUT;
             attained by majority vote, i.e. teacher-forced optimum, NOT a rollout bound)
  explained_change = 1 - minerr/changes   (changes the best LUT predicts, minus the
             unchanged pixels it breaks, net, as a share of changes)
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ca.core import transition_counts, neigh_index
from ca.video import binarize, gray_frames

mp4 = os.path.join(os.path.dirname(__file__), "..", "refs", "Bad-Apple-Starter", "BadApple!!.mp4")


def audit(clip, r=1, boundary=0):
    c, _ = transition_counts(clip, r, boundary)
    N = int(c.sum()); minerr = int(np.minimum(c[:, 0], c[:, 1]).sum())
    changes = int((clip[1:] != clip[:-1]).sum())
    # which side is the minority? count changed transitions that the majority LUT gets right
    idx = np.concatenate([neigh_index(f, r, boundary).ravel() for f in clip[:-1]])
    cur = np.concatenate([f.ravel() for f in clip[:-1]]); nxt = np.concatenate([f.ravel() for f in clip[1:]])
    if c.shape[0] == 2 ** ((2 * r + 1) ** 2):
        maj = (c[idx, 1] > c[idx, 0])
        tie = c[idx, 1] == c[idx, 0]
        pred = np.where(tie, cur, maj)
    else:
        pred = None
    out = dict(N=N, changes=changes, minerr=minerr, minerr_equals_changes=minerr == changes,
               conflicting_contexts=int(((c[:, 0] > 0) & (c[:, 1] > 0)).sum()), contexts_seen=int((c.sum(1) > 0).sum()),
               explained_change=1 - minerr / changes if changes else None)
    if pred is not None:
        ch = cur != nxt
        out.update(changes_predicted=int((pred[ch] == nxt[ch]).sum()), unchanged_broken=int((pred[~ch] != nxt[~ch]).sum()))
    return out


R = {}
for W in (16, 32, 64, 128):
    B = binarize(gray_frames(mp4, W)[0])
    rows = {}
    for s in (10, 30, 62, 100, 150):
        clip = B[s * 30:s * 30 + 90]
        rows[f"t{s}s"] = {"r1": audit(clip, 1), "r2": audit(clip, 2)}
    for r in ("r1", "r2"):
        clips = [k for k in rows if k.startswith("t")]
        e = [rows[k][r]["explained_change"] for k in clips]
        pooled = 1 - sum(rows[k][r]["minerr"] for k in clips) / sum(rows[k][r]["changes"] for k in clips)
        rows[f"summary_{r}"] = dict(unweighted_mean_of_clips=float(np.mean(e)), pooled=float(pooled),
                                    min=float(np.min(e)), max=float(np.max(e)))
    R[f"w{W}"] = rows

# synthetic: exact conflict counts
from ca.synthetic import grow_shrink, rotate_bar, translate
R["synthetic"] = {k: audit(f()) for k, f in [("translate", translate), ("grow_shrink", grow_shrink), ("rotate_bar", rotate_bar)]}
os.makedirs(os.path.join(os.path.dirname(__file__), "..", "runs", "exp01b"), exist_ok=True)
json.dump(R, open(os.path.join(os.path.dirname(__file__), "..", "runs", "exp01b", "audit.json"), "w"), indent=1)

for W in (16, 32, 64, 128):
    print(f"w{W}", {r: {k: round(v, 4) for k, v in R[f'w{W}'][f'summary_{r}'].items()} for r in ('r1', 'r2')})
print("w64 per clip r1:")
for k, v in R["w64"].items():
    if k.startswith("t"):
        a = v["r1"]; print(f"  {k}: N={a['N']} changes={a['changes']} minerr={a['minerr']} equal={a['minerr_equals_changes']} "
                           f"predicted_changes={a['changes_predicted']} broken_unchanged={a['unchanged_broken']} explained={a['explained_change']:.4f}")
for k, a in R["synthetic"].items():
    print(f"syn {k}: N={a['N']} changes={a['changes']} minerr={a['minerr']} conflicting_ctx={a['conflicting_contexts']}/{a['contexts_seen']} "
          f"predicted_changes={a['changes_predicted']} broken_unchanged={a['unchanged_broken']}")
