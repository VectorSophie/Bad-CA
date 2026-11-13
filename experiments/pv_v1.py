"""Path V1 (one targeted change): does lifting the 1 px/frame speed cap help? Same oracle search, same alphabet and bits
as vcaL, but 2 substeps per frame (speed in 1/2 px units, up to 2 px/frame). Test clips of the frozen V0 manifest,
3x3 grid, events every 8 frames and once (with the frames-1..8 persistence test). Compared with the V0 rows.
usage: pv_v1.py  -> runs/path_v/v1_s2.json"""
import json, os, sys
from multiprocessing import Pool
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path[:0] = [ROOT, os.path.dirname(os.path.abspath(__file__))]
import pv_v0


def job(a):
    pv_v0.SUB = 2; r, _ = pv_v0.job(a); r["method"] = r["method"] + "_s2"; return r


if __name__ == "__main__":
    clips = [c for c in json.load(open(pv_v0.MAN))["clips"] if c["split"] == "test"]
    J = [(c, m, (3, 3), e) for c in clips for m in ("vcaL", "vcaL_v") for e in (8, None)]
    with Pool(64) as p: rows = p.map(job, J, chunksize=1)
    json.dump(rows, open(os.path.join(pv_v0.OUT, "v1_s2.json"), "w"), default=float); print(len(rows), "rows")
