"""Track G / Track S checks.
1. Registry: both datasets listed; Fantasmagorie frames match the manifest shape; the local source (if present) has the
   recorded SHA-256 and Commons SHA-1; Bad Apple frames come from the original loader unchanged.
2. Splits: train / val / test windows never overlap across roles (windows are 18 frames); 'contig' budgets only use the
   first train blocks; 'spread' budgets have the requested size; both are deterministic.
3. Window evaluation alignment: a rule whose update is zero is exactly 'hold' after the 2-frame prefix, so its error
   equals the hold error on every window (prediction vs target frame indices line up).
4. Phased export round trip: the pilot export (if present) replays to the error recorded by training (numpy carry).
5. Shape CAs: zero field / zero growth are identities (ffdca without cleanup, ffdwarp, lsca on a shape without 1-px
   parts); a uniform field of +Q moves an image exactly 1 px per frame (ffdca == ffdwarp == shift); interp_field is
   exact at lattice nodes; morph dilate equals scipy dilation; lsca growth at g = +Q equals 4-neighbour dilation per frame."""
import hashlib, json, os, sys, tempfile
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
from ca import datasets
import gen_train as G, shape_ca as S, pv_vca as V

# 1. registry
assert datasets.ids() == ["bad_apple", "fantasmagorie"]
m = datasets.manifest("fantasmagorie"); F = datasets.frames("fantasmagorie"); assert F.shape == (1687, 36, 48) and set(np.unique(F)) <= {0, 1}
src = os.path.join(ROOT, m["source"]["local_path"])
if os.path.exists(src):
    data = open(src, "rb").read(); assert hashlib.sha256(data).hexdigest() == m["source"]["sha256"] and hashlib.sha1(data).hexdigest() == m["source"]["commons_sha1"]
from exp45_c12 import target
assert (datasets.frames("bad_apple")[:50] == target("0:50")).all()

# 2. splits
for T in (6572, 1687):
    win, blocks = G.splits(T); cover = {}
    for role, ws in win.items():
        for s in ws:
            for t in range(s, s + G.WL): cover.setdefault(t, set()).add(role)
    assert all(len(r) == 1 for r in cover.values()), T
    sp = G.pick_train(win, blocks, 0.25, "spread", 0); assert len(sp) == round(0.25 * len(win["train"])) and (sp == G.pick_train(win, blocks, 0.25, "spread", 0)).all()
    ct = G.pick_train(win, blocks, 0.25, "contig", 0); tb = [b for b in blocks if b[0] == "train"]; last = tb[round(0.25 * len(tb)) - 1][2]
    assert ct.max() < last and len(ct) > 0

# 3. hold rule alignment
with tempfile.TemporaryDirectory() as d:
    from ca import nca
    P = nca.init_params(np.random.default_rng(0), 16, 64); np.savez(os.path.join(d, "rule.npz"), **P)
    json.dump(dict(ds="fantasmagorie", rep=None, frac=1.0, mode="spread", seed=0, stride=1), open(os.path.join(d, "config.json"), "w"))
    r = G.eval_win(d)
    for split in ("val", "test"): assert abs(r[split]["err"] - r[split]["hold"]) < 1e-12 and r[split]["n"] > 0

# 4. phased export round trip
pf = os.path.join(ROOT, "runs/gen/pilot_fit")
if os.path.exists(os.path.join(pf, "model.ncab")):
    from ca import phase_nca
    hd, E, sched, h0, f0 = phase_nca.load(os.path.join(pf, "model.ncab")); B, _ = phase_nca.rollout_np(E, f0, sched, hd["S"], h0)
    rec = json.load(open(os.path.join(pf, "train", "chunks.json")))["chunks"][0]["replay_err_numpy"]
    assert abs(float((B[1:] != F[:hd["T"]][1:]).mean()) - rec) < 1e-6

# 5. shape CAs
f = np.zeros((36, 48), np.uint8); f[10:20, 12:26] = 1; f[14:16, 26:34] = 1
z = np.zeros((1, 3, 3, 2))
assert (S.ffdca_run(f, 5, {0: z}, (2, 2), cleanup=False)[0] == f).all() and (S.ffdwarp_run(f, 5, {0: z}, (2, 2))[0] == f).all()
blk = np.zeros((36, 48), np.uint8); blk[10:20, 12:26] = 1
assert (S.lsca_run(blk, 5, {0: np.zeros((1, 9, 2), int)}, V.region_map((3, 3)))[0] == blk).all()
u = np.zeros((1, 3, 3, 2)); u[..., 1] = V.Q
a1 = S.ffdca_run(f, 4, {0: u}, (2, 2), cleanup=False)[0]; a2 = S.ffdwarp_run(f, 4, {0: u}, (2, 2))[0]
for t in range(4): assert (a1[t] == V.sh(f, 0, t + 1)).all() and (a2[t] == V.sh(f, 0, t + 1)).all(), t
nodes = np.random.default_rng(0).normal(size=(1, 3, 4, 2)); fld = S.interp_field(nodes, 36, 48)
assert np.allclose(fld[0, 0, 0], nodes[0, 0, 0]) and np.allclose(fld[0, -1, -1], nodes[0, -1, -1])
from scipy import ndimage
dl = S.morph_run(f, 1, {0: np.ones((1, 9, 1), int)}, V.region_map((3, 3)))[0, 0]; assert (dl.astype(bool) == ndimage.binary_dilation(f, ndimage.generate_binary_structure(2, 1))).all()
g4 = np.zeros((1, 9, 2), int); g4[..., 0] = V.Q; gl = S.lsca_run(blk, 2, {0: g4}, V.region_map((3, 3)))[0]   # phase starts at Q/2: first layer at frame 1
cr = ndimage.generate_binary_structure(2, 1); assert (gl[0].astype(bool) == ndimage.binary_dilation(blk, cr)).all() and (gl[1].astype(bool) == ndimage.binary_dilation(blk, cr, 2)).all()
print("test_gen_shape: all checks passed")
