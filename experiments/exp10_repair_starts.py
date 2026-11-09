"""Exp10 (Campaign 04, part 2): starting rules for SAT repair, all in the target family F_priv k=1
(n = 10 taps: visible Moore bits 0..8, own hidden bit 9; outputs visible, hidden).

  keep     visible -> own centre bit, hidden -> own hidden bit (the exp03c default)
  conway   explicit same-family embedding of B3/S23: visible output = Life(visible 3x3, bits 0..8),
           ignoring the hidden bit; hidden output = own hidden bit (keep). With S=2 the embedded rule
           applies Life twice per observed frame.
  random   fixed uniform random LUT, numpy default_rng(0).random((2, 1024)) < 0.5
  learned  a hardened DiffLogic LUT (ca.difflogic.hard_lut), given by path; its GPU cost is recorded
           from the producing run, separately from the SAT repair cost.

The repair is exp03c with REF_LUTS = start: solver decision phases prefer the start's value for every
LUT entry (search PREFERENCE, never a hard constraint), and the anytime objective is Hamming distance
to the start, so every exact rule found is reported with an UPPER BOUND on its repair distance.
usage: exp10_repair_starts.py OUTDIR [learned_name=path.npy ...]
"""
import json, os, shutil, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from ca.core import life_like_lut
from ca.rule import family_taps
from ca import costs

taps = family_taps(1); n = len(taps); idx = np.arange(2 ** n)
assert taps[4] == (0, 0, 0) and taps[9] == (1, 0, 0)
out = os.path.join(ROOT, "runs", sys.argv[1]); os.makedirs(out, exist_ok=True)
starts = {
    "keep": costs.default_luts(taps, 2),
    "conway": np.stack([life_like_lut({3}, {2, 3})[idx & 511], (idx >> 9) & 1]).astype(np.uint8),
    "random": (np.random.default_rng(0).random((2, 2 ** n)) < 0.5).astype(np.uint8),
}
for a in sys.argv[2:]:
    name, p = a.split("=", 1)
    starts[name] = np.load(p).astype(np.uint8)
    shutil.copy(os.path.join(os.path.dirname(p), "result.json"), os.path.join(out, f"{name}.source_result.json"))
meta = {}
for name, L in starts.items():
    assert L.shape == (2, 2 ** n)
    np.save(os.path.join(out, f"{name}.npy"), L)
    meta[name] = dict(lut_entries=int(L.size), ones=int(L.sum()), keep_exceptions=int((L != starts["keep"]).sum()))
json.dump(meta, open(os.path.join(out, "starts.json"), "w"), indent=1)
print(json.dumps(meta))
