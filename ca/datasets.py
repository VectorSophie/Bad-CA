"""Versioned dataset registry (data/manifests/<id>.json). New experiments load frames through `frames(id, rep)` instead
of hard-coding the Bad Apple path; existing campaign scripts keep their own loaders (unchanged, backwards compatible).

  ids()                    -> registered dataset ids
  manifest(id)             -> the manifest dict (source, hash, timing, rights, derived representations, statistics)
  frames(id, rep=None)     -> (T, H, W) uint8 binary frames of a derived representation (default: the dataset's
                              primary one); Bad Apple's comes from the original pipeline (exp45_c12.target, 48x36, >= 128)
Large sources live under data/source/ (git-ignored); derived binary films under data/derived/<id>/ (*.npz, packed bits)."""
import json, os
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")); MAN = os.path.join(ROOT, "data", "manifests")
PRIMARY = {"bad_apple": "binary_48x36_v1", "fantasmagorie": "binary_48x36_lines_tm3_v1"}


def ids(): return sorted(f[:-5] for f in os.listdir(MAN) if f.endswith(".json"))


def manifest(ds): return json.load(open(os.path.join(MAN, ds + ".json")))


def frames(ds, rep=None):
    rep = rep or PRIMARY[ds]; m = manifest(ds)
    if rep not in m["derived"]: raise KeyError(f"{ds} has no representation {rep}; known: {sorted(m['derived'])}")
    if ds == "bad_apple" and rep == "binary_48x36_v1":
        import sys; sys.path.insert(0, os.path.join(ROOT, "experiments"))
        from exp45_c12 import target
        return target("0:6572").astype(np.uint8)
    z = np.load(os.path.join(ROOT, m["derived"][rep]["file"]))
    return np.unpackbits(z["bits"])[:int(np.prod(z["shape"]))].reshape(tuple(z["shape"])).astype(np.uint8)
