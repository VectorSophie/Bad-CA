"""Dataset registry (data/manifests/<id>.json). New code loads frames through `frames(id, rep)` instead of hard-coding
a video path; older campaign scripts keep their own loaders (unchanged).

  ids()                    registered dataset ids
  manifest(id)             the manifest dict (source, hash, timing, rights, derived representations, statistics)
  frames(id, rep=None)     (T, H, W) uint8 binary frames of a derived representation (default: the dataset's primary
                           one). Bad Apple's 48x36 frames come from the original pipeline (exp45_c12.target).
  python -m ca.datasets    list datasets and which representations are available locally

Paths in manifests are relative to the data root: the checkout by default, or $BADCA_DATA if set (keep large sources
and derived arrays outside the repository by pointing it at a directory with the same data/... layout).
Large sources live under data/source/ (git-ignored); derived binary films under data/derived/<id>/ (packed .npz)."""
import json, os, sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")); MAN = os.path.join(ROOT, "data", "manifests")
PRIMARY = {"bad_apple": "binary_48x36_v1", "fantasmagorie": "binary_48x36_lines_tm3_v1"}
HOW = {"bad_apple": "Bad Apple!! is not distributed here: place the video at refs/Bad-Apple-Starter/BadApple!!.mp4 (README, Data and licence); "
                    "96x72 frames are then made with `python experiments/gen_data.py ba96`",
       "fantasmagorie": "download the public-domain film into data/source/fantasmagorie/ (source URL in data/manifests/fantasmagorie.json), "
                        "then run `python experiments/gen_data.py prep`"}


def data_root(): return os.environ.get("BADCA_DATA", ROOT)


def ids(): return sorted(f[:-5] for f in os.listdir(MAN) if f.endswith(".json"))


def manifest(ds):
    p = os.path.join(MAN, ds + ".json")
    if not os.path.exists(p): raise KeyError(f"unknown dataset {ds!r}; registered: {ids()}")
    return json.load(open(p))


def _path(ds, rep):
    d = manifest(ds)["derived"][rep]
    return os.path.join(data_root(), d["file"]) if "file" in d else None


def frames(ds, rep=None):
    m = manifest(ds); rep = rep or PRIMARY[ds]
    if rep not in m["derived"]: raise KeyError(f"{ds} has no representation {rep!r}; known: {sorted(m['derived'])}")
    if ds == "bad_apple" and rep == "binary_48x36_v1":
        sys.path.insert(0, os.path.join(ROOT, "experiments"))
        from exp45_c12 import target, SRC
        if not os.path.exists(SRC): raise FileNotFoundError(f"{SRC} is missing. {HOW[ds]}")
        return target("0:6572").astype(np.uint8)
    p = _path(ds, rep)
    if not os.path.exists(p): raise FileNotFoundError(f"{p} is missing. {HOW[ds]}")
    z = np.load(p)
    return np.unpackbits(z["bits"])[:int(np.prod(z["shape"]))].reshape(tuple(z["shape"])).astype(np.uint8)


if __name__ == "__main__":
    for ds in ids():
        m = manifest(ds); print(f"{ds}: {m.get('title', '')}")
        for rep, d in m["derived"].items():
            p = _path(ds, rep); ok = os.path.exists(p) if p else os.path.exists(os.path.join(ROOT, "refs", "Bad-Apple-Starter", "BadApple!!.mp4"))
            print(f"  {rep:28s} {'available' if ok else 'missing  '}  {'(primary) ' if rep == PRIMARY[ds] else ''}{d.get('preprocessing', d.get('loader', ''))[:70]}")
