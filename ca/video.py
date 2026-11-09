"""Video -> grayscale at target resolution (cached) -> binary (derived, cheap).

Cache key = source sha256 + PREPROC_VERSION + geometry + stride. Threshold is applied
after the cache so threshold sweeps don't re-decode.
shortcut: whole-clip .npz cache, no chunked/lazy store yet; upgrade to zarr/memmap chunks
when full-song multi-resolution variants are requested from the UI.
"""
import hashlib, json, os
import av
import numpy as np
from PIL import Image

PREPROC_VERSION = 1
CACHE = os.path.join(os.path.dirname(__file__), "..", "cache")


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def gray_frames(path, width, stride=1):
    """Return (gray uint8 (T,H,W), timestamps seconds (T,)). Height follows source aspect."""
    src = _sha(path)
    key = hashlib.sha256(json.dumps([src, PREPROC_VERSION, width, stride]).encode()).hexdigest()[:16]
    fn = os.path.join(CACHE, f"gray_{key}.npz")
    if os.path.exists(fn):
        z = np.load(fn)
        return z["gray"], z["t"]
    frames, ts = [], []
    with av.open(path) as c:
        s = c.streams.video[0]
        # shortcut: display-rotation metadata ignored; handle it before accepting phone uploads
        height = None
        for i, fr in enumerate(c.decode(s)):
            if i % stride:
                continue
            img = fr.to_image().convert("L")  # applies colour matrix; ITU luma
            if height is None:
                height = round(width * img.height / img.width)
            frames.append(np.asarray(img.resize((width, height), Image.BOX)))
            ts.append(float(fr.pts * s.time_base))
    gray, t = np.stack(frames), np.array(ts)
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(fn, gray=gray, t=t, meta=json.dumps(dict(src=src, v=PREPROC_VERSION, width=width, stride=stride)))
    return gray, t


def binarize(gray, threshold=128):
    return (gray >= threshold).astype(np.uint8)
