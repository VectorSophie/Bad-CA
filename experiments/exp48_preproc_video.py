"""Exp48 (Campaign 12): preprocessing loss example, separate from CA loss. Panels: source frames converted to gray at full
480x360 / the 48x36 box-resized gray (nearest upscaled x10) / the binary target (gray >= 128) used by every CA.
usage: exp48_preproc_video.py START LEN"""
import os, sys
import av, numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from ca.video import binarize, gray_frames
from ca.video_out import make_video
a, n = int(sys.argv[1]), int(sys.argv[2]); src = os.path.join(ROOT, "refs", "Bad-Apple-Starter", "BadApple!!.mp4"); full = []
with av.open(src) as c:
    for i, fr in enumerate(c.decode(video=0)):
        if i >= a + n: break
        if i >= a: full.append(np.asarray(fr.to_image().convert("L")))
g48 = gray_frames(src, 48)[0][a:a + n]; up = lambda X: np.repeat(np.repeat(X, 10, 1), 10, 2)
rgb = lambda X: np.repeat(X[..., None], 3, -1)
ch = make_video([("source, gray 480x360", rgb(np.stack(full))), ("48x36 gray (box), x10", rgb(up(g48))), ("binary target (>=128), x10", up(binarize(g48)))],
                ["PREPROCESSING ONLY (no CA): loss from 480x360 -> 48x36 box resize -> threshold 128", "every CA in this project is trained and scored against the right panel"],
                lambda t: f"src {a + t}  t={(a + t) / 30:6.2f}s", os.path.join(ROOT, "runs", "c12", "media", f"c12_preprocessing_loss_{a}-{a + n - 1}"), scale=1)
print([c["ok"] for c in ch])
