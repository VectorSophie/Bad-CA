"""Assemble PNG screenshots (sorted by name) into an MP4 at a given seconds-per-image; verifies by decoding back.
usage: assemble_frames.py PNG_DIR OUT_STEM SECONDS_PER_IMAGE"""
import glob, os, sys
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ca.video_out import verify_mp4, write_mp4
d, stem, spi = sys.argv[1], sys.argv[2], float(sys.argv[3]); fps = 10
ims = [np.asarray(Image.open(p).convert("RGB")) for p in sorted(glob.glob(os.path.join(d, "*.png")))]
H = max(i.shape[0] for i in ims); W = max(i.shape[1] for i in ims); H += H % 2; W += W % 2
pad = lambda a: np.pad(a, ((0, H - a.shape[0]), (0, W - a.shape[1]), (0, 0)))
frames = [pad(a) for a in ims for _ in range(int(round(spi * fps)))]
write_mp4(frames, stem + ".mp4", fps); print(verify_mp4(stem + ".mp4", len(frames), fps))
