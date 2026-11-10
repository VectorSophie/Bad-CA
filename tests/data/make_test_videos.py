"""Regenerate the synthetic test clips in tests/data (made by this project; no external content).
color_disc_96x64.mp4: red disc moving right over a blue->green gradient, 40 frames.
color_ball_96x64.mp4: yellow ball moving diagonally over a dark red->blue gradient plus a static light bar, 48 frames.
Both 96x64, 30 fps, H.264 yuv420p via PyAV. usage: python tests/data/make_test_videos.py [OUT_DIR]"""
import os, sys
from fractions import Fraction
import av, numpy as np


def write(path, frames):
    c = av.open(path, "w"); s = c.add_stream("libx264", rate=Fraction(30)); s.width, s.height, s.pix_fmt = 96, 64, "yuv420p"
    for img in frames:
        for p in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")): c.mux(p)
    for p in s.encode(): c.mux(p)
    c.close()


def disc():
    yy, xx = np.mgrid[0:64, 0:96]
    for t in range(40):
        img = np.zeros((64, 96, 3), np.uint8); img[..., 2] = (255 * (1 - xx / 96)).astype(np.uint8); img[..., 1] = (200 * xx / 96).astype(np.uint8)
        img[(xx - (15 + t * 1.5)) ** 2 + (yy - 32) ** 2 < 14 ** 2] = (250, 30, 30); yield img


def ball():
    yy, xx = np.mgrid[0:64, 0:96]
    for t in range(48):
        img = np.zeros((64, 96, 3), np.uint8); img[..., 0] = (120 * (1 - xx / 96)).astype(np.uint8); img[..., 2] = (140 * xx / 96).astype(np.uint8)
        img[(xx - (12 + t * 1.5)) ** 2 + (yy - (14 + t * 0.8)) ** 2 < 10 ** 2] = (250, 220, 40); img[40:48, 10:80] = (230, 230, 230); yield img


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    write(os.path.join(out, "color_disc_96x64.mp4"), disc()); write(os.path.join(out, "color_ball_96x64.mp4"), ball())
