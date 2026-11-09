"""Tiny synthetic target sequences (T,H,W) uint8. Defaults reproduce the 24x24 exp01 clips."""
import numpy as np


def translate(n=24):
    fs = []
    for t in range(12):
        c = np.zeros((n, n), np.uint8); c[8:14, 3 + t:9 + t] = 1; fs.append(c)
    return np.stack(fs)


def grow_shrink(n=24, radii=(2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2)):
    yy, xx = np.mgrid[:n, :n]; m = (n - 1) / 2
    return np.stack([((yy - m) ** 2 + (xx - m) ** 2 < r ** 2).astype(np.uint8) for r in radii])


def rotate_bar(n=24, half=9, thick=1.6, frames=12):
    yy, xx = np.mgrid[:n, :n] - (n - 1) / 2; fs = []
    for k in range(frames):
        a = np.pi * k / frames
        d = np.abs(-np.sin(a) * xx + np.cos(a) * yy); along = np.abs(np.cos(a) * xx + np.sin(a) * yy)
        fs.append(((d < thick) & (along < half)).astype(np.uint8))
    return np.stack(fs)


TINY = {
    "disc12": lambda: grow_shrink(12, (2, 3, 4, 5, 4, 3, 2)),
    "bar12": lambda: rotate_bar(12, 5, 1.2, 12),
}
