"""Path V figures from saved predictions (no new fits). Every CA frame shown is the actual output of the oracle-controlled
velocity CA (pv_v0 / pv_long); clips for the "success" figures were SELECTED after scoring (best test clip by lag-1 F1
in a category) and are demonstrations, not evidence of generalisation.
Writes runs/path_v/fig_*.png and fig_success.gif."""
import json, os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path[:0] = [ROOT, os.path.dirname(os.path.abspath(__file__))]
import pv_v0, pd_vis
OUT = pv_v0.OUT; X = pv_v0.video(); N = pv_v0.N
R = json.load(open(os.path.join(OUT, "v0_test.json"))); PR = np.load(os.path.join(OUT, "v0_test_preds.npz"))


def pred(start, method, grid, every):
    k = f"{start}|{method}|{grid}|{every}"; return np.unpackbits(PR[k])[:(N - 1) * 36 * 48].reshape(N - 1, 36, 48)


def pick(cat, method, grid, every, best=True):
    rows = [r for r in R if r["clip"]["cat"] == cat and r["method"] == method and r["grid"] == grid and r["every"] == every]
    return (max if best else min)(rows, key=lambda r: r["f1"] if best else r["iou"])


def sheet(start, method, grid, every, name, title):
    P = pred(start, method, str(list(grid)) if grid else "None", every); F = X[start:start + N]; B = np.concatenate([F[:1], P])
    lab = ["prefix"] + ["test"] * (N - 1); ev = set(range(0, N - 1, every)) if every else {0}
    pd_vis.contact_sheet(os.path.join(OUT, name), F, B, list(range(start, start + N)), lab, every=2, scale=3,
                         title=title + f" | events at +{sorted(ev)} (oracle control); rows: original, CA, diff (red = extra, blue = missing)")


def gif(start, method, grid, every, name):
    from PIL import Image, ImageDraw
    P = pred(start, method, str(list(grid)), every); F = X[start:start + N]; ims = []; sc = 6
    for t in range(N):
        a = F[t]; b = F[0] if t == 0 else P[t - 1]; im = Image.new("RGB", (2 * 48 * sc + 12, 36 * sc + 22), (246, 245, 242)); d = ImageDraw.Draw(im)
        for i, f in enumerate((a, b)): im.paste(Image.fromarray((np.repeat(np.repeat(f, sc, 0), sc, 1) * 255).astype(np.uint8)).convert("RGB"), (i * (48 * sc + 12), 20))
        d.text((2, 3), f"original {start + t}", fill=(0, 0, 0)); d.text((48 * sc + 14, 3), f"velocity CA (oracle {grid[0]}x{grid[1]}, events every {every})" + (" EVENT" if every and t % every == 0 and t < N - 1 else ""), fill=(180, 0, 0) if every and t % every == 0 else (0, 0, 0))
        ims.append(im)
    ims[0].save(os.path.join(OUT, name), save_all=True, append_images=ims[1:], duration=120, loop=0)


if __name__ == "__main__":
    s = pick("translation", "vcaL", [3, 3], 8); sheet(s["clip"]["start"], "vcaL", [3, 3], 8, "fig_success_translation.png",
        f'SELECTED best translation test clip {s["clip"]["start"]}: vcaL 3x3 every 8, err {s["err"]:.3f} (hold {s["hold"]:.3f}), F1 {s["f1"]:.2f}, IoU {s["iou"]:.2f}, {s["bits"]:.0f} bits')
    gif(s["clip"]["start"], "vcaL", (3, 3), 8, "fig_success.gif")
    b = [r for r in R if r["clip"]["start"] == s["clip"]["start"] and r["method"] == "block" and r["grid"] == [3, 3] and r["every"] == 8][0]
    sheet(s["clip"]["start"], "block", [3, 3], 8, "fig_success_translation_block.png", f'same clip, block shift 3x3 every 8: err {b["err"]:.3f}, F1 {b["f1"]:.2f}, IoU {b["iou"]:.2f}, {b["bits"]:.0f} bits')
    o = [r for r in R if r["clip"]["start"] == s["clip"]["start"] and r["method"] == "vcaL" and r["grid"] == [3, 3] and r["every"] is None][0]
    sheet(s["clip"]["start"], "vcaL", [3, 3], None, "fig_once_translation.png", f'same clip, ONE oracle event at frame 0 then free: err {o["err"]:.3f}, F1 {o["f1"]:.2f}, IoU {o["iou"]:.2f}, {o["bits"]:.0f} bits')
    f = pick("deformation", "vcaL", [3, 3], 8, best=False); sheet(f["clip"]["start"], "vcaL", [3, 3], 8, "fig_failure_deformation.png",
        f'FAILURE (lowest IoU deformation test clip {f["clip"]["start"]}): err {f["err"]:.3f} (hold {f["hold"]:.3f}), F1 {f["f1"]:.2f}, IoU {f["iou"]:.2f}')
    g = pick("major_transform", "vcaL", [3, 3], 8, best=False); sheet(g["clip"]["start"], "vcaL", [3, 3], 8, "fig_failure_major.png",
        f'FAILURE (lowest IoU major_transform test clip {g["clip"]["start"]}): err {g["err"]:.3f} (hold {g["hold"]:.3f}), F1 {g["f1"]:.2f}, IoU {g["iou"]:.2f}')
    print("figures:", s["clip"]["start"], f["clip"]["start"], g["clip"]["start"])
