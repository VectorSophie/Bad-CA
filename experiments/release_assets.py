"""Release assets (GIF/PNG) rendered ONLY from saved exports or saved predictions; every CA frame is replayed by the
project's executors (ca.dlca.replay, ca.phase_nca.rollout_np, saved Path V / Track S predictions). Writes assets/release/.
Each asset gets a caption entry in assets/release/captions.json (what it shows, what it does not show).
usage: release_assets.py [name ...]   names: gh9 logicfail ncastate ncacompare shape fanca"""
import json, os, sys
import numpy as np
from PIL import Image, ImageDraw
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path[:0] = [ROOT, os.path.dirname(os.path.abspath(__file__))]
OUT = os.path.join(ROOT, "assets", "release"); CAP = os.path.join(OUT, "captions.json")
BG, FG, ACC = (246, 245, 242), (20, 20, 20), (200, 40, 40)


def panel(frames, labels, title, sc=5, colors=None):
    """frames: list of 2-D arrays (0/1 or 0..1) or RGB arrays, side by side with labels; returns a PIL image."""
    H, W = frames[0].shape[:2]; pad = 10; top = 34; img = Image.new("RGB", (len(frames) * (W * sc + pad) + pad, H * sc + top + 6), BG); d = ImageDraw.Draw(img)
    d.text((pad, 4), title, fill=FG)
    for i, (f, lab) in enumerate(zip(frames, labels)):
        a = f if f.ndim == 3 else np.stack([(np.clip(f, 0, 1) * 255).astype(np.uint8)] * 3, -1)
        img.paste(Image.fromarray(np.repeat(np.repeat(a, sc, 0), sc, 1)), (pad + i * (W * sc + pad), top)); d.text((pad + i * (W * sc + pad), top - 13), lab, fill=FG)
    return img


def save_gif(imgs, name, ms=100):
    imgs[0].save(os.path.join(OUT, name), save_all=True, append_images=imgs[1:], duration=ms, loop=0, optimize=True)


def caption(name, text):
    c = json.load(open(CAP)) if os.path.exists(CAP) else {}; c[name] = text; json.dump(c, open(CAP, "w"), indent=1, ensure_ascii=False)


def gh9():
    from ca import dlca
    import pd_d1
    d = dlca.load(os.path.join(ROOT, "ui/static/logic_examples/greenberg_hastings.logic.json"))
    d["init_state"] = pd_d1.init_states("gh", 1, np.random.default_rng(1), size=48, density=(0.004, 0.004))[0].astype(np.uint8)   # D2.5 regime: spirals persist
    B = dlca.replay(d, 160); imgs = []
    for t in range(0, 160, 2):
        rgb = np.full(B.shape[1:3] + (3,), 15, np.uint8); rgb[B[t, ..., 1] == 1] = (60, 90, 200); rgb[B[t, ..., 0] == 1] = (255, 230, 120)
        imgs.append(panel([rgb], [f"step {t}"], "9 gates = Greenberg-Hastings", sc=5))
    save_gif(imgs, "logic_gh_9gates.gif", 80)
    caption("logic_gh_9gates.gif", "Replay of ui/static/logic_examples/greenberg_hastings.logic.json (9 gates) from a sparse random start (the D2.5 regime; seed 1). Yellow = excited, blue = refractory. "
            "A learned 1,598-gate circuit simplified against its own dynamics; it equals the KNOWN generating rule on all 262,144 neighbourhoods. "
            "Recovery of a synthetic rule, not a rule discovered from Bad Apple.")


def logicfail():
    from ca import dlca
    d = dlca.load(os.path.join(ROOT, "ui/static/logic_examples/d4_G_s0_b144k.logic.json")); B = dlca.replay(d)[..., 0]
    from exp45_c12 import target
    s0 = d["meta"]["source_frames"][0]; F = target(f"{s0}:{len(B)}").astype(np.uint8); imgs = []
    for t in range(len(B)):
        imgs.append(panel([F[t], B[t]], [f"original {s0 + t}", "503-gate logic CA" + (" (prefix)" if t <= 2 else "")],
                          "Path D4: a learned Boolean circuit on real Bad Apple, continuous run (fails after ~5-9 steps)", sc=5))
    save_gif(imgs, "logic_realvideo_failure.gif", 160)
    caption("logic_realvideo_failure.gif", f"Replay of ui/static/logic_examples/d4_G_s0_b144k.logic.json from frame {s0} with its 2-frame observation prefix, then free. "
            "The circuit holds the silhouette for a few steps, then turns it into a growing blob (docs/path-d/D4_LOG.md). Left: derived 48x36 frames of Bad Apple!!.")


def nca_frames(a, b):
    """Featured whole-song NCA: numpy replay of the export from frame 0; returns visible (b-a) frames and 3 hidden channels."""
    from ca import phase_nca, nca
    hd, E, sched, h0, f0 = phase_nca.load(os.path.join(ROOT, "runs/featured/whole_song_s0.ncab")); C = E["Wu"].shape[1]
    s = np.zeros((C,) + f0.shape, np.float32); s[0] = f0
    if h0 is not None: s[1:] = h0
    vis, hid = [], []
    for t in range(1, b):
        p = {k: E[k][sched[t]] for k in phase_nca.PNAMES}
        for _ in range(hd["S"]): s = nca.step_np(p, s)
        if t >= a: vis.append((s[0] > 0.5).astype(np.uint8)); hid.append(s[1:4].copy())
    return np.stack(vis), np.stack(hid), sched[a:b]


def ncastate():
    a, b = 1850, 1940; V, Hd, sc_ = nca_frames(a, b); imgs = []
    lo, hi = np.percentile(Hd, 1, axis=(0, 2, 3)), np.percentile(Hd, 99, axis=(0, 2, 3))
    for t in range(0, len(V), 2):
        ch = [(Hd[t, c] - lo[c]) / max(hi[c] - lo[c], 1e-6) for c in range(3)]
        imgs.append(panel([V[t]] + ch, [f"visible, frame {a + t}", "hidden ch 1", "hidden ch 2", "hidden ch 3"],
                          f"Featured Neural CA state (phase {int(sc_[t])} of 110): every cell runs the same rule on its 3x3 neighbourhood, 4 substeps per frame", sc=3).quantize(64))
    save_gif(imgs, "nca_state.gif", 133)
    caption("nca_state.gif", "Numpy replay of runs/featured/whole_song_s0.ncab from frame 0 (no frames injected), shown for frames 1850-1939: the thresholded visible channel and 3 of the "
            "15 hidden channels (each min-max scaled over the clip). The rule in effect changes at phase boundaries (external schedule).")


def ncacompare():
    a, b = 1850, 2030; V, _, _ = nca_frames(a, b)
    from exp45_c12 import target
    F = target(f"{a}:{b - a}").astype(np.uint8); imgs = [panel([F[t], V[t]], [f"original (48x36) {a + t}", "Neural CA replay"], "Bad CA!!: original vs. phased Neural CA (same frames)", sc=5) for t in range(0, len(V), 2)]
    save_gif(imgs, "nca_vs_original.gif", 66)
    caption("nca_vs_original.gif", "Frames 1850-2029 (every 2nd), original 48x36 binary frame vs. the featured export's numpy replay from frame 0. The NCA uses 110 phase rules on an "
            "external schedule plus carried hidden state; it reproduces this one video (4.2 MB model), it is not a general video generator.")


def shape():
    z = np.load(os.path.join(ROOT, "runs/shape/synth_preds.npz")); import shape_s
    F = shape_s.task("ring"); P = {m: np.unpackbits(z[f"ring|{m}"])[:15 * 36 * 48].reshape(15, 36, 48) for m in ("block", "vcaL", "ffdca")}; imgs = []
    for t in range(16):
        fr = [F[t]] + [F[0] if t == 0 else P[m][t - 1] for m in ("block", "vcaL", "ffdca")]
        imgs.append(panel(fr, ["intended", "block shift 3x3", "velocity CA 3x3", "lattice transport (S2)"], "Track S synthetic: squashing a thin ring with oracle controls every 4 frames", sc=4))
    save_gif(imgs, "shape_ring.gif", 160)
    caption("shape_ring.gif", "runs/shape/synth_preds.npz. All three are hand-written rules driven by ORACLE controls chosen from the intended frames. The lattice transport (9 control "
            "nodes, smooth interpolated field; the interpolation is non-local) keeps the ring connected; region-wise shifts tear it.")


def fanca():
    run = os.path.join(ROOT, "runs/gen/fit/fa_full");
    if not os.path.exists(os.path.join(run, "replay_bits.npz")): print("no fa_full replay yet"); return
    from ca import datasets
    z = np.load(os.path.join(run, "replay_bits.npz")); B = np.unpackbits(z["bits"])[:int(np.prod(z["shape"]))].reshape(tuple(z["shape"])); F = datasets.frames("fantasmagorie")[:len(B)]
    a = json.load(open(os.path.join(run, "gif_range.json")))["start"] if os.path.exists(os.path.join(run, "gif_range.json")) else 160
    imgs = [panel([F[t], B[t]], [f"Fantasmagorie (1908) 48x36, frame {a + t}" if False else f"source-derived frame {t}", "phased Neural CA replay"], "External video: Fantasmagorie (1908, public domain), same architecture trained from scratch", sc=5) for t in range(a, min(a + 200, len(B)), 2)]
    save_gif(imgs, "fantasmagorie_nca.gif", 80)
    caption("fantasmagorie_nca.gif", "runs/gen/fit/fa_full: the C13 recipe trained from scratch on the animation part of Fantasmagorie (lines_tm3 preprocessing). Replay of the export from frame 0. "
            "Evidence that the architecture can be re-fitted to another video, not that one model generalises across videos.")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for n in (sys.argv[1:] or ["gh9", "logicfail", "ncastate", "ncacompare", "shape"]): globals()[n](); print("done", n, flush=True)
