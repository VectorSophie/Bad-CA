"""Track G data: Fantasmagorie (Emile Cohl, 1908) acquisition record, preprocessing comparison and redundancy statistics
for Fantasmagorie and Bad Apple. Writes data/manifests/{fantasmagorie,bad_apple}.json and data/derived/<id>/... .
The source video is never modified; derived arrays are regenerated from it (data/source and large derived arrays are
git-ignored; the 48x36 / 96x72 binary films of the public-domain film are committed as packed .npz).

Fantasmagorie preprocessing candidates (all from the decoded luma, ITU-R 601-2 via libswscale 'gray'):
  simple   the Bad Apple recipe: box resize 720x540 -> 48x36, binary = gray >= 128            (pipeline v1)
  otsu     box resize -> 48x36, per-frame Otsu threshold                                        (flicker-adaptive)
  lines    box resize to 192x144 (x3.75); remove the slowly varying background (flicker, vignetting, grain level)
           by subtracting a 15x15 median; line pixel = residual >= 24; then a cell of the target grid is 1 if
           >= 2 of its 16 (48x36) / 4 (96x72) line pixels are set (area-fraction pooling keeps 1-2 px lines that a
           box average + global threshold erases).
  lines_tm3  lines + a centred 3-frame temporal majority (first/last frame kept): removes isolated one-frame flips.
           SELECTED as the training target. Reason (measured, prep): 27-32 % of all binary changes in every
           variant are one-frame blips (pixel differs from both neighbours, which agree) - film grain, flicker and gate
           weave on a ~16 fps film in a 25 fps transfer - and a model must not be rewarded for reproducing them.
           The unfiltered `lines` film is kept and reported.
Animation interval: the title cards added by the DVD edition (frames 0..~740) and the trailing black are excluded by a
fixed rule: first / last frame whose mean luma at 192x144 exceeds 5 after frame 700.
usage: gen_data.py probe | prep | stats"""
import hashlib, json, os, sys, time
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path[:0] = [ROOT, os.path.dirname(os.path.abspath(__file__))]
SRC = os.path.join(ROOT, "data", "source", "fantasmagorie", "Fantasmagorie_(1908).webm"); DER = os.path.join(ROOT, "data", "derived")
MAN = os.path.join(ROOT, "data", "manifests"); PREP_VERSION = 1


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def decode_gray(path, size=(192, 144)):
    import av
    from PIL import Image
    out, ts = [], []
    with av.open(path) as c:
        s = c.streams.video[0]
        for fr in c.decode(s): out.append(np.asarray(Image.fromarray(fr.to_ndarray(format="gray")).resize(size, Image.BOX))); ts.append(float(fr.pts * s.time_base))
    return np.stack(out), np.array(ts)


def interval(G):
    m = G.reshape(len(G), -1).mean(1); idx = np.flatnonzero(m[700:] > 5) + 700; return int(idx[0]), int(idx[-1]) + 1


def otsu(g):
    h = np.bincount(g.ravel(), minlength=256).astype(float); p = h / h.sum(); w = np.cumsum(p); mu = np.cumsum(p * np.arange(256)); mt = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"): sb = (mt * w - mu) ** 2 / (w * (1 - w))
    return int(np.nanargmax(sb))


def pool(mask, f, k):
    H, W = mask.shape[-2] // f, mask.shape[-1] // f
    return (mask[..., :H * f, :W * f].reshape(mask.shape[:-2] + (H, f, W, f)).sum((-3, -1)) >= k).astype(np.uint8)


def methods(G192):
    from PIL import Image
    from scipy import ndimage
    g48 = np.stack([np.asarray(Image.fromarray(g).resize((48, 36), Image.BOX)) for g in G192])
    simple = (g48 >= 128).astype(np.uint8); ots = np.stack([(g >= otsu(g)).astype(np.uint8) for g in g48])
    bg = np.stack([ndimage.median_filter(g, size=15) for g in G192]); line = (G192.astype(np.int16) - bg) >= 24
    return dict(simple=simple, otsu=ots, lines=pool(line, 4, 2), lines96=pool(line, 2, 2), line192=line.astype(np.uint8))


def redundancy(B, fps):
    """Binary-level temporal redundancy of a (T, H, W) film."""
    ch = (B[1:] != B[:-1]).mean((1, 2)); ident = ch == 0
    holds = []; r = 1
    for x in ident:
        if x: r += 1
        else: holds.append(r); r = 1
    holds.append(r)
    uniq = len({b.tobytes() for b in B})
    return dict(frames=len(B), fps=fps, seconds=round(len(B) / fps, 2), identical_consecutive=float(ident.mean()), near_dup_le_0p2pct=float((ch <= 0.002).mean()),
                distinct_frames=uniq, mean_change=float(ch.mean()), median_change=float(np.median(ch)), hold_length_hist={str(k): holds.count(k) for k in sorted(set(holds))[:12]},
                fg_fraction=float(B.mean()), empty_frames=int((B.reshape(len(B), -1).sum(1) == 0).sum()))


def prep():
    t0 = time.time(); G, ts = decode_gray(SRC); a, b = interval(G); d = os.path.join(DER, "fantasmagorie"); os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "gray_192x144.npy"), G); M = methods(G[a:b])
    from PIL import Image, ImageDraw
    pick = list(range(0, b - a, (b - a) // 12))[:12]; sc = 3; rows = ("line192", "simple", "otsu", "lines", "lines96")
    im = Image.new("L", (len(pick) * (192 + 4), len(rows) * (144 + 14)), 90); dr = ImageDraw.Draw(im)
    for r, name in enumerate(rows):
        for k, i in enumerate(pick):
            x = M[name][i]; x = np.asarray(Image.fromarray(x * 255).resize((192, 144), Image.NEAREST)); im.paste(Image.fromarray(x), (k * 196, r * 158 + 12))
            if r == 0: dr.text((k * 196, 0), f"{a + i}", fill=255)
        dr.text((2, r * 158 + 1), name, fill=255)
    im.save(os.path.join(d, "preproc_compare.png"))
    q = {n: dict(fg_fraction=float(M[n].mean()), empty_frames=int((M[n].reshape(len(M[n]), -1).sum(1) == 0).sum()), mean_change=float((M[n][1:] != M[n][:-1]).mean()))
         for n in ("simple", "otsu", "lines", "lines96")}
    np.savez_compressed(os.path.join(d, "binary_48x36_lines_v1.npz"), bits=np.packbits(M["lines"]), shape=M["lines"].shape, source_frames=np.arange(a, b))
    np.savez_compressed(os.path.join(d, "binary_96x72_lines_v1.npz"), bits=np.packbits(M["lines96"]), shape=M["lines96"].shape, source_frames=np.arange(a, b))
    L = M["lines"]; tm = L.copy(); tm[1:-1] = ((L[:-2].astype(int) + L[1:-1] + L[2:]) >= 2).astype(np.uint8)
    np.savez_compressed(os.path.join(d, "binary_48x36_lines_tm3_v1.npz"), bits=np.packbits(tm), shape=tm.shape, source_frames=np.arange(a, b))
    L9 = M["lines96"]; tm9 = L9.copy(); tm9[1:-1] = ((L9[:-2].astype(int) + L9[1:-1] + L9[2:]) >= 2).astype(np.uint8)
    np.savez_compressed(os.path.join(d, "binary_96x72_lines_tm3_v1.npz"), bits=np.packbits(tm9), shape=tm9.shape, source_frames=np.arange(a, b))
    for n, x in (("lines_tm3", tm), ("lines96_tm3", tm9)):
        q[n] = dict(fg_fraction=float(x.mean()), empty_frames=int((x.reshape(len(x), -1).sum(1) == 0).sum()), mean_change=float((x[1:] != x[:-1]).mean()))
    for n in q:
        x = {"lines_tm3": tm, "lines96_tm3": tm9}.get(n, M.get(n)).astype(np.int8); q[n]["blip_rate"] = float(((x[:-2] == x[2:]) & (x[1:-1] != x[:-2])).mean())
    np.savez_compressed(os.path.join(d, "binary_48x36_simple_v1.npz"), bits=np.packbits(M["simple"]), shape=M["simple"].shape, source_frames=np.arange(a, b))
    json.dump(dict(interval=[a, b], quality=q, seconds=round(time.time() - t0, 1)), open(os.path.join(d, "preproc_compare.json"), "w"), indent=1)
    print("interval", a, b, json.dumps(q))


def load(dataset="fantasmagorie", rep="binary_48x36_lines_tm3_v1"):
    z = np.load(os.path.join(DER, dataset, rep + ".npz")); return np.unpackbits(z["bits"])[:int(np.prod(z["shape"]))].reshape(z["shape"]), z["source_frames"]


def stats():
    """Redundancy and diversity for both videos at 48x36, plus the registry manifests."""
    import av
    os.makedirs(MAN, exist_ok=True); from exp45_c12 import target, SRC as BA_SRC
    BA = target("0:6572").astype(np.uint8); FA, fsrc = load(); FA96, _ = load(rep="binary_96x72_lines_tm3_v1"); FAraw, _ = load(rep="binary_48x36_lines_v1"); cmp_ = json.load(open(os.path.join(DER, "fantasmagorie", "preproc_compare.json")))
    r_ba, r_fa = redundancy(BA, 30.0), redundancy(FA, 25.0); r_fa12 = redundancy(FA[::2], 12.5); r_fa96 = redundancy(FA96, 25.0)
    with av.open(SRC) as c: s = c.streams.video[0]; cc = s.codec_context; fa_src = dict(codec=cc.name, width=cc.width, height=cc.height, time_base=str(s.time_base), rate=str(s.average_rate))
    with av.open(BA_SRC) as c: s = c.streams.video[0]; cc = s.codec_context; ba_src = dict(codec=cc.name, width=cc.width, height=cc.height, time_base=str(s.time_base), rate=str(s.average_rate))
    G = np.load(os.path.join(DER, "fantasmagorie", "gray_192x144.npy"))
    fa = dict(id="fantasmagorie", title="Fantasmagorie (1908), Emile Cohl", source=dict(page="https://commons.wikimedia.org/wiki/File:Fantasmagorie_(1908).webm",
              url="https://upload.wikimedia.org/wikipedia/commons/f/f2/Fantasmagorie_%281908%29.webm", commons_sha1="7ba4199a67aeb4ae8547a8fbb3d9e2bec3e44aa7",
              local_path="data/source/fantasmagorie/Fantasmagorie_(1908).webm", sha256=sha256(SRC), bytes=os.path.getsize(SRC), retrieved="2026-11-13", **fa_src,
              decoded_frames=int(len(G)), duration_s=98.2, pts_step_s=0.04, decode_errors=0, aspect="4:3", audio_streams=0,
              edition="Commons file credited 'Du Praxinoscope au Cellulo PAL DVD' (25 fps PAL transfer; frames 0..~740 are the DVD's modern title cards)"),
              rights=dict(commons_license="Public domain (LicenseShortName 'Public domain', Copyrighted False, AttributionRequired false; checked via the Commons API 2026-11-13)",
                          attribution="Fantasmagorie (1908) by Emile Cohl, Gaumont; file via Wikimedia Commons (from a PAL DVD edition)",
                          note="the DVD's added title cards are excluded from all derived data"),
              animation_interval=dict(source_frames=cmp_["interval"], rule="first/last frame with mean luma > 5 (192x144) after frame 700"),
              derived={"binary_48x36_lines_tm3_v1": dict(file="data/derived/fantasmagorie/binary_48x36_lines_tm3_v1.npz", preprocessing="lines + centred 3-frame temporal majority (SELECTED training target)", polarity="1 = drawn line", version=PREP_VERSION),
                       "binary_96x72_lines_tm3_v1": dict(file="data/derived/fantasmagorie/binary_96x72_lines_tm3_v1.npz", preprocessing="96x72 lines + 3-frame temporal majority", polarity="1 = drawn line", version=PREP_VERSION),
                       "binary_48x36_lines_v1": dict(file="data/derived/fantasmagorie/binary_48x36_lines_v1.npz", preprocessing=__doc__.split("lines    ")[1].split("Selected")[0].strip(), polarity="1 = drawn line (bright), 0 = background", version=PREP_VERSION),
                       "binary_96x72_lines_v1": dict(file="data/derived/fantasmagorie/binary_96x72_lines_v1.npz", preprocessing="same line mask, 2x2 pooling, >= 2 of 4", polarity="1 = line", version=PREP_VERSION),
                       "binary_48x36_simple_v1": dict(file="data/derived/fantasmagorie/binary_48x36_simple_v1.npz", preprocessing="box resize, gray >= 128 (Bad Apple recipe), kept as the rejected baseline", polarity="1 = bright")},
              preprocessing_comparison=cmp_["quality"], redundancy_48x36_25fps=r_fa, redundancy_48x36_12p5fps=r_fa12, redundancy_96x72_25fps=r_fa96, redundancy_48x36_unfiltered_25fps=redundancy(FAraw, 25.0))
    ba = dict(id="bad_apple", title="Bad Apple!! (shadow-art PV)", source=dict(local_path="refs/Bad-Apple-Starter/BadApple!!.mp4", sha256=sha256(BA_SRC), **ba_src, decoded_frames=6572,
              origin="local clone of C-mykk/Bad-Apple-Starter; upstream identity not verified (README)"),
              rights=dict(note="not redistributed; derived 48x36 frames are used for research comparison (README 'Data and licence')"),
              derived={"binary_48x36_v1": dict(loader="experiments/exp45_c12.target('0:6572') (= ca.pipeline prep width 48, threshold 128)", polarity="1 = white")},
              redundancy_48x36_30fps=r_ba, redundancy_48x36_15fps=redundancy(BA[::2], 15.0))
    for m in (fa, ba): json.dump(m, open(os.path.join(MAN, m["id"] + ".json"), "w"), indent=1)
    print(json.dumps(dict(bad_apple=r_ba, fantasmagorie=r_fa, fa12=r_fa12, fa96=r_fa96), indent=1))


def ba96():
    """Bad Apple at 96x72 with exactly the 48x36 recipe (ca.video.gray_frames box resize, gray >= 128); registered."""
    from ca.video import gray_frames, binarize
    from exp45_c12 import SRC as BA_SRC
    B = binarize(gray_frames(BA_SRC, 96)[0]); d = os.path.join(DER, "bad_apple"); os.makedirs(d, exist_ok=True)
    np.savez_compressed(os.path.join(d, "binary_96x72_v1.npz"), bits=np.packbits(B), shape=B.shape, source_frames=np.arange(len(B)))
    m = json.load(open(os.path.join(MAN, "bad_apple.json")))
    m["derived"]["binary_96x72_v1"] = dict(file="data/derived/bad_apple/binary_96x72_v1.npz", preprocessing="ca.video.gray_frames(width 96, PIL BOX) then gray >= 128 (same recipe as 48x36)",
                                           polarity="1 = white", note="derived from the copyrighted source: git-ignored, regenerate locally"); m["redundancy_96x72_30fps"] = redundancy(B, 30.0)
    json.dump(m, open(os.path.join(MAN, "bad_apple.json"), "w"), indent=1); print(B.shape, m["redundancy_96x72_30fps"]["mean_change"])


if __name__ == "__main__":
    dict(prep=prep, stats=stats, ba96=ba96)[sys.argv[1]]()
