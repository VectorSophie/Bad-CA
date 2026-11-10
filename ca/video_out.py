"""Labelled side-by-side comparison videos (MP4 via PyAV libx264 + GIF), with decode-back verification.

Panels are binary frame stacks (T,H,W) shown at the same frame index and the same integer nearest-
neighbour scale; a 'diff' panel colours false negatives (target white, replay black) red and false
positives (target black, replay white) blue. Frames are never interpolated, edited or re-timed: output
frame i shows input frame i of every panel. Text: panel title + per-frame line (source frame number,
time) + footer.
"""
import json
from fractions import Fraction
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


def _font(sz):
    try: return ImageFont.truetype(FONT, sz)
    except OSError: return ImageFont.load_default()


def diff_rgb(P, F):
    f, p = F.astype(bool), P.astype(bool)
    d = np.zeros(F.shape + (3,), np.uint8); d[f & p] = 255
    d[f & ~p] = (235, 50, 50); d[~f & p] = (60, 110, 255)
    return d


def compose(panels, footer, frame_label, scale=6):
    """panels: list of (title, (T,H,W) uint8 binary | (T,H,W,3) rgb). frame_label(t) -> str. Returns list of RGB arrays."""
    T = len(panels[0][1]); H, W = panels[0][1].shape[1:3]; pw, ph = W * scale, H * scale
    ft, fs = _font(min(16, max(12, scale * 2 + 2))), _font(min(14, max(11, scale * 2)))   # capped: x12 zooms overflowed
    head, foot, gap = ft.size + fs.size + 14, (fs.size + 6) * len(footer) + 8, 8
    Wt = len(panels) * pw + (len(panels) + 1) * gap; Ht = head + ph + foot
    Wt += Wt % 2; Ht += Ht % 2                                             # yuv420p needs even sizes
    out = []
    for t in range(T):
        img = Image.new("RGB", (Wt, Ht), (24, 24, 28)); dr = ImageDraw.Draw(img)
        for i, (title, X) in enumerate(panels):
            x0 = gap + i * (pw + gap)
            a = X[t]; rgb = a if a.ndim == 3 else np.repeat((a.astype(np.uint8) * 255)[..., None], 3, -1)
            img.paste(Image.fromarray(rgb).resize((pw, ph), Image.NEAREST), (x0, head))
            dr.text((x0, 4), title, fill=(240, 240, 240), font=ft)
            dr.text((x0, 8 + ft.size), frame_label(t), fill=(170, 200, 255), font=fs)
        for j, line in enumerate(footer):
            dr.text((gap, head + ph + 4 + j * (fs.size + 6)), line, fill=(200, 200, 200), font=fs)
        out.append(np.asarray(img))
    return out


def write_mp4(frames, path, fps):
    import av
    with av.open(path, "w") as c:
        s = c.add_stream("libx264", rate=Fraction(fps).limit_denominator(1000)); s.width, s.height = frames[0].shape[1], frames[0].shape[0]
        s.pix_fmt = "yuv420p"; s.options = {"crf": "12", "preset": "medium"}
        for f in frames:
            for pkt in s.encode(av.VideoFrame.from_ndarray(f, format="rgb24")): c.mux(pkt)
        for pkt in s.encode(): c.mux(pkt)


def write_gif(frames, path, fps):
    ims = [Image.fromarray(f) for f in frames]
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=int(round(1000 / fps)), loop=0)


def verify_mp4(path, n_expected, fps_expected):
    import av
    with av.open(path) as c:
        st = c.streams.video[0]; n = sum(1 for _ in c.decode(st))
        dur = float(st.duration * st.time_base) if st.duration else None
        rate = float(st.average_rate) if st.average_rate else None
        codec = st.codec_context.name; wh = (st.codec_context.width, st.codec_context.height)
    return dict(path=path, decoded_frames=n, expected_frames=n_expected, fps=rate, expected_fps=fps_expected, duration_s=dur,
                expected_duration_s=round(n_expected / fps_expected, 3), codec=codec, size=wh,
                ok=(n == n_expected and rate is not None and abs(rate - fps_expected) < 1e-6))


def verify_gif(path, n_expected):
    im = Image.open(path); n = 0
    try:
        while True: n += 1; im.seek(im.tell() + 1)
    except EOFError: pass
    return dict(path=path, decoded_frames=n, expected_frames=n_expected, frame_ms=im.info.get("duration"), ok=n == n_expected)


def make_video(panels, footer, frame_label, stem, fps_list=((30, "1x"), (7.5, "0.25x")), scale=6, gif=True):
    frames = compose(panels, footer, frame_label, scale); checks = []
    for fps, tag in fps_list:
        mp4, gifp = f"{stem}_{tag}.mp4", f"{stem}_{tag}.gif"
        write_mp4(frames, mp4, fps); checks.append(verify_mp4(mp4, len(frames), fps))
        if gif: write_gif(frames, gifp, fps); checks.append(verify_gif(gifp, len(frames)))   # gif=False for very long videos
    json.dump(checks, open(stem + ".video_check.json", "w"), indent=1)
    return checks
