"""Exp53 (Campaign 13): README demo video rendered from an ncab-v2 EXPORT replay (numpy reference, first frame + stored
initial hidden only). Layout: left = CA output (raw binary s0 > 0.5, nearest x10); right = the update rule actually executed,
fixed structure in BLUE, the current phase's parameter tensors in RED (with ids, shapes and real statistics of the rule in
effect: L2 norms and the change vs the previous phase); header = phase i/K, frame range, seconds, rule id; bottom = whole
timeline with every phase boundary and the playhead. Time stamps are simulation time (frame / fps), not wall time.
At a boundary the 'changed' line is highlighted for 15 frames: the switch itself is instantaneous (hard switch).
usage: exp53_demo_video.py EXPORT OUT_STEM [--compare] [--range a:b] [--fps 30]
  --compare adds the binary original as a small inset (separate version); --range renders a cut of the full replay."""
import json, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca import phase_nca
from ca.video_out import verify_mp4, write_mp4

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"; FB = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
BG, FG, DIM, BLUE, RED, AMB = (14, 14, 16), (232, 228, 218), (130, 130, 138), (110, 168, 255), (255, 96, 86), (255, 176, 0)
W_IMG, H_IMG = 1280, 640


def font(sz, bold=False): return ImageFont.truetype(FB if bold else FONT, sz)


_orig_text = ImageDraw.ImageDraw.text
def _checked_text(self, xy, text, *a, **kw):   # every label must fit inside the frame (the C12 videos had overlapping text)
    if xy[0] + self.textlength(text, font=kw.get("font")) > W_IMG - 8: raise ValueError(f"text overflow: {text}")
    return _orig_text(self, xy, text, *a, **kw)
ImageDraw.ImageDraw.text = _checked_text


def stats(E, K):
    out = []
    for k in range(K):
        d = dict(Wp=float(np.linalg.norm(E["Wp"][k])), bp=float(np.linalg.norm(E["bp"][k])), Wu=float(np.linalg.norm(E["Wu"][k])), bu=float(np.linalg.norm(E["bu"][k])))
        if k:
            dv = np.concatenate([(E[n][k] - E[n][k - 1]).ravel() for n in phase_nca.PNAMES])
            d.update(changed=int((np.abs(dv) > 1e-6).sum()), dl2=float(np.linalg.norm(dv)), dmax=float(np.abs(dv).max()))
        out.append(d)
    return out


def right_panel(dr, x0, hd, k, ph, st, recent):
    y = 70; f, fs, fb = font(15), font(14), font(16, True)   # fits rule ids up to 3 digits (overflow guard)
    dr.text((x0, y), "rule in effect", font=fs, fill=DIM); y += 28
    lines = [  # (segments) each segment (text, colour)
        [("every cell at once · 4 substeps per video frame · zero padding", BLUE)],
        [("p_f = ReLU( Σ_c Σ_(dy,dx) ", BLUE), (f"Wp⁽{k}⁾[f,c,dy,dx]", RED), (" · s_c(y+dy,x+dx) + ", BLUE), (f"bp⁽{k}⁾[f]", RED), (" )", BLUE)],
        [("      f = 0..63, c = 0..15, dy,dx ∈ {-1,0,1}", BLUE)],
        [("s_c ← s_c + Σ_f ", BLUE), (f"Wu⁽{k}⁾[c,f]", RED), (" · p_f + ", BLUE), (f"bu⁽{k}⁾[c]", RED), ("   (16 channels)", BLUE)],
        [("shown pixel = 1 if s_0 > 0.5   (s_0 is a state value, not a probability)", BLUE)]]
    for segs in lines:
        x = x0
        for t, c in segs:
            dr.text((x, y), t, font=f, fill=c); x += dr.textlength(t, font=f)
        if x > W_IMG - 12: raise ValueError(f"text overflow ({x:.0f}px): {''.join(t for t, _ in segs)}")
        y += 30
    y += 8; dr.text((x0, y), "blue = fixed structure (all phases)   red = parameters of this phase's rule", font=fs, fill=DIM); y += 34
    s = st[k]
    for name, shape, n in (("Wp", "64×16×3×3", 9216), ("bp", "64", 64), ("Wu", "16×64", 1024), ("bu", "16", 16)):
        dr.text((x0, y), f"{name}⁽{k}⁾", font=fb, fill=RED); dr.text((x0 + 90, y), f"{shape:>10}  {n:>5} values   ‖·‖₂ = {s[name]:8.3f}", font=f, fill=FG); y += 27
    y += 6; dr.text((x0, y), "10,320 float32 parameters in this rule (independent rule per phase)", font=fs, fill=DIM); y += 30
    if k:
        txt = f"vs phase {k - 1}: {s['changed']:,} of 10,320 values changed · ‖Δ‖₂ {s['dl2']:.2f} · max |Δ| {s['dmax']:.3f}"
        if recent: dr.rectangle((x0 - 6, y - 4, x0 + dr.textlength(txt, font=fs) + 6, y + 22), outline=AMB)
        dr.text((x0, y), txt, font=fs, fill=AMB if recent else FG)
        if recent: dr.text((x0, y + 26), "hard switch at the boundary frame (no interpolation)", font=fs, fill=AMB)
    else:
        dr.text((x0, y), "first phase (frame 1 onward); initial hidden = stored low-res array" if hd["init"]["kind"] == "lowres" else "first phase", font=fs, fill=FG)


def render(path, stem, compare=False, rng=None, fps=30):
    hd, E, sched, h0, f0 = phase_nca.load(path); T, K = hd["T"], hd["K"]; B, _ = phase_nca.rollout_np(E, f0, sched, hd["S"], h0)
    F = None
    if compare:
        from exp45_c12 import target
        F = target(f"{hd['meta']['src_start']}:{T}")
    st = stats(E, K); bounds = [p["start"] for p in hd["schedule"]]; a, b = rng or (0, T)
    base = Image.new("RGB", (W_IMG, H_IMG), BG); bd = ImageDraw.Draw(base); tl_y, tl_h = 590, 18
    bd.rectangle((20, tl_y, W_IMG - 20, tl_y + tl_h), outline=DIM)
    for s0 in bounds[1:]: x = 20 + (W_IMG - 40) * s0 / T; bd.line((x, tl_y, x, tl_y + tl_h), fill=BLUE)
    bd.text((20, tl_y + 22), f"0 s", font=font(13), fill=DIM); bd.text((W_IMG - 80, tl_y + 22), f"{(T - 1) / fps:.1f} s", font=font(13), fill=DIM)
    bd.text((20, 14), "Bad CA!! · autonomous replay of the export (first frame + stored initial hidden, nothing injected)", font=font(15), fill=DIM)
    import av
    from fractions import Fraction
    os.makedirs(os.path.dirname(stem), exist_ok=True); cont = av.open(stem + ".mp4", "w"); vs = cont.add_stream("libx264", rate=Fraction(fps).limit_denominator(1000))
    vs.width, vs.height, vs.pix_fmt = W_IMG, H_IMG, "yuv420p"; vs.options = {"crf": "16", "preset": "medium"}; n = 0; poster_t = a + (b - a) // 3
    cache = {}   # frames are streamed to the encoder (a whole-song list of frames would not fit in memory)
    for t in range(a, b):
        k = int(sched[t]) if t else int(sched[1]); ph = hd["schedule"][[i for i, p in enumerate(hd["schedule"]) if p["start"] <= max(t, 1) < p["end"]][0]]
        pi = hd["schedule"].index(ph); recent = t > 1 and 0 <= t - ph["start"] < 15 and pi > 0
        key = (k, recent)
        if key not in cache:
            im = base.copy(); right_panel(ImageDraw.Draw(im), 520, hd, k, ph, st, recent); cache[key] = im
        im = cache[key].copy(); dr = ImageDraw.Draw(im)
        dr.text((520, 40), f"phase {pi + 1} of {len(hd['schedule'])}  ·  frames {ph['start']}–{ph['end'] - 1}  ·  {ph['start'] / fps:.2f}–{(ph['end'] - 1) / fps:.2f} s  ·  rule id {k}", font=font(17, True), fill=FG)
        ca = Image.fromarray((B[t] * 255).astype(np.uint8)).resize((480, 360), Image.NEAREST).convert("RGB"); im.paste(ca, (20, 70))
        dr.rectangle((19, 69, 500, 430), outline=DIM)
        dr.text((20, 440), f"frame {t:4d} / {T - 1}   t = {t / fps:6.2f} s (simulation time)", font=font(16), fill=FG)
        dr.text((20, 466), f"{t * hd['S']:6d} CA substeps since the first frame", font=font(14), fill=DIM)
        if F is not None:
            dr.text((20, 494), "original (binary target), comparison only:", font=font(13), fill=DIM)
            im.paste(Image.fromarray((F[t] * 255).astype(np.uint8)).resize((96, 72), Image.NEAREST).convert("RGB"), (20, 512))
        x = 20 + (W_IMG - 40) * t / T; dr.rectangle((x - 1, tl_y - 4, x + 1, tl_y + tl_h + 4), fill=AMB)
        for pkt in vs.encode(av.VideoFrame.from_ndarray(np.asarray(im), format="rgb24")): cont.mux(pkt)
        if t == poster_t: im.save(stem + "_poster.png")
        n += 1
    for pkt in vs.encode(): cont.mux(pkt)
    cont.close(); chk = verify_mp4(stem + ".mp4", n, fps)
    json.dump(dict(export=os.path.relpath(path, ROOT), frames=[a, b - 1], check=chk), open(stem + ".video_check.json", "w"), indent=1)
    return chk


if __name__ == "__main__":
    args = sys.argv[1:]; opt = lambda k: args[args.index(k) + 1] if k in args else None
    r = tuple(map(int, opt("--range").split(":"))) if opt("--range") else None
    print(render(os.path.join(ROOT, args[0]), os.path.join(ROOT, args[1]), "--compare" in args, r, float(opt("--fps") or 30)))
