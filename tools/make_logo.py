"""Bad CA!! logo and icons from ONE real frame of the source video (no generated imagery).
Source: refs/Bad-Apple-Starter/BadApple!!.mp4 (sha256 d8f21232...cd0, 480x360, 30 fps), decoded frame 310 (t = 10.333 s):
the girl's profile with an open mouth and the apple (with stem) at her lips, the moment before the bite. Chosen after
comparing frames 260-344 and the falling apple 370-460 (no visible bite mark there); see assets/logo/README.md.
Conversion: luma (PIL 'L') -> crop (no stretching) -> box downsample to a coarse grid -> threshold 128 -> nearest upscale.
Writes assets/logo/{bad-ca-logo.png, bad-ca-mark.png, favicon-16.png, favicon-32.png, icon-180.png, icon-512.png,
check_light_dark.png} and ui/static/favicon.png. usage: .venv/bin/python tools/make_logo.py"""
import hashlib, os
import av, numpy as np
from PIL import Image, ImageDraw, ImageFont
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); SRC = os.path.join(ROOT, "refs", "Bad-Apple-Starter", "BadApple!!.mp4")
SHA = "d8f21232313506da4f8d631ca289793bcf2f0a33440d68ccb02f4ac52e963cd0"; FRAME = 310
OUT = os.path.join(ROOT, "assets", "logo"); FONT = os.path.join(ROOT, "ui", "static", "fonts", "departure-mono.woff2")
SCENE = (110, 110, 310, 260)  # 200x150 (4:3): apple with stem at the lips, nose and mouth in profile
MARK = (158, 150, 290, 282)   # 132x132: apple (with stem) at the open mouth, for icons


def frame():
    h = hashlib.sha256(open(SRC, "rb").read()).hexdigest(); assert h == SHA, "unexpected source file"
    with av.open(SRC) as c:
        for i, f in enumerate(c.decode(video=0)):
            if i == FRAME: return f.to_image().convert("L")


def pixel(img, box, gw, gh, scale):
    g = img.crop(box).resize((gw, gh), Image.BOX); b = (np.asarray(g) >= 128).astype(np.uint8) * 255
    return Image.fromarray(b).resize((gw * scale, gh * scale), Image.NEAREST)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); im = frame()
    scene = pixel(im, SCENE, 80, 60, 4)                                                  # 80x60 cells (2.5 source px each), shown 320x240
    try: f = ImageFont.truetype(FONT, 64)
    except OSError: f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 60)
    W = scene.width + 40 + 360; logo = Image.new("RGBA", (W, scene.height + 8), (0, 0, 0, 0)); d = ImageDraw.Draw(logo)
    d.rectangle((0, 0, scene.width + 7, scene.height + 7), fill=(255, 255, 255, 255), outline=(0, 0, 0, 255), width=4)
    logo.paste(scene.convert("RGBA"), (4, 4)); d.text((scene.width + 40, scene.height // 2 - 40), "Bad CA!!", font=f, fill=(128, 128, 128, 255))
    logo.save(os.path.join(OUT, "bad-ca-logo.png")); scene.save(os.path.join(OUT, "bad-ca-mark.png"))
    for n in (16, 32, 180, 512):
        g = 16 if n <= 32 else 32; ic = pixel(im, MARK, g, g, 1).resize((n, n), Image.NEAREST).convert("RGB")
        ImageDraw.Draw(ic).rectangle((0, 0, n - 1, n - 1), outline=(0, 0, 0)); ic.save(os.path.join(OUT, f"{'favicon' if n <= 32 else 'icon'}-{n}.png"))
    Image.open(os.path.join(OUT, "favicon-32.png")).save(os.path.join(ROOT, "ui", "static", "favicon.png"))
    chk = Image.new("RGB", (2 * (W + 40), scene.height + 160), (255, 255, 255)); dc = ImageDraw.Draw(chk); dc.rectangle((W + 40, 0, 2 * (W + 40), chk.height), fill=(13, 17, 23))
    for k, bg in enumerate(((255, 255, 255), (13, 17, 23))):
        x0 = k * (W + 40) + 20; tile = Image.new("RGB", logo.size, bg); tile.paste(logo, (0, 0), logo); chk.paste(tile, (x0, 20)); x = x0
        for n in (16, 32, 180):
            ic = Image.open(os.path.join(OUT, f"{'favicon' if n <= 32 else 'icon'}-{n}.png")).resize((min(n, 120),) * 2, Image.NEAREST); chk.paste(ic, (x, scene.height + 30)); x += ic.width + 20
    chk.save(os.path.join(OUT, "check_light_dark.png")); print("ok", logo.size)
