"""Render preview images of a built font.

  python3 scripts/preview.py --font dist/BieXianQiHand-Regular.ttf --out previews/text.png
  python3 scripts/preview.py --compare --font build/test/x.ttf --out previews/compare.png
"""
import argparse, glob, os, sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import hwlib  # noqa: E402

LABEL_FONT = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
DEFAULT_TEXT = [
    "别嫌弃麻烦",
    "天地玄黄，宇宙洪荒。日月盈昃，辰宿列张。",
    "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。",
    "我手写我心，一笔一划都是自己的样子。",
    "The quick brown fox jumps over the lazy dog. 0123456789",
]


def label_font(size):
    try:
        return ImageFont.truetype(LABEL_FONT, size)
    except OSError:
        return ImageFont.load_default()


def render_lines(font_path, lines, size=72, width=1800, pad=40):
    f = ImageFont.truetype(font_path, size)
    lh = int(size * 1.45)
    im = Image.new("L", (width, pad * 2 + lh * len(lines)), 255)
    d = ImageDraw.Draw(im)
    for i, t in enumerate(lines):
        d.text((pad, pad + i * lh + int(size * 0.2)), t, font=f, fill=20)
    return im


def real_row(folder, text, cell=200):
    """Show the real glyph PNGs at the same scale a font would draw them."""
    im = Image.new("L", (cell * len(text) + 80, int(cell * 1.5)), 255)
    for i, ch in enumerate(text):
        p = os.path.join(folder, f"u{ord(ch):04X}.png")
        if not os.path.exists(p):
            continue
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        s = cell / 1000 * hwlib.CANVAS_SIZE / g.shape[1]
        g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        ox = 40 + i * cell + int(hwlib.CANVAS_X0 / 1000 * cell)
        oy = int(cell * 0.25) - int((hwlib.CANVAS_YTOP - 880) / 1000 * cell)
        tile = Image.fromarray(g)
        im.paste(Image.composite(Image.new("L", tile.size, 20), Image.new("L", tile.size, 255),
                                 tile.point(lambda v: 255 - v)), (ox, oy), tile.point(lambda v: 255 - v))
    return im


def stack(rows, labels, width):
    lf = label_font(30)
    h = sum(r.height + 50 for r in rows) + 20
    out = Image.new("L", (width, h), 255)
    d = ImageDraw.Draw(out)
    y = 10
    for r, l in zip(rows, labels):
        d.text((20, y), l, font=lf, fill=110)
        out.paste(r, (0, y + 40))
        y += r.height + 50
        d.line([(20, y - 6), (width - 20, y - 6)], fill=215, width=2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--font", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", action="store_true", help="real sample vs generated vs base")
    ap.add_argument("--base", default=os.path.join(ROOT, "build", "LongCang-Regular.ttf"))
    ap.add_argument("--glyphs", default=os.path.join(ROOT, "glyphs"))
    ap.add_argument("--text", nargs="*")
    ap.add_argument("--size", type=int, default=72)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if a.compare:
        t = "别嫌弃麻烦"
        rows = [real_row(a.glyphs, t), render_lines(a.font, [t], 200, 1100, 40),
                render_lines(a.base, [t], 200, 1100, 40)]
        im = stack(rows, ["① 照片中的真实手写（矢量化前）", "② 按你的风格生成的字形", "③ 原始骨架字体 Long Cang"], 1100)
    else:
        lines = a.text or DEFAULT_TEXT
        im = render_lines(a.font, lines, a.size)
    im.save(a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
