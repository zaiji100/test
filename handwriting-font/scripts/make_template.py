"""Generate printable A4 handwriting templates (PDF + PNG + JSON).

Print, write one character per box with your usual pen, photograph or scan
each page (flat, whole page visible incl. the 4 corner markers), then run
ingest_sheets.py.

  python3 scripts/make_template.py --count 468            # 468 most common chars, 3 pages
  python3 scripts/make_template.py --chars "我的名字"       # any characters you like
"""
import argparse, json, os, sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import sheet_layout as L  # noqa: E402

DPI = 300
PX = DPI / 25.4
HINT_FONTS = [
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    os.path.join(ROOT, "build", "LongCang-Regular.ttf"),
]


def find_font(path):
    for p in ([path] if path else []) + HINT_FONTS:
        if p and os.path.exists(p):
            return p
    raise SystemExit("no CJK font found for the hint characters; pass --hint-font")


def mm(v):
    return int(round(v * PX))


def draw_page(chars, page, npages, font_path, title):
    W, H = mm(L.PAGE_W), mm(L.PAGE_H)
    im = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(im)
    aruco = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    for mid in list(L.MARKER_IDS) + [L.PAGE_ID_BASE + page]:
        (x, y) = L.marker_corners_mm(mid)[0]
        m = cv2.aruco.generateImageMarker(aruco, mid, mm(L.MARKER))
        im.paste(Image.fromarray(m), (mm(x), mm(y)))
    f_head = ImageFont.truetype(font_path, mm(3.6))
    f_note = ImageFont.truetype(font_path, mm(2.4))
    f_hint = ImageFont.truetype(font_path, mm(3.0))
    d.text((mm(24), mm(9)), f"{title}   第 {page + 1}/{npages} 页", font=f_head, fill=60)
    d.text((mm(24), mm(15)), "用平时写字的笔，在方框里一格写一个字；可以写得随意，但不要压到别的格子。",
           font=f_note, fill=90)
    d.text((mm(24), mm(19)), "写完后整页平铺拍照/扫描（四角黑色标记都要拍进去），再运行 ingest_sheets.py。",
           font=f_note, fill=90)
    for i, ch in enumerate(chars):
        x, y, s = L.cell_box(i)
        d.text((mm(x), mm(y - L.LABEL_H + 0.4)), ch, font=f_hint, fill=120)
        d.text((mm(x + 4.2), mm(y - L.LABEL_H + 1.0)), f"{ord(ch):04X}", font=f_note, fill=170)
        # very light guides: box + centre cross (dropped by the ink threshold)
        d.rectangle([mm(x), mm(y), mm(x + s), mm(y + s)], outline=185, width=2)
        cx, cy = mm(x + s / 2), mm(y + s / 2)
        for k in range(mm(x) + 6, mm(x + s) - 6, 16):
            d.line([(k, cy), (k + 7, cy)], fill=225, width=1)
        for k in range(mm(y) + 6, mm(y + s) - 6, 16):
            d.line([(cx, k), (cx, k + 7)], fill=225, width=1)
    return im


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chars", help="characters to put on the sheets")
    ap.add_argument("--chars-file", default=os.path.join(ROOT, "charsets", "common3500.txt"))
    ap.add_argument("--start", type=int, default=0, help="offset into --chars-file")
    ap.add_argument("--count", type=int, default=L.PER_PAGE * 3)
    ap.add_argument("--glyphs", default=os.path.join(ROOT, "glyphs"))
    ap.add_argument("--include-existing", action="store_true",
                    help="also list characters you already have real glyphs for")
    ap.add_argument("--name", help="output name (default derived from the selection)")
    ap.add_argument("--out", default=os.path.join(ROOT, "templates"))
    ap.add_argument("--hint-font")
    a = ap.parse_args()

    if a.chars:
        pool = a.chars
    else:
        pool = open(a.chars_file, encoding="utf-8").read().strip()[a.start:]
    seen, chars = set(), []
    for ch in pool:
        if ch.isspace() or ch in seen:
            continue
        seen.add(ch)
        if not a.include_existing and os.path.exists(os.path.join(a.glyphs, f"u{ord(ch):04X}.png")):
            continue
        chars.append(ch)
        if not a.chars and len(chars) >= a.count:
            break
    if not chars:
        raise SystemExit("nothing to write")
    pages = [chars[i:i + L.PER_PAGE] for i in range(0, len(chars), L.PER_PAGE)]
    name = a.name or (f"common-{a.start + 1}-{a.start + len(chars)}" if not a.chars else f"custom-{len(chars)}")
    os.makedirs(a.out, exist_ok=True)
    font_path = find_font(a.hint_font)
    imgs = []
    for p, cs in enumerate(pages):
        im = draw_page(cs, p, len(pages), font_path, f"手写字体采集模板 · {name}")
        im.save(os.path.join(a.out, f"{name}-p{p + 1}.png"), dpi=(DPI, DPI))
        imgs.append(im.convert("RGB"))
    pdf = os.path.join(a.out, f"{name}.pdf")
    imgs[0].save(pdf, save_all=True, append_images=imgs[1:], resolution=DPI)
    meta = {"name": name, "layout": "a4-12x13-v1", "pages": ["".join(cs) for cs in pages]}
    with open(os.path.join(a.out, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"{len(chars)} characters on {len(pages)} pages -> {pdf}")


if __name__ == "__main__":
    main()
