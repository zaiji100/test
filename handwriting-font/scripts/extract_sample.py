"""Extract the real handwritten glyphs from the sample photo.

Photo -> crop -> background flattening -> deskew -> threshold -> per-character
polygon masks -> canonical glyph PNGs in glyphs/ (black ink on white).

Canonical glyph PNG: a square canvas covering font units
X in [-400, 1400], Y in [-520, 1280]  (the 1000-unit em box, -120..880, is centred).
"""
import argparse, json, os
import cv2
import numpy as np
from hwlib import CANVAS_X0, CANVAS_YTOP, CANVAS_SIZE

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# crop of the original 1080x1919 photo that holds the handwriting
CROP = (300, 850, 700, 1150)          # x0, y0, x1, y1
UPSCALE = 4
DESKEW_DEG = -36.0                    # the line of text rises ~36 deg in the photo
INK_THRESHOLD = 0.80                  # ink = pixel darker than 80% of local paper
# per-character polygons in deskewed "ink" coordinates (see debug/ink_grid.png)
TEXT = "别嫌弃麻烦"
POLYS = {
    "别": [(0, 0), (358, 0), (358, 300), (340, 330), (318, 420), (0, 420)],
    "嫌": [(358, 0), (665, 0), (665, 650), (345, 650), (345, 420), (340, 330), (358, 300)],
    "弃": [(655, 0), (860, 0), (860, 500), (655, 500)],
    "麻": [(860, 0), (1108, 0), (1108, 500), (820, 500), (820, 330), (860, 250)],
    "烦": [(1108, 0), (1399, 0), (1399, 500), (1108, 500)],
}
# vertical placement: ink-row LINE_TOP..LINE_BOTTOM maps to font units 830..-70
LINE_TOP, LINE_BOTTOM = 20, 445
OUT_PPU = 0.5                          # pixels per font unit in the saved PNGs


def deskew(path):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x0, y0, x1, y1 = CROP
    c = im[y0:y1, x0:x1].astype(np.float32)
    c = cv2.resize(c, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)
    norm = c / cv2.GaussianBlur(c, (0, 0), 40)          # flatten uneven lighting
    h, w = norm.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), DESKEW_DEG, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    r = cv2.warpAffine(norm, M, (nw, nh), flags=cv2.INTER_CUBIC, borderValue=1.0)
    ink = (r < INK_THRESHOLD).astype(np.uint8)
    ink = drop_small(ink, 150)
    ys, xs = np.nonzero(ink)
    return ink[ys.min() - 20:ys.max() + 20, xs.min() - 20:xs.max() + 20]


def drop_small(b, min_area):
    n, lab, st, _ = cv2.connectedComponentsWithStats(b, 8)
    keep = np.zeros_like(b)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= min_area:
            keep[lab == i] = 1
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo", default=os.path.join(ROOT, "sample", "sample.jpg"))
    ap.add_argument("--out", default=os.path.join(ROOT, "glyphs"))
    ap.add_argument("--debug", default=os.path.join(ROOT, "build", "debug"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.debug, exist_ok=True)

    ink = deskew(a.photo)
    grid = cv2.cvtColor((1 - ink) * 255, cv2.COLOR_GRAY2BGR)
    for x in range(0, grid.shape[1], 50):
        cv2.line(grid, (x, 0), (x, grid.shape[0]), (255, 170, 170), 1)
        cv2.putText(grid, str(x), (x + 2, 14), 0, 0.4, (255, 0, 0), 1)
    for ch, p in POLYS.items():
        cv2.polylines(grid, [np.array(p, np.int32)], True, (0, 0, 255), 2)
    cv2.imwrite(os.path.join(a.debug, "ink_grid.png"), grid)

    src_ppu = (LINE_BOTTOM - LINE_TOP) / 900.0           # ink pixels per font unit
    cx0, cytop, csize = CANVAS_X0, CANVAS_YTOP, CANVAS_SIZE
    N = int(round(csize * OUT_PPU))
    meta = {}
    for ch in TEXT:
        m = np.zeros_like(ink)
        cv2.fillPoly(m, [np.array(POLYS[ch], np.int32)], 1)
        g = drop_small(ink * m, 400)
        ys, xs = np.nonzero(g)
        # font-unit position of the glyph: centred horizontally on X=500,
        # vertical position taken from the line of writing
        bx_mid = (xs.min() + xs.max()) / 2.0
        # affine: ink (col,row) -> canvas pixel
        s = OUT_PPU / src_ppu
        X_of_col = lambda col: 500 + (col - bx_mid) / src_ppu
        Y_of_row = lambda row: 830 - (row - LINE_TOP) / src_ppu
        A = np.array([[s, 0, (X_of_col(0) - cx0) * OUT_PPU],
                      [0, s, (cytop - Y_of_row(0)) * OUT_PPU]], np.float32)
        canvas = cv2.warpAffine(g.astype(np.float32), A, (N, N), flags=cv2.INTER_AREA)
        canvas = (canvas > 0.45).astype(np.uint8)
        # close pin-holes, remove jpeg crumbs
        canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
        canvas = drop_small(canvas, 12)
        name = f"u{ord(ch):04X}.png"
        cv2.imwrite(os.path.join(a.out, name), (1 - canvas) * 255)
        meta[name] = {"char": ch, "source": "sample/sample.jpg"}
        print("wrote", name, ch)
    with open(os.path.join(a.out, "sources.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
