"""Read photos/scans of filled-in templates into glyphs/uXXXX.png.

  python3 scripts/ingest_sheets.py --template templates/common-1-468.json photos/*.jpg
  python3 scripts/build_font.py          # rebuild: your glyphs replace generated ones

Each page is located by its ArUco corner markers (perspective is corrected),
lighting is flattened, every box is cut out, printed guides are dropped by the
ink threshold, and the box is mapped onto the font's em square.
"""
import argparse, glob, json, os, sys
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import hwlib  # noqa: E402
import sheet_layout as L  # noqa: E402

R = 16.0  # px per mm of the rectified page


def locate_page(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250),
                                  cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None:
        return None, None, 0
    src, dst, page = [], [], None
    for c, i in zip(corners, ids.reshape(-1)):
        i = int(i)
        if i in L.MARKER_IDS or i >= L.PAGE_ID_BASE:
            if i >= L.PAGE_ID_BASE:
                page = i - L.PAGE_ID_BASE
            src += c.reshape(4, 2).tolist()
            dst += L.marker_corners_mm(i)
    if len(src) < 8:
        return None, page, len(src) // 4
    Hm, _ = cv2.findHomography(np.float32(dst), np.float32(src), cv2.RANSAC, 4.0)
    return Hm, page, len(src) // 4


def rectify(img, Hm):
    """Rectified page (float, 1 = paper, 0 = ink) at R px/mm."""
    M = Hm @ np.diag([1 / R, 1 / R, 1.0])
    W, H = int(L.PAGE_W * R), int(L.PAGE_H * R)
    page = cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_REPLICATE)
    dark = page.min(axis=2).astype(np.float32)           # catches blue/black/any dark ink
    k = int(R * 2.5) | 1
    bg = cv2.dilate(dark, np.ones((k, k), np.uint8))       # remove ink -> paper level
    bg = cv2.GaussianBlur(bg, (0, 0), R * 3)
    return np.clip(dark / np.maximum(bg, 1), 0, 1)


def cell_glyph(norm, i, thr, inset_mm, ppu):
    """Box i of the rectified page -> canonical canvas (uint8 ink mask) or None."""
    x, y, s = L.cell_box(i)
    N = int(round(hwlib.CANVAS_SIZE * ppu))
    # canonical px -> font units -> page mm -> rectified px
    a = (1 / ppu) * s / 1000 * R                         # scale
    tx = (x + (hwlib.CANVAS_X0 + 0.5 / ppu) / 1000 * s) * R
    ty = (y + (880 - hwlib.CANVAS_YTOP + 0.5 / ppu) / 1000 * s) * R
    A = np.float32([[a, 0, tx], [0, a, ty]])
    g = cv2.warpAffine(norm, A, (N, N), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=1.0)
    # keep only the inside of the box
    u0 = (inset_mm / s * 1000 - hwlib.CANVAS_X0) * ppu
    u1 = ((s - inset_mm) / s * 1000 - hwlib.CANVAS_X0) * ppu
    v0 = (hwlib.CANVAS_YTOP - 880 + inset_mm / s * 1000) * ppu
    v1 = (hwlib.CANVAS_YTOP - 880 + (s - inset_mm) / s * 1000) * ppu
    mask = np.zeros((N, N), np.uint8)
    mask[int(v0):int(v1), int(u0):int(u1)] = 1
    ink = ((g < thr) & (mask > 0)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, 8)
    keep = np.zeros_like(ink)
    for j in range(1, n):
        if st[j, cv2.CC_STAT_AREA] >= 25:
            keep[lab == j] = 1
    if keep.sum() < 150:
        return None
    return keep


def fit_page(glyphs, target_h, ppu):
    """Scale all glyphs of a page by one factor so their median height ~ target."""
    hs = []
    for g in glyphs.values():
        ys = np.nonzero(g.any(axis=1))[0]
        hs.append((ys.max() - ys.min()) / ppu)
    if not hs:
        return glyphs, 1.0
    f = float(np.clip(target_h / np.median(hs), 0.8, 1.7))
    N = next(iter(glyphs.values())).shape[0]
    cx = (500 - hwlib.CANVAS_X0) * ppu
    cy = (hwlib.CANVAS_YTOP - 380) * ppu
    A = np.float32([[f, 0, cx - f * cx], [0, f, cy - f * cy]])
    out = {k: (cv2.warpAffine(g.astype(np.float32), A, (N, N), flags=cv2.INTER_LINEAR) > 0.5).astype(np.uint8)
           for k, g in glyphs.items()}
    return out, f


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photos", nargs="+")
    ap.add_argument("--template", required=True, help="the .json written by make_template.py")
    ap.add_argument("--glyphs", default=os.path.join(ROOT, "glyphs"))
    ap.add_argument("--threshold", type=float, default=0.62, help="ink if darker than this x paper (raise for pencil)")
    ap.add_argument("--inset", type=float, default=0.55, help="mm trimmed inside each box border")
    ap.add_argument("--fit-height", type=float, default=800, help="median glyph height per page (0 = as written)")
    ap.add_argument("--page", type=int, help="page number (1-based) if the page marker is not visible")
    ap.add_argument("--debug", default=os.path.join(ROOT, "build", "debug"))
    a = ap.parse_args()

    tpl = json.load(open(a.template, encoding="utf-8"))
    os.makedirs(a.glyphs, exist_ok=True)
    os.makedirs(a.debug, exist_ok=True)
    src_path = os.path.join(a.glyphs, "sources.json")
    sources = json.load(open(src_path, encoding="utf-8")) if os.path.exists(src_path) else {}
    ppu = 0.5
    photos = [p for pat in a.photos for p in sorted(glob.glob(pat))]
    total = 0
    for path in photos:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"! {path}: cannot read")
            continue
        Hm, page, nmark = locate_page(img)
        if a.page:
            page = a.page - 1
        if Hm is None or page is None:
            print(f"! {path}: found {nmark} markers, page={page}. Photograph the whole page incl. all corner markers.")
            continue
        if page >= len(tpl["pages"]):
            print(f"! {path}: page {page + 1} is not in {a.template}")
            continue
        chars = tpl["pages"][page]
        norm = rectify(img, Hm)
        glyphs = {}
        for i, ch in enumerate(chars):
            g = cell_glyph(norm, i, a.threshold, a.inset, ppu)
            if g is not None:
                glyphs[ch] = g
        f = 1.0
        if a.fit_height > 0:
            glyphs, f = fit_page(glyphs, a.fit_height, ppu)
        for ch, g in glyphs.items():
            name = f"u{ord(ch):04X}.png"
            cv2.imwrite(os.path.join(a.glyphs, name), (1 - g) * 255)
            sources[name] = {"char": ch, "source": os.path.basename(path)}
        # contact sheet for checking
        if glyphs:
            th = [cv2.resize((1 - g) * 255, (120, 120), interpolation=cv2.INTER_AREA) for g in glyphs.values()]
            th += [np.full((120, 120), 255, np.uint8)] * (-len(th) % 12)
            rows = [np.hstack(th[r:r + 12]) for r in range(0, len(th), 12)]
            cv2.imwrite(os.path.join(a.debug, f"ingest-{os.path.splitext(os.path.basename(path))[0]}.png"), np.vstack(rows))
        print(f"{path}: page {page + 1}, {nmark} markers, {len(glyphs)}/{len(chars)} boxes written, scale x{f:.2f}")
        total += len(glyphs)
    with open(src_path, "w", encoding="utf-8") as fp:
        json.dump(sources, fp, ensure_ascii=False, indent=1)
    print(f"done: {total} glyphs in {a.glyphs}. Now run: python3 scripts/build_font.py")


if __name__ == "__main__":
    main()
