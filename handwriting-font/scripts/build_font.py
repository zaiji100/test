"""Build the handwriting font.

1. Every glyph of an open-source base font (Long Cang, OFL 1.1, covers all 6763
   GB2312 hanzi) is re-written in the style measured from the sample:
     - narrow & tall proportions  (sample width/height ~0.6-0.8)
     - horizontals rising to the right (shear)
     - per-glyph jitter (rotation / size / baseline) and a smooth wobble
     - skeletonised and re-drawn with a thin round ballpoint pen
       (stroke ~3% of character height, slight pressure variation)
2. Real handwritten glyphs in glyphs/uXXXX.png (from the photo, or from filled-in
   template sheets) replace the generated ones.
3. Output: dist/<PSName>.ttf (+ .woff2 when brotli is installed).
"""
import argparse, glob, json, math, os, sys, time, urllib.request, zlib
from multiprocessing import Pool

import cv2
import numpy as np
from fontTools.ttLib import TTFont, newTable
from fontTools.pens.freetypePen import FreeTypePen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from skimage.morphology import skeletonize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import hwlib  # noqa: E402

BASE_URL = "https://raw.githubusercontent.com/google/fonts/main/ofl/longcang/LongCang-Regular.ttf"

FAMILY_EN = "BieXianQi Hand"
FAMILY_ZH = "别嫌弃手写体"
PS_NAME = "BieXianQiHand-Regular"
VERSION = "1.000"

STYLE = dict(
    adv=940,                  # advance width of full-width (CJK) glyphs (sample writes tight)
    sx=0.80, sy=1.13,         # global scale: narrow & tall like the sample
    src_cx=500, src_cy=300,   # visual centre of the base font's hanzi
    cy=375,                   # visual centre of the new hanzi
    rise_deg=9.0,             # horizontals rise to the right
    lean_deg=3.0,             # verticals lean slightly to the right
    rot_sd=1.3,               # per-glyph rotation jitter (deg)
    scale_sd=0.03,            # per-glyph size jitter
    dx_sd=10, dy_sd=16,       # per-glyph position jitter (units)
    warp_amp=14, warp_scale=420,  # smooth wobble: amplitude / wavelength (units)
    pen=30, pen_var=0.15,     # pen width (units) and pressure variation
    overshoot=38,             # stroke ends run on by up to this much (units)
    latin_sx=0.95, latin_sy=1.0,  # half-width glyphs (Latin, digits, ASCII punctuation):
    latin_rise_deg=2.0,           #   gentler shear, wobble and overshoot, since they are
    latin_warp_amp=6,             #   much smaller than a hanzi
    latin_overshoot=10,
    ppu=0.6,                  # raster resolution (pixels per unit)
    spur=30,                  # prune skeleton side-branches shorter than this (units)
)


# ------------------------------------------------------------------ helpers
def affine(st, rng, is_full, adv0):
    """Forward 3x3 matrix: base-font units -> new-font units, plus new advance."""
    if is_full:
        adv = st["adv"]
        rot = math.radians(rng.normal(0, st["rot_sd"]))
        sc = 1 + rng.normal(0, st["scale_sd"])
        dx, dy = rng.normal(0, st["dx_sd"]), rng.normal(0, st["dy_sd"])
        C = np.array([[1, 0, -st["src_cx"]], [0, 1, -st["src_cy"]], [0, 0, 1]], float)
        S = np.diag([st["sx"] * sc, st["sy"] * sc, 1.0])
        T = np.array([[1, 0, adv / 2 + dx], [0, 1, st["cy"] + dy], [0, 0, 1]], float)
        cx = adv / 2
    else:
        adv = int(round(adv0 * st["latin_sx"]))
        rot = math.radians(rng.normal(0, st["rot_sd"] * 0.6))
        C = np.eye(3)
        S = np.diag([st["latin_sx"], st["latin_sy"], 1.0])
        T = np.eye(3)
        cx = adv / 2
    R = np.array([[math.cos(rot), -math.sin(rot), 0], [math.sin(rot), math.cos(rot), 0], [0, 0, 1]])
    k = math.tan(math.radians(st["rise_deg"]))
    Sh = np.array([[1, 0, 0], [k, 1, -k * cx], [0, 0, 1]], float)
    j = math.tan(math.radians(st["lean_deg"]))
    cy = st["cy"] if is_full else 300
    Ln = np.array([[1, j, -j * cy], [0, 1, 0], [0, 0, 1]], float)
    return Ln @ Sh @ T @ R @ S @ C, adv


_G = {}


def _init(base_path, st):
    f = TTFont(base_path)
    _G["gs"] = f.getGlyphSet()
    _G["glyf"] = f["glyf"]
    _G["hmtx"] = f["hmtx"].metrics
    _G["st"] = st


def process(job):
    """Re-write one base glyph in the sample's style. Returns (name, glyph, adv)."""
    name, is_full = job
    st, gs = _G["st"], _G["gs"]
    if not is_full:
        st = dict(st, rise_deg=st["latin_rise_deg"], warp_amp=st["latin_warp_amp"],
                  overshoot=st["latin_overshoot"])
    adv0 = _G["hmtx"][name][0]
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    F, adv = affine(st, rng, is_full, adv0)
    g0 = _G["glyf"][name]
    if not hasattr(g0, "xMin") or g0.numberOfContours == 0:
        return name, TTGlyphPen(None).glyph(), adv

    # 1. rasterise the base glyph
    pb = st["ppu"]
    bx0, by0 = g0.xMin - 60, g0.yMin - 60
    bw, bh = int((g0.xMax - g0.xMin + 120) * pb) + 1, int((g0.yMax - g0.yMin + 120) * pb) + 1
    pen = FreeTypePen(gs)
    gs[name].draw(pen)
    base = pen.array(width=bw, height=bh, transform=(pb, 0, 0, pb, -bx0 * pb, -by0 * pb),
                     contain=False, evenOdd=False).astype(np.float32)

    # 2. output canvas = bounds of the transformed glyph (+margin)
    corners = np.array([[g0.xMin, g0.yMin, 1], [g0.xMax, g0.yMin, 1],
                        [g0.xMin, g0.yMax, 1], [g0.xMax, g0.yMax, 1]], float) @ F.T
    m = 60 + st["warp_amp"] * 2 + st["pen"] + st["overshoot"]
    X0, X1 = corners[:, 0].min() - m, corners[:, 0].max() + m
    Y0, Y1 = corners[:, 1].min() - m, corners[:, 1].max() + m
    p = st["ppu"]
    W, H = int((X1 - X0) * p) + 1, int((Y1 - Y0) * p) + 1
    cols, rows = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    X = X0 + (cols + 0.5) / p
    Y = Y1 - (rows + 0.5) / p
    # smooth wobble (inverse-mapped)
    wsc = st["warp_scale"] * p
    X = X - st["warp_amp"] * hwlib.smooth_noise((H, W), wsc, rng)
    Y = Y - st["warp_amp"] * hwlib.smooth_noise((H, W), wsc, rng)
    Fi = np.linalg.inv(F)
    x = Fi[0, 0] * X + Fi[0, 1] * Y + Fi[0, 2]
    y = Fi[1, 0] * X + Fi[1, 1] * Y + Fi[1, 2]
    mapx = ((x - bx0) * pb - 0.5).astype(np.float32)
    mapy = (bh - (y - by0) * pb - 0.5).astype(np.float32)
    warped = cv2.remap(base, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    binm = warped > 0.5
    if binm.sum() < 3:
        return name, TTGlyphPen(None).glyph(), adv

    # 3. skeleton -> prune -> thin ballpoint pen
    sk = skeletonize(binm)
    sk = hwlib.prune_spurs(sk, int(st["spur"] * p))
    if st["overshoot"] > 0:
        sk = hwlib.extend_ends(sk, st["overshoot"] * p, rng)
    ink = hwlib.restroke(sk, st["pen"] / 2 * p, st["pen_var"], rng, noise_scale_px=wsc / 2)

    # 4. trace
    glyph = hwlib.bitmap_to_glyph(ink, X0, Y1, p)
    return name, glyph, adv


def load_real_glyphs(folder, adv):
    """glyphs/uXXXX.png (canonical canvas) -> {codepoint: glyph}"""
    out = {}
    for path in sorted(glob.glob(os.path.join(folder, "u*.png"))):
        cp = int(os.path.basename(path)[1:].split(".")[0], 16)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        ink = (img < 128).astype(np.uint8)
        p = img.shape[1] / hwlib.CANVAS_SIZE
        x0 = hwlib.CANVAS_X0 + (adv - 1000) / 2
        out[cp] = hwlib.bitmap_to_glyph(ink, x0, hwlib.CANVAS_YTOP, p, blur=1.3, eps=0.35, min_area_px=10)
    return out


def set_names(font):
    name = font["name"]
    name.names = []
    lic = ("This Font Software is licensed under the SIL Open Font License, Version 1.1. "
           "This license is available with a FAQ at: https://openfontlicense.org")
    cr = ("Copyright 2018 The Long Cang Project Authors (https://github.com/googlefonts/longcang). "
          "Modified: re-drawn in the style of a handwriting sample; sample glyphs added.")
    en = {0: cr, 1: FAMILY_EN, 2: "Regular", 3: f"{VERSION};NONE;{PS_NAME}", 4: f"{FAMILY_EN} Regular",
          5: f"Version {VERSION}", 6: PS_NAME, 13: lic, 14: "https://openfontlicense.org"}
    for nid, s in en.items():
        name.setName(s, nid, 3, 1, 0x409)
        name.setName(s, nid, 1, 0, 0)
    for nid, s in {1: FAMILY_ZH, 2: "Regular", 4: f"{FAMILY_ZH} Regular"}.items():
        name.setName(s, nid, 3, 1, 0x804)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=os.path.join(ROOT, "build", "LongCang-Regular.ttf"))
    ap.add_argument("--glyphs", default=os.path.join(ROOT, "glyphs"), help="folder with real glyph PNGs")
    ap.add_argument("--out", default=os.path.join(ROOT, "dist"))
    ap.add_argument("--chars", help="only build these characters (quick test)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--no-real", action="store_true", help="do not merge real handwritten glyphs")
    ap.add_argument("--style", help="JSON with STYLE overrides, e.g. '{\"pen\":30}'")
    a = ap.parse_args()

    st = dict(STYLE)
    if a.style:
        st.update(json.loads(a.style))
    if not os.path.exists(a.base):
        os.makedirs(os.path.dirname(a.base), exist_ok=True)
        print("downloading base font", BASE_URL)
        urllib.request.urlretrieve(BASE_URL, a.base)

    font = TTFont(a.base)
    cmap = font.getBestCmap()
    rev = {}
    for cp, gn in cmap.items():
        rev.setdefault(gn, cp)
    hmtx = font["hmtx"].metrics
    order = font.getGlyphOrder()
    if a.chars:
        names = [cmap[ord(c)] for c in dict.fromkeys(a.chars) if ord(c) in cmap]
    else:
        names = [n for n in order if n != ".notdef"]
    jobs = [(n, hmtx[n][0] >= 900) for n in names]

    t0 = time.time()
    results = []
    with Pool(a.jobs, initializer=_init, initargs=(a.base, st)) as pool:
        for i, r in enumerate(pool.imap_unordered(process, jobs, chunksize=16)):
            results.append(r)
            if (i + 1) % 500 == 0:
                el = time.time() - t0
                print(f"  {i + 1}/{len(jobs)} glyphs  {el:.0f}s  (eta {el / (i + 1) * (len(jobs) - i - 1):.0f}s)", flush=True)
    print(f"styled {len(results)} glyphs in {time.time() - t0:.0f}s")

    glyf = font["glyf"]
    for gname, g, adv in results:
        glyf[gname] = g
        g.recalcBounds(glyf)
        hmtx[gname] = (adv, getattr(g, "xMin", 0))

    real = {} if a.no_real else load_real_glyphs(a.glyphs, st["adv"])
    cmap_tables = [t for t in font["cmap"].tables if t.isUnicode()]
    for cp, g in real.items():
        gname = cmap.get(cp)
        if gname is None:
            gname = f"uni{cp:04X}" if cp <= 0xFFFF else f"u{cp:05X}"
            font.setGlyphOrder(font.getGlyphOrder() + [gname])
            for t in cmap_tables:
                if cp <= 0xFFFF or t.format in (12, 13):
                    t.cmap[cp] = gname
        glyf[gname] = g
        g.recalcBounds(glyf)
        hmtx[gname] = (st["adv"], getattr(g, "xMin", 0))
    print(f"merged {len(real)} real handwritten glyphs: {''.join(chr(c) for c in sorted(real))}")

    if a.chars:  # subset test font: keep only what was built
        from fontTools import subset
        keep = set(a.chars) | set(chr(c) for c in real)
        opts = subset.Options()
        opts.name_IDs = ["*"]
        opts.notdef_outline = True
        sub = subset.Subsetter(opts)
        sub.populate(text="".join(keep))
        sub.subset(font)

    # tables that no longer match the new outlines
    for tag in ("DSIG", "GPOS", "prep", "fpgm", "cvt ", "hdmx", "LTSH", "VDMX"):
        if tag in font:
            del font[tag]
    gasp = newTable("gasp")
    gasp.version = 1
    gasp.gaspRange = {0xFFFF: 0x000F}
    font["gasp"] = gasp
    maxp = font["maxp"]
    for k in ("maxZones", "maxTwilightPoints", "maxStorage", "maxFunctionDefs",
              "maxInstructionDefs", "maxStackElements", "maxSizeOfInstructions"):
        setattr(maxp, k, 1 if k == "maxZones" else 0)

    # vertical metrics: line spacing from typo metrics, no clipping on Windows
    ymin = min(getattr(glyf[n], "yMin", 0) for n in font.getGlyphOrder())
    ymax = max(getattr(glyf[n], "yMax", 0) for n in font.getGlyphOrder())
    os2, hhea = font["OS/2"], font["hhea"]
    os2.sTypoAscender, os2.sTypoDescender, os2.sTypoLineGap = 880, -120, 300
    os2.fsSelection |= 1 << 7  # USE_TYPO_METRICS
    os2.usWinAscent, os2.usWinDescent = int(max(ymax, 1000)), int(max(-ymin, 300))
    hhea.ascent, hhea.descent, hhea.lineGap = 1030, -270, 0
    os2.achVendID = "NONE"
    font["head"].fontRevision = float(VERSION)
    set_names(font)
    if "post" in font:
        font["post"].formatType = 2.0

    os.makedirs(a.out, exist_ok=True)
    stem = PS_NAME + ("-test" if a.chars else "")
    ttf = os.path.join(a.out, stem + ".ttf")
    font.save(ttf)
    print("wrote", ttf, f"{os.path.getsize(ttf) / 1e6:.1f} MB")
    try:
        import brotli  # noqa: F401
        font.flavor = "woff2"
        w2 = os.path.join(a.out, stem + ".woff2")
        font.save(w2)
        print("wrote", w2, f"{os.path.getsize(w2) / 1e6:.1f} MB")
    except ImportError:
        print("(pip install brotli to also get a .woff2 web font)")


if __name__ == "__main__":
    main()
