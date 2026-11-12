"""Shared helpers: bitmap <-> TrueType outlines, skeleton pruning, pen re-stroking."""
import numpy as np
import cv2

# Canonical real-glyph PNG canvas, in font units: X from CANVAS_X0, Y from
# CANVAS_YTOP downwards, CANVAS_SIZE units square. The em box (0..1000,
# -120..880) sits in the middle.
CANVAS_X0, CANVAS_YTOP, CANVAS_SIZE = -400, 1280, 1800
from skimage import measure
from skimage.morphology import skeletonize
from fontTools.pens.ttGlyphPen import TTGlyphPen


# ---------------------------------------------------------------- vectorize
def bitmap_to_glyph(bm, x0, ytop, ppu, blur=0.9, eps=0.28, min_area_px=6.0):
    """Trace a binary bitmap (1 = ink, row 0 = top) into a TrueType glyph.

    x0, ytop : font-unit coordinates of the top-left pixel corner
    ppu      : pixels per font unit
    Outline = iso-line 0.5 of the lightly blurred bitmap (sub-pixel, smooth),
    simplified with Douglas-Peucker and emitted as a closed quadratic B-spline
    (all control points off-curve, one explicit on-curve start point).
    """
    img = np.pad(bm.astype(np.float32), 3)
    if blur > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    contours = measure.find_contours(img, 0.5)
    polys = []
    for c in contours:
        if len(c) < 8:
            continue
        rows, cols = c[:, 0] - 3, c[:, 1] - 3
        X = x0 + (cols + 0.5) / ppu
        Y = ytop - (rows + 0.5) / ppu
        p = np.stack([X, Y], 1)
        if np.allclose(p[0], p[-1]):
            p = p[:-1]
        area = 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
        if abs(area) * ppu * ppu < min_area_px:
            continue
        # simplify in pixel space
        approx = cv2.approxPolyDP((p * ppu).astype(np.float32).reshape(-1, 1, 2), eps, True)
        q = approx.reshape(-1, 2) / ppu
        if len(q) < 3:
            continue
        polys.append((area, q))
    if not polys:
        return TTGlyphPen(None).glyph()
    # all islands share one winding, all holes the other; make islands clockwise
    biggest = max(polys, key=lambda t: abs(t[0]))
    flip = biggest[0] > 0
    pen = TTGlyphPen(None)
    for area, q in polys:
        if flip:
            q = q[::-1]
        pts = [(int(round(x)), int(round(y))) for x, y in q]
        # drop consecutive duplicates created by rounding
        clean = [pts[0]]
        for pt in pts[1:]:
            if pt != clean[-1]:
                clean.append(pt)
        if clean[0] == clean[-1]:
            clean.pop()
        if len(clean) < 3:
            continue
        a, b = clean[-1], clean[0]
        start = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
        pen.moveTo(start)
        pen.qCurveTo(*clean, start)
        pen.closePath()
    return pen.glyph()


# ---------------------------------------------------------------- skeleton
_K = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)


def _neighbors(sk):
    return cv2.filter2D(sk.astype(np.uint8), -1, _K, borderType=cv2.BORDER_CONSTANT)


def prune_spurs(sk, max_len, rounds=2):
    """Remove short side branches (spurs) that end in a junction."""
    sk = sk.astype(bool).copy()
    H, W = sk.shape
    offs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for _ in range(rounds):
        nb = _neighbors(sk)
        ends = np.argwhere(sk & (nb == 1))
        removed = False
        for y, x in ends:
            path = [(y, x)]
            prev = None
            cy, cx = y, x
            hit_junction = False
            while len(path) <= max_len:
                nxt = []
                for dy, dx in offs:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < H and 0 <= nx < W and sk[ny, nx] and (ny, nx) != prev and (ny, nx) not in path:
                        nxt.append((ny, nx))
                if not nxt:
                    break  # isolated segment: keep it (dots etc.)
                if len(nxt) > 1 or nb[nxt[0]] >= 3:
                    hit_junction = True
                    break
                prev = (cy, cx)
                cy, cx = nxt[0]
                path.append((cy, cx))
            if hit_junction and len(path) <= max_len:
                for p in path:
                    sk[p] = False
                removed = True
        if not removed:
            break
    return skeletonize(sk)


def extend_ends(sk, max_px, rng, look=8, min_px=0.0):
    """Overshoot: prolong every free stroke end along its own direction by a
    random length, like a quick hand that does not stop exactly on target."""
    sk = sk.astype(np.uint8).copy()
    H, W = sk.shape
    nb = _neighbors(sk)
    ends = np.argwhere((sk > 0) & (nb == 1))
    offs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for y, x in ends:
        path = [(y, x)]
        cy, cx = y, x
        for _ in range(look):
            nxt = [(cy + dy, cx + dx) for dy, dx in offs
                   if 0 <= cy + dy < H and 0 <= cx + dx < W and sk[cy + dy, cx + dx] and (cy + dy, cx + dx) not in path]
            if len(nxt) != 1:
                break
            cy, cx = nxt[0]
            path.append((cy, cx))
        if len(path) < 4:
            continue
        v = np.array([y - path[-1][0], x - path[-1][1]], float)
        v /= np.linalg.norm(v) + 1e-9
        L = rng.uniform(min_px, max_px)
        ey, ex = y + v[0] * L, x + v[1] * L
        cv2.line(sk, (int(x), int(y)), (int(round(ex)), int(round(ey))), 1, 1)
    return sk.astype(bool)


def smooth_noise(shape, scale_px, rng, octaves=2):
    """Low-frequency noise in [-1, 1] (sum of upsampled random grids)."""
    H, W = shape
    out = np.zeros(shape, np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        s = max(2, int(scale_px / (2 ** o)))
        gh, gw = H // s + 3, W // s + 3
        g = rng.standard_normal((gh, gw)).astype(np.float32)
        up = cv2.resize(g, (gw * s, gh * s), interpolation=cv2.INTER_CUBIC)[:H, :W]
        out += amp * up
        total += amp
        amp *= 0.5
    out /= total
    m = np.abs(out).max() or 1.0
    return out / m


def restroke(skel, radius_px, var=0.0, rng=None, noise_scale_px=60):
    """Draw a skeleton with a round pen of (optionally varying) radius."""
    dist = cv2.distanceTransform((~skel.astype(bool)).astype(np.uint8), cv2.DIST_L2, 5)
    if var > 0 and rng is not None:
        r = radius_px * (1.0 + var * smooth_noise(skel.shape, noise_scale_px, rng))
    else:
        r = radius_px
    return (dist <= r).astype(np.uint8)
