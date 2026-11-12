"""Geometry of the A4 handwriting template, shared by make_template / ingest_sheets.
All values in millimetres, origin top-left of the page."""

PAGE_W, PAGE_H = 210.0, 297.0
MARKER = 12.0                      # ArUco marker side
MARKER_IDS = (0, 1, 2, 3)          # TL, TR, BL, BR
PAGE_ID_BASE = 10                  # page n carries marker id PAGE_ID_BASE + n
MARKER_POS = {                     # top-left corner of each marker
    0: (8.0, 8.0),
    1: (PAGE_W - 8.0 - MARKER, 8.0),
    2: (8.0, PAGE_H - 8.0 - MARKER),
    3: (PAGE_W - 8.0 - MARKER, PAGE_H - 8.0 - MARKER),
}
PAGE_MARKER_POS = ((PAGE_W - MARKER) / 2, PAGE_H - 8.0 - MARKER)

COLS, ROWS = 12, 13
GRID_X0, GRID_Y0 = 15.0, 30.0
PITCH_X = 15.0                     # 12 * 15 = 180 mm
LABEL_H = 4.0                      # printed hint above each box
BOX = 13.5                         # writing square = the font's em box
PITCH_Y = LABEL_H + BOX + 1.0      # 13 * 18.5 = 240.5 mm
PER_PAGE = COLS * ROWS


def cell_box(i):
    """Writing square of cell i on a page: (x, y, size) in mm."""
    r, c = divmod(i, COLS)
    x = GRID_X0 + c * PITCH_X + (PITCH_X - BOX) / 2
    y = GRID_Y0 + r * PITCH_Y + LABEL_H
    return x, y, BOX


def marker_corners_mm(mid, page=None):
    """4 corners (TL, TR, BR, BL) of a marker, in mm."""
    if mid in MARKER_POS:
        x, y = MARKER_POS[mid]
    else:
        x, y = PAGE_MARKER_POS
    return [(x, y), (x + MARKER, y), (x + MARKER, y + MARKER), (x, y + MARKER)]
