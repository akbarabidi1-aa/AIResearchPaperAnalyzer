"""Phase 8 reading order: the blocks of one page -> its columns -> the order in which they are read.

    boxes of the body blocks  ->  detect_columns   gutters between blocks that stand side by side
                              ->  order_blocks     bands between full-width blocks; inside a band the
                                                   columns left to right, each from top to bottom

Geometry only. Nothing here looks at text, and nothing sorts a page by "y, then x": on a two-column
page that would read across the gutter.
"""
LAYOUT_SINGLE = "single_column"
LAYOUT_TWO = "two_column"
LAYOUT_MULTI = "multi_column"       # three or more columns
LAYOUT_UNKNOWN = "unknown"          # blocks on both sides of the page that do not form columns
LAYOUTS = (LAYOUT_SINGLE, LAYOUT_TWO, LAYOUT_MULTI, LAYOUT_UNKNOWN)

MIN_VOTER_WIDTH = 0.05      # of the page width; narrower blocks (equation numbers) do not shape columns
GUTTER_SEARCH = (0.2, 0.8)  # a gutter lies in this part of the width that is being split
MIN_GUTTER = 4.0            # points of free space a gutter must have
SPAN_TOLERANCE = 3.0        # points a block must reach beyond a gutter on both sides to span it
SAME_ROW_OVERLAP = 0.5      # two blocks are side by side when they share this much of the lower one's height


def _overlap(a0, a1, b0, b1):
    return min(a1, b1) - max(a0, b0)


def _find_gutter(boxes, low, high, min_side=1):
    """The x position of a gutter in [low, high], or None.

    A position qualifies when at least `min_side` blocks lie wholly on each side of it and two of
    them, one from each side, stand side by side. Blocks that reach across a position are full-width
    blocks there. Among the qualifying positions the gutter is where the fewest blocks reach across
    (only the real full-width blocks, not the paragraphs of a column); of several such bands the
    widest one, which must be at least MIN_GUTTER wide. The gutter is the middle of the band."""
    x, end = low + GUTTER_SEARCH[0] * (high - low), low + GUTTER_SEARCH[1] * (high - low)
    positions = []
    while x <= end + 0.5:
        left = [b for b in boxes if b[2] <= x]
        right = [b for b in boxes if b[0] >= x]
        if len(left) >= min_side and len(right) >= min_side and any(
                _overlap(a[1], a[3], b[1], b[3]) > 0 for a in left for b in right):
            positions.append((x, len(boxes) - len(left) - len(right)))
        x += 1.0
    if not positions:
        return None
    fewest = min(crossing for _, crossing in positions)
    bands, band = [], []
    for x, crossing in positions:
        if crossing == fewest and (not band or x - band[-1] <= 1.0):
            band.append(x)
        else:
            bands.append(band)
            band = [x] if crossing == fewest else []
    bands.append(band)
    widest = max(bands, key=lambda band: band[-1] - band[0] if band else -1.0)
    if not widest or widest[-1] - widest[0] < MIN_GUTTER:
        return None
    return (widest[0] + widest[-1]) / 2


def detect_columns(boxes, page_width):
    """Columns of a page from the boxes [x0, y0, x1, y1] of its body blocks (headers, footers and page
    numbers left out by the caller).

    Returns {"layout": "single_column" | "two_column" | "multi_column" | "unknown",
             "columns": [[x0, x1], ...] left to right, "gutters": [x, ...]}.

    A gutter is a vertical band of free space, in the middle part of the page, with blocks on both of
    its sides that stand side by side (they overlap vertically). Blocks that cross the band are
    full-width blocks and do not count against it. Each side is searched again for a further gutter
    (three or more columns), which then needs two blocks on each side.

      no gutter, no block confined to the right half     -> single_column
      no gutter, but blocks confined to each half         -> unknown (they are not side by side:
                                                             a column structure cannot be told)
      one gutter -> two_column; more -> multi_column
    An empty page is single_column with no columns.
    """
    voters = [tuple(float(v) for v in box) for box in boxes if box[2] - box[0] >= MIN_VOTER_WIDTH * page_width]
    if not voters:
        return {"layout": LAYOUT_SINGLE, "columns": [], "gutters": []}
    low, high = min(b[0] for b in voters), max(b[2] for b in voters)

    def split(low, high, min_side):
        inside = [b for b in voters if b[0] >= low - SPAN_TOLERANCE and b[2] <= high + SPAN_TOLERANCE]
        gutter = _find_gutter(inside, low, high, min_side)
        if gutter is None:
            return []
        left_edge = max(b[2] for b in inside if b[2] <= gutter)
        right_edge = min(b[0] for b in inside if b[0] >= gutter)
        return split(low, left_edge, 2) + [gutter] + split(right_edge, high, 2)

    gutters = split(low, high, 1)
    if not gutters:
        middle = (low + high) / 2
        sided = (any(b[2] <= middle for b in voters), any(b[0] >= middle for b in voters))
        layout = LAYOUT_UNKNOWN if all(sided) else LAYOUT_SINGLE
        return {"layout": layout, "columns": [[low, high]], "gutters": []}
    columns = []
    for index in range(len(gutters) + 1):
        left = gutters[index - 1] if index else low - 1
        right = gutters[index] if index < len(gutters) else high + 1
        members = [b for b in voters if b[0] >= left - SPAN_TOLERANCE and b[2] <= right + SPAN_TOLERANCE]
        columns.append([min(b[0] for b in members), max(b[2] for b in members)] if members else [left, right])
    return {"layout": LAYOUT_TWO if len(gutters) == 1 else LAYOUT_MULTI, "columns": columns, "gutters": gutters}


def spans_columns(box, gutters):
    """True for a box that reaches across a gutter: a full-width block."""
    return any(box[0] < gutter - SPAN_TOLERANCE and box[2] > gutter + SPAN_TOLERANCE for gutter in gutters)


def column_index(box, gutters):
    """Index of the column a box stands in: the number of gutters left of its centre."""
    centre = (box[0] + box[2]) / 2
    return sum(1 for gutter in gutters if gutter < centre)


def top_down(boxes, indices):
    """`indices` of `boxes` from top to bottom; blocks side by side (an equation and its number) left
    to right."""
    ordered = sorted(indices, key=lambda i: (boxes[i][1], boxes[i][0]))
    for position in range(len(ordered) - 1):
        a, b = boxes[ordered[position]], boxes[ordered[position + 1]]
        shared = _overlap(a[1], a[3], b[1], b[3])
        if shared >= SAME_ROW_OVERLAP * min(a[3] - a[1], b[3] - b[1]) and b[0] < a[0]:
            ordered[position], ordered[position + 1] = ordered[position + 1], ordered[position]
    return ordered


def order_blocks(boxes, layout):
    """Reading order of the boxes of one page, as a list of indices into `boxes`. `layout` is a
    result of detect_columns.

    Blocks that span a gutter (title, wide figure, wide table, a heading across the page) are put in
    order from top to bottom. They cut the page into bands. A column block belongs to the band its
    vertical centre lies in. The order is: the first band, column by column from left to right and
    from top to bottom inside each column; then the full-width block below it; then the next band.
    So text above a wide figure is read in both columns before the figure, and text below it after.

    Without gutters (single_column, unknown) this is one column: top to bottom.
    """
    gutters = layout["gutters"]
    everything = list(range(len(boxes)))
    spanning = top_down(boxes, [i for i in everything if spans_columns(boxes[i], gutters)])
    cuts = [(boxes[i][1] + boxes[i][3]) / 2 for i in spanning]
    bands = [[[] for _ in range(len(gutters) + 1)] for _ in range(len(spanning) + 1)]
    for i in everything:
        if i in spanning:
            continue
        centre = (boxes[i][1] + boxes[i][3]) / 2
        bands[sum(1 for cut in cuts if cut < centre)][column_index(boxes[i], gutters)].append(i)
    order = []
    for band, columns in enumerate(bands):
        for members in columns:
            order.extend(top_down(boxes, members))
        if band < len(spanning):
            order.append(spanning[band])
    return order


def pairwise_order_accuracy(expected, predicted):
    """Share of the pairs of `expected` (ids in the right order) that `predicted` has in the same
    order. A pair with an id that `predicted` lacks counts as wrong. None for fewer than two ids."""
    position = {key: index for index, key in enumerate(predicted)}
    pairs = correct = 0
    for i, first in enumerate(expected):
        for second in expected[i + 1:]:
            pairs += 1
            correct += first in position and second in position and position[first] < position[second]
    return correct / pairs if pairs else None


def exact_order(expected, predicted):
    """True when `predicted`, reduced to the ids of `expected`, is exactly `expected`."""
    wanted = set(expected)
    return [key for key in predicted if key in wanted] == list(expected)
