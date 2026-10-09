"""Debug drawing of an extracted table on a copy of its page image (Pillow). Development aid only."""
from .common import OutputDirectoryError

TABLE_IMAGE_PATTERN = "page_{:03d}_table_{:03d}_debug.png"

TABLE_COLOR = (20, 140, 60)         # outline of the table region
BOUNDARY_COLOR = (220, 120, 0)      # row and column boundaries
CELL_COLOR = (30, 100, 200)         # cell boxes
HEADER_COLOR = (200, 30, 30)        # cell boxes of header rows


def table_image_name(page_number, table_number):
    return TABLE_IMAGE_PATTERN.format(page_number, table_number)


def grid_boundaries(table):
    """Row and column boundaries of a table in page pixels, from its cell boxes: (ys, xs).

    A boundary lies halfway between the cells of one row (column) and the next. The outer boundaries
    are the edges of the table region. Only cells that span a single row (column) are used.
    """
    def boundaries(index_key, span_key, low, high, n, start, end):
        extents = [[None, None] for _ in range(n)]
        for cell in table["cells"]:
            if cell[span_key] != 1:
                continue
            extent = extents[cell[index_key]]
            extent[0] = cell["image_bbox"][low] if extent[0] is None else min(extent[0], cell["image_bbox"][low])
            extent[1] = cell["image_bbox"][high] if extent[1] is None else max(extent[1], cell["image_bbox"][high])
        lines = [start]
        for before, after in zip(extents, extents[1:]):
            if before[1] is not None and after[0] is not None:
                lines.append((before[1] + after[0]) / 2)
        return lines + [end]

    x0, y0, x1, y1 = table["image_bbox"]
    return (boundaries("row", "row_span", 1, 3, table["n_rows"], y0, y1),
            boundaries("column", "col_span", 0, 2, table["n_columns"], x0, x1))


def draw_table(page_image, table, output_path):
    """Save a copy of the page image (PIL) with the table box, the row and column boundaries and every
    cell box of `table` (an entry of tables.json) drawn on it. The page image is not modified."""
    from PIL import ImageDraw

    image = page_image.convert("RGB")
    draw = ImageDraw.Draw(image)
    line = max(2, round(image.width / 700))
    x0, y0, x1, y1 = table["image_bbox"]

    ys, xs = grid_boundaries(table)
    for y in ys[1:-1]:
        draw.line([x0, y, x1, y], fill=BOUNDARY_COLOR, width=line)
    for x in xs[1:-1]:
        draw.line([x, y0, x, y1], fill=BOUNDARY_COLOR, width=line)
    for cell in table["cells"]:
        draw.rectangle(cell["image_bbox"], outline=HEADER_COLOR if cell["is_header"] else CELL_COLOR, width=1)
    draw.rectangle([x0, y0, x1, y1], outline=TABLE_COLOR, width=line + 1)

    try:
        image.save(output_path, format="PNG")
    except OSError as exc:
        raise OutputDirectoryError(f"Cannot write table image: {output_path} ({exc})") from exc
    return output_path
