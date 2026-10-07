"""Debug drawing of layout regions on a copy of the page image (Pillow). Development aid only."""
from .common import OutputDirectoryError
from .layout import RegionType

LAYOUT_IMAGE_PATTERN = "page_{:03d}_layout.png"

COLORS = {
    RegionType.TITLE: (200, 30, 30),
    RegionType.TEXT: (30, 100, 200),
    RegionType.TABLE: (20, 140, 60),
    RegionType.FIGURE: (150, 50, 180),
    RegionType.EQUATION: (220, 120, 0),
    RegionType.CAPTION: (0, 150, 160),
    RegionType.HEADER: (120, 120, 120),
    RegionType.FOOTER: (120, 120, 120),
    RegionType.PAGE_NUMBER: (90, 90, 90),
    RegionType.LIST: (140, 100, 40),
    RegionType.OTHER: (0, 0, 0),
}


def layout_image_name(page_number):
    return LAYOUT_IMAGE_PATTERN.format(page_number)


def draw_layout(image_path, regions, output_path):
    """Save a copy of the page image with every region's box, canonical class and score drawn on it.

    The page image itself is not modified. Returns output_path.
    """
    from PIL import Image, ImageDraw, ImageFont     # only needed for this optional step

    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    line = max(2, round(image.width / 500))
    try:
        font = ImageFont.load_default(size=max(12, round(image.width / 70)))
    except TypeError:                               # Pillow < 10.1 has no sized default font
        font = ImageFont.load_default()

    for region in regions:
        color = COLORS[region.type]
        x0, y0, x1, y1 = region.image_bbox
        draw.rectangle([x0, y0, x1, y1], outline=color, width=line)
        label = f"{region.type.value} {region.score:.2f}"
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        width, height = right - left + 6, bottom - top + 6
        # Above the box when there is room, otherwise inside its top edge.
        label_y = y0 - height if y0 >= height else y0
        draw.rectangle([x0, label_y, x0 + width, label_y + height], fill=color)
        draw.text((x0 + 3 - left, label_y + 3 - top), label, fill=(255, 255, 255), font=font)

    try:
        image.save(output_path, format="PNG")
    except OSError as exc:
        raise OutputDirectoryError(f"Cannot write layout image: {output_path} ({exc})") from exc
    return output_path
